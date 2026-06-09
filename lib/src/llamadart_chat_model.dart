import 'dart:async';
import 'dart:convert';
import 'package:dartantic_interface/dartantic_interface.dart';
import 'package:llamadart/llamadart.dart';
import 'package:meta/meta.dart';

import 'llamadart_chat_options.dart';
import 'llamadart_provider.dart';

class LlamadartChatModel extends ChatModel<LlamadartChatOptions> {
  final LlamadartProvider provider;
  // ignore: overridden_fields, annotate_overrides
  final List<Tool<Object>>? tools;

  LlamaEngine? _engine;
  ChatSession? _session;

  LlamadartChatModel({
    required this.provider,
    required super.name,
    this.tools,
    required super.defaultOptions,
  });

  Future<void> _ensureInitialized() async {
    if (_engine != null) return;

    final backend = LlamaBackend();
    _engine = LlamaEngine(backend);

    await _engine!.loadModel(
      provider.modelPath,
      modelParams: ModelParams(
        contextSize: defaultOptions.nCtx ?? 8192,
        gpuLayers: defaultOptions.nGpuLayers ?? ModelParams.maxGpuLayers,
        preferredBackend: defaultOptions.preferredBackend,
        liteRtLmBackend: defaultOptions.liteRtLmBackend,
        chatTemplate: defaultOptions.chatTemplate,
      ),
    );

    _session = ChatSession(_engine!);
  }

  Future<ChatFormat> _getChatFormat() async {
    await _ensureInitialized();
    final metadata = await _engine!.getMetadata();
    final template = metadata['tokenizer.chat_template'];
    return ChatTemplateEngine.detectFormat(template);
  }

  @override
  Stream<ChatResult<ChatMessage>> sendStream(
    List<ChatMessage> messages, {
    LlamadartChatOptions? options,
    Schema? outputSchema,
  }) async* {
    await _ensureInitialized();

    final format = await _getChatFormat();
    final hasTools =
        outputSchema != null || (tools != null && tools!.isNotEmpty);

    _session!.reset();

    final allMessages = messages.toList();
    if (allMessages.isEmpty) return;

    final lastMessage = allMessages.removeLast();

    for (final msg in allMessages) {
      _session!.addMessage(
        toLlamaMessage(msg, format: format, hasTools: hasTools),
      );
    }

    final contentParts = _toLlamaContentParts(lastMessage);

    final effectiveOptions = options ?? defaultOptions;
    final buffer = StringBuffer();
    final thinkingBuffer = StringBuffer();
    int toolCallIdCounter = 0;

    final llamadartTools = tools
        ?.map((t) => _convertToolToDefinition(t))
        .toList();

    await for (final chunk in _session!.create(
      contentParts,
      enableThinking: true,
      params: GenerationParams(
        temp: effectiveOptions.temp ?? 0.8,
        topK: effectiveOptions.topK ?? 40,
        topP: effectiveOptions.topP ?? 0.9,
        penalty: effectiveOptions.repeatPenalty ?? 1.1,
        minP: effectiveOptions.minP ?? 0.05,
        maxTokens: effectiveOptions.maxTokens ?? 0,
        speculativeDecoding: effectiveOptions.speculativeDecoding ?? false,
      ),
      tools: llamadartTools,
      toolChoice: llamadartTools != null && llamadartTools.isNotEmpty
          ? ToolChoice.auto
          : null,
    )) {
      final delta = chunk.choices.firstOrNull?.delta;
      if (delta == null) continue;

      if (delta.thinking != null && delta.thinking!.isNotEmpty) {
        thinkingBuffer.write(delta.thinking!);
        yield ChatResult(
          output: ChatMessage(
            role: ChatMessageRole.model,
            parts: [ThinkingPart(delta.thinking!)],
          ),
        );
      }

      if (delta.toolCalls != null && delta.toolCalls!.isNotEmpty) {
        final parts = <Part>[];
        for (final tc in delta.toolCalls!) {
          final args = <String, dynamic>{};
          if (tc.function?.arguments != null) {
            try {
              final argsMap =
                  jsonDecode(tc.function!.arguments!) as Map<String, dynamic>;
              args.addAll(argsMap);
            } catch (_) {
              args['raw'] = tc.function!.arguments;
            }
          }
          parts.add(
            ToolPart.call(
              callId: tc.id ?? 'call_${toolCallIdCounter++}',
              toolName: tc.function?.name ?? 'unknown',
              arguments: args,
            ),
          );
        }
        yield ChatResult(
          output: ChatMessage(role: ChatMessageRole.model, parts: parts),
        );
        continue;
      }

      if (delta.content != null && delta.content!.isNotEmpty) {
        buffer.write(delta.content!);

        final bufferedContent = buffer.toString();

        final toolCallPatterns = _getToolCallPatterns(format);
        List<RegExpMatch>? matches;

        for (final pattern in toolCallPatterns) {
          final found = pattern.allMatches(bufferedContent).toList();
          if (found.isNotEmpty) {
            matches = found;
            break;
          }
        }

        matches ??= [];

        if (matches.isNotEmpty) {
          int lastEnd = 0;
          final parts = <Part>[];

          for (final match in matches) {
            if (match.start > lastEnd) {
              parts.add(
                TextPart(bufferedContent.substring(lastEnd, match.start)),
              );
            }

            final toolCallContent = match.group(1)!;
            final parsed = _parseToolCall(
              toolCallContent,
              'call_${toolCallIdCounter++}',
              format,
            );
            parts.add(parsed);

            lastEnd = match.end;
          }

          if (lastEnd < bufferedContent.length) {
            buffer.clear();
            buffer.write(bufferedContent.substring(lastEnd));
          } else {
            buffer.clear();
          }

          yield ChatResult(
            output: ChatMessage(role: ChatMessageRole.model, parts: parts),
          );
        } else {
          yield ChatResult(
            output: ChatMessage(
              role: ChatMessageRole.model,
              parts: [TextPart(delta.content!)],
            ),
          );
        }
      }
    }

    final remainingContent = buffer.toString();
    if (remainingContent.isNotEmpty) {
      yield ChatResult(
        output: ChatMessage(
          role: ChatMessageRole.model,
          parts: [TextPart(remainingContent)],
        ),
      );
    }
  }

  ToolDefinition _convertToolToDefinition(Tool<Object> tool) {
    final examples = _extractExamples(tool.inputSchema);
    final fullDescription = examples.isNotEmpty
        ? '${tool.description}\n\nExamples:\n${examples.map((e) => '- $e').join('\n')}'
        : tool.description;

    return ToolDefinition(
      name: tool.name,
      description: fullDescription,
      parameters: _convertSchemaToParams(tool.inputSchema),
      handler: (params) async {
        return await tool.onCall(params.raw);
      },
    );
  }

  List<String> _extractExamples(Schema? schema) {
    if (schema == null) return [];
    final examples = schema['examples'];
    if (examples is List) {
      return examples.cast<String>();
    }
    return [];
  }

  List<ToolParam> _convertSchemaToParams(Schema? schema) {
    if (schema == null) return [];

    final params = <ToolParam>[];
    final properties = schema['properties'] as Map<String, dynamic>?;
    final required = schema['required'] as List<dynamic>?;

    if (properties == null) return params;

    for (final entry in properties.entries) {
      final name = entry.key;
      final prop = entry.value as Map<String, dynamic>;
      final isRequired = required?.contains(name) ?? false;
      final paramType = prop['type'] as String? ?? 'string';

      ToolParam param;
      if (prop.containsKey('enum')) {
        final enumValues = (prop['enum'] as List).cast<String>();
        param = ToolParam.enumType(
          name,
          values: enumValues,
          description: prop['description'] as String?,
          required: isRequired,
        );
      } else {
        switch (paramType) {
          case 'integer':
            param = ToolParam.integer(
              name,
              description: prop['description'] as String?,
              required: isRequired,
            );
            break;
          case 'number':
            param = ToolParam.number(
              name,
              description: prop['description'] as String?,
              required: isRequired,
            );
            break;
          case 'boolean':
            param = ToolParam.boolean(
              name,
              description: prop['description'] as String?,
              required: isRequired,
            );
            break;
          case 'array':
            param = ToolParam.string(
              name,
              description: prop['description'] as String?,
              required: isRequired,
            );
            break;
          default:
            param = ToolParam.string(
              name,
              description: prop['description'] as String?,
              required: isRequired,
            );
        }
      }
      params.add(param);
    }

    return params;
  }

  List<RegExp> _getToolCallPatterns(ChatFormat format) {
    final formatStr = format.name;

    if (formatStr.contains('gemma4')) {
      return [
        RegExp(r'<\|tool_call>call:(\w+)\{(.*?)}<\|tool_call\|>', dotAll: true),
      ];
    }

    switch (format) {
      case ChatFormat.functionGemma:
        return [
          RegExp(
            r'<start_function_call>(.*?)<end_function_call>',
            dotAll: true,
          ),
        ];
      case ChatFormat.hermes:
      case ChatFormat.deepseekV3:
        return [RegExp(r'<tool_call>(.*?)</tool_call>', dotAll: true)];
      default:
        return [
          RegExp(r'<tool_call>(.*?)</tool_call>', dotAll: true),
          RegExp(
            r'<\|tool_call>call:(\w+)\{(.*?)}<\|tool_call\|>',
            dotAll: true,
          ),
        ];
    }
  }

  ToolPart _parseToolCall(String content, String callId, ChatFormat format) {
    try {
      try {
        final json = jsonDecode(content) as Map<String, dynamic>;
        final toolName = json.keys.first;
        final parameters = (json[toolName] as Map<String, dynamic>?) ?? {};
        return ToolPart.call(
          callId: callId,
          toolName: toolName,
          arguments: parameters,
        );
      } catch (_) {}

      final formatStr = format.name;
      final isGemma4 = formatStr.contains('gemma4');

      if (isGemma4) {
        final callPattern = RegExp(r'^call:(\w+)\{(.+)\}$');
        final match = callPattern.firstMatch(content.trim());
        if (match != null) {
          final toolName = match.group(1)!;
          final argsString = match.group(2)!;

          final argsPattern = RegExp(
            r'(\w+):(?:<\|"\|>([^<]*)<\|"\|>|([^,}]+))',
          );
          final arguments = <String, dynamic>{};

          for (final argMatch in argsPattern.allMatches(argsString)) {
            final key = argMatch.group(1)!;
            final value =
                (argMatch.group(2) ?? argMatch.group(3))?.trim() ?? '';
            final cleanValue = value
                .replaceAll('<|"|>', '')
                .replaceAll('"', '')
                .trim();
            if (cleanValue.isEmpty) continue;
            arguments[key] = _castValue(cleanValue);
          }

          return ToolPart.call(
            callId: callId,
            toolName: toolName,
            arguments: arguments,
          );
        }
      }

      final json = jsonDecode(content) as Map<String, dynamic>;
      final toolName = json.keys.first;
      final parameters = (json[toolName] as Map<String, dynamic>?) ?? {};
      return ToolPart.call(
        callId: callId,
        toolName: toolName,
        arguments: parameters,
      );
    } catch (e) {
      return ToolPart.call(
        callId: callId,
        toolName: 'error',
        arguments: {'error': 'Invalid tool call format: $content'},
      );
    }
  }

  dynamic _castValue(String v) {
    if (v == 'true') return true;
    if (v == 'false') return false;
    final intVal = int.tryParse(v);
    if (intVal != null) return intVal;
    final doubleVal = double.tryParse(v);
    if (doubleVal != null) return doubleVal;
    return v;
  }

  @visibleForTesting
  LlamaChatMessage toLlamaMessage(
    ChatMessage msg, {
    required ChatFormat format,
    required bool hasTools,
  }) {
    var parts = msg.parts;

    if (format == ChatFormat.functionGemma &&
        msg.role == ChatMessageRole.system &&
        hasTools) {
      const trigger =
          'You are a model that can do function calling with the following functions';
      final text = msg.text;
      if (!text.contains(trigger)) {
        if (msg.parts.isNotEmpty && msg.parts.first is TextPart) {
          final first = msg.parts.first as TextPart;
          parts = [TextPart('$trigger\n\n${first.text}'), ...msg.parts.skip(1)];
        } else {
          parts = [TextPart('$trigger\n\n'), ...msg.parts];
        }
      }
    }

    return LlamaChatMessage.withContent(
      role: _toLlamaRole(msg.role),
      content: _toLlamaContentPartsFromList(parts),
    );
  }

  List<LlamaContentPart> _toLlamaContentPartsFromList(List<Part> parts) {
    return parts.map((part) {
      if (part is TextPart) {
        return LlamaTextContent(part.text);
      }
      return LlamaTextContent(part.toString());
    }).toList();
  }

  List<LlamaContentPart> _toLlamaContentParts(ChatMessage msg) {
    return _toLlamaContentPartsFromList(msg.parts);
  }

  LlamaChatRole _toLlamaRole(ChatMessageRole role) {
    switch (role) {
      case ChatMessageRole.user:
        return LlamaChatRole.user;
      case ChatMessageRole.model:
        return LlamaChatRole.assistant;
      case ChatMessageRole.system:
        return LlamaChatRole.system;
    }
  }

  @override
  void dispose() {
    _engine?.dispose();
  }
}
