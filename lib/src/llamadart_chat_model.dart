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

    // When a GGUF MTP drafter is configured, llama.cpp requires the context to
    // reserve at least `draftTokenMax` recurrent-state rollback snapshots
    // (n_rs_seq) — otherwise generation fails with "MTP speculative decoding is
    // not available for this model/context". Reserve them at load time.
    final mtpDraft = defaultOptions.mtpDraftModelPath;
    final ggufMtpOn = mtpDraft != null && mtpDraft.isNotEmpty;
    final draftTokenMax = defaultOptions.mtpDraftTokenMax ?? 1;

    await _engine!.loadModel(
      provider.modelPath,
      modelParams: ModelParams(
        contextSize: defaultOptions.nCtx ?? 8192,
        gpuLayers: defaultOptions.nGpuLayers ?? ModelParams.maxGpuLayers,
        preferredBackend: defaultOptions.preferredBackend,
        liteRtLmBackend: defaultOptions.liteRtLmBackend,
        chatTemplate: defaultOptions.chatTemplate,
        speculativeRollbackTokenMax: ggufMtpOn ? draftTokenMax : 0,
      ),
    );

    _session = ChatSession(_engine!);
  }

  /// Tears down the engine/session so the next call rebuilds a clean one.
  ///
  /// The native LiteRT-LM runtime can leave the engine in a corrupted state
  /// after a failed generation; reusing it then crashes (SIGSEGV) on a worker
  /// thread. Disposing here converts that fatal native crash into a recoverable
  /// per-call error.
  Future<void> _resetEngine() async {
    try {
      _engine?.dispose();
    } catch (_) {
      // Best effort — the engine may already be in a bad state.
    }
    _engine = null;
    _session = null;
  }

  Future<ChatFormat> _getChatFormat() async {
    await _ensureInitialized();
    final metadata = await _engine!.getMetadata();
    final template = metadata['tokenizer.chat_template'];
    return ChatTemplateEngine.detectFormat(template);
  }

  /// Builds [GenerationParams] from [options], honoring backend capabilities.
  ///
  /// The LiteRT-LM backend only supports a subset of sampling controls and
  /// throws on llama.cpp-specific knobs (`minP`, `penalty`) whose values differ
  /// from the [GenerationParams] defaults. When [isLiteRtLm] is true those
  /// fields are left at their defaults so the runtime accepts the request;
  /// GGUF/llama.cpp receives the full set.
  @visibleForTesting
  GenerationParams buildGenerationParams(
    LlamadartChatOptions options, {
    required bool isLiteRtLm,
  }) {
    const genDefaults = GenerationParams();

    // On llama.cpp/GGUF, a non-empty MTP draft path enables draft-mtp
    // speculative decoding via a separate drafter; the LiteRT-LM backend instead
    // carries its MTP heads in the bundle and is driven by the legacy bool.
    final mtpDraft = options.mtpDraftModelPath;
    final useGgufMtp =
        !isLiteRtLm && mtpDraft != null && mtpDraft.isNotEmpty;

    return GenerationParams(
      temp: options.temp ?? 0.8,
      topK: options.topK ?? 40,
      topP: options.topP ?? 0.9,
      penalty: isLiteRtLm
          ? genDefaults.penalty
          : (options.repeatPenalty ?? 1.1),
      minP: isLiteRtLm ? genDefaults.minP : (options.minP ?? 0.05),
      maxTokens: options.maxTokens ?? 0,
      // LiteRT-LM honours the legacy bool; GGUF self-MTP also uses it. GGUF with
      // a separate drafter uses the explicit config below instead. draftTokenMax
      // must match the rollback snapshots reserved at model load (see
      // _ensureInitialized).
      speculativeDecoding: useGgufMtp ? false : (options.speculativeDecoding ?? false),
      speculativeDecodingConfig: useGgufMtp
          ? SpeculativeDecodingConfig.mtp(
              draftModelPath: mtpDraft,
              draftTokenMax: options.mtpDraftTokenMax ?? 1,
            )
          : null,
    );
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

    // Tool result messages use the 'tool' role, not 'user'. Add them as a
    // session message so the chat template formats them correctly, then call
    // create([]) to let the model generate its response.
    final hasToolResult = lastMessage.parts.any(
      (p) => p is ToolPart && p.kind == ToolPartKind.result,
    );
    final List<LlamaContentPart> contentParts;
    if (hasToolResult) {
      _session!.addMessage(
        toLlamaMessage(lastMessage, format: format, hasTools: hasTools),
      );
      contentParts = [];
    } else {
      contentParts = _toLlamaContentParts(lastMessage);
    }

    final effectiveOptions = options ?? defaultOptions;
    final buffer = StringBuffer();
    final thinkingBuffer = StringBuffer();
    int toolCallIdCounter = 0;

    final llamadartTools = tools
        ?.map((t) => _convertToolToDefinition(t))
        .toList();

    // LiteRT-LM rejects llama.cpp-only sampling knobs (minP, penalty) when they
    // differ from GenerationParams defaults. Route on the model path the same
    // way llamadart does so those fields stay at their defaults for .litertlm
    // bundles while GGUF keeps the full sampler controls.
    final isLiteRtLm = provider.modelPath.toLowerCase().endsWith('.litertlm');

    try {
      await for (final chunk in _session!.create(
        contentParts,
        enableThinking: true,
        params: buildGenerationParams(effectiveOptions, isLiteRtLm: isLiteRtLm),
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
    } catch (_) {
      // A failed native generation can corrupt the engine; reusing it on the
      // next call segfaults. Rebuild on the next call instead of crashing.
      await _resetEngine();
      rethrow;
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
        // If a zone-scoped tool-target map is present (keyed by #toolTargets),
        // prefer the real handler from that map over the placeholder onCall.
        final zoneTargets =
            Zone.current[#toolTargets] as Map<String, Tool<Object>>?;
        final actualTool = zoneTargets?[tool.name] ?? tool;
        return await actualTool.onCall(params.raw);
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

    final properties = schema['properties'] as Map<String, dynamic>?;
    final required = schema['required'] as List<dynamic>?;
    if (properties == null) return [];

    return [
      for (final entry in properties.entries)
        _schemaPropertyToToolParam(
          name: entry.key,
          prop: entry.value as Map<String, dynamic>,
          isRequired: required?.contains(entry.key) ?? false,
        ),
    ];
  }

  ToolParam _schemaPropertyToToolParam({
    required String name,
    required Map<String, dynamic> prop,
    required bool isRequired,
  }) {
    final description = prop['description'] as String?;

    if (prop.containsKey('enum')) {
      final enumValues = (prop['enum'] as List).cast<String>();
      return ToolParam.enumType(
        name,
        values: enumValues,
        description: description,
        required: isRequired,
      );
    }

    final paramType = prop['type'] as String? ?? 'string';
    switch (paramType) {
      case 'integer':
        return ToolParam.integer(name, description: description, required: isRequired);
      case 'number':
        return ToolParam.number(name, description: description, required: isRequired);
      case 'boolean':
        return ToolParam.boolean(name, description: description, required: isRequired);
      case 'array':
        final itemSchema = prop['items'] as Map<String, dynamic>?;
        final itemParam = itemSchema != null
            ? _schemaPropertyToToolParam(name: 'item', prop: itemSchema, isRequired: false)
            : ToolParam.string('item');
        return ToolParam.array(name, itemType: itemParam, description: description, required: isRequired);
      case 'object':
        final nestedProps = prop['properties'] as Map<String, dynamic>?;
        final nestedRequired = prop['required'] as List<dynamic>?;
        final nestedParams = nestedProps?.entries
            .map((e) => _schemaPropertyToToolParam(
                  name: e.key,
                  prop: e.value as Map<String, dynamic>,
                  isRequired: nestedRequired?.contains(e.key) ?? false,
                ))
            .toList() ?? [];
        return ToolParam.object(name, properties: nestedParams, description: description, required: isRequired);
      default:
        return ToolParam.string(name, description: description, required: isRequired);
    }
  }

  List<RegExp> _getToolCallPatterns(ChatFormat format) {
    final formatStr = format.name;

    if (formatStr.contains('gemma4')) {
      // Gemma4Handler already extracts tool calls from the full output and
      // yields them as delta.toolCalls with complete arguments.  Applying a
      // text regex here too would fire on the raw content chunks BEFORE the
      // native chunk arrives, capturing only the tool name (group 1 of the
      // old regex was (\w+), not the full call expression) and creating
      // spurious `error` tool calls that confuse the agent.
      return [];
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

    // Tool result messages must use LlamaChatRole.tool so the chat template
    // formats them as <start_of_turn>tool rather than <start_of_turn>user.
    final hasToolResults = parts.any(
      (p) => p is ToolPart && p.kind == ToolPartKind.result,
    );
    final role = hasToolResults ? LlamaChatRole.tool : _toLlamaRole(msg.role);

    return LlamaChatMessage.withContent(
      role: role,
      content: _toLlamaContentPartsFromList(parts),
    );
  }

  List<LlamaContentPart> _toLlamaContentPartsFromList(List<Part> parts) {
    return parts.map((part) {
      if (part is TextPart) {
        return LlamaTextContent(part.text);
      }
      if (part is ToolPart) {
        if (part.kind == ToolPartKind.call) {
          return LlamaToolCallContent(
            id: part.callId,
            name: part.toolName,
            arguments: Map<String, dynamic>.from(part.arguments ?? {}),
            rawJson: part.argumentsRaw,
          );
        } else {
          final result = part.result;
          final resultStr = result is Map || result is List
              ? jsonEncode(result)
              : result?.toString() ?? '';
          return LlamaToolResultContent(
            id: part.callId,
            name: part.toolName,
            result: resultStr,
          );
        }
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
