import 'dart:async';
import 'dart:convert';
import 'package:dartantic_interface/dartantic_interface.dart';
import 'package:llamadart/llamadart.dart';

import 'llamadart_chat_options.dart';
import 'llamadart_provider.dart';

/// A chat model implementation using the Llamadart engine.
class LlamadartChatModel extends ChatModel<LlamadartChatOptions> {
  final LlamadartProvider provider;

  LlamaEngine? _engine;
  ChatSession? _session;

  LlamadartChatModel({
    required this.provider,
    required super.name,
    required super.defaultOptions,
  });

  Future<void> _ensureInitialized() async {
    if (_engine != null) return;

    // Initialize engine with native backend
    final backend = LlamaBackend();
    _engine = LlamaEngine(backend);

    // Load the model
    await _engine!.loadModel(
      provider.modelPath,
      modelParams: ModelParams(
        contextSize: defaultOptions.nCtx ?? 2048,
        gpuLayers: defaultOptions.nGpuLayers ?? 0,
      ),
    );

    _session = ChatSession(_engine!);
  }

  @override
  Stream<ChatResult<ChatMessage>> sendStream(
    List<ChatMessage> messages, {
    LlamadartChatOptions? options,
    Schema? outputSchema,
  }) async* {
    await _ensureInitialized();

    _session!.reset();

    final allMessages = messages.toList();
    if (allMessages.isEmpty) return;

    final lastMessage = allMessages.removeLast();

    for (final msg in allMessages) {
      _session!.addMessage(_toLlamaMessage(msg));
    }

    final contentParts = _toLlamaContentParts(lastMessage);

    final effectiveOptions = options ?? defaultOptions;
    final buffer = StringBuffer();
    int toolCallIdCounter = 0;

    await for (final chunk in _session!.create(
      contentParts,
      enableThinking: true,
      params: GenerationParams(
        temp: effectiveOptions.temp ?? 0.8,
        topK: effectiveOptions.topK ?? 40,
        topP: effectiveOptions.topP ?? 0.9,
        penalty: effectiveOptions.repeatPenalty ?? 1.1,
      ),
    )) {
      final delta = chunk.choices.firstOrNull?.delta;
      if (delta == null) continue;

      if (delta.content != null && delta.content!.isNotEmpty) {
        buffer.write(delta.content!);

        final bufferedContent = buffer.toString();
        final toolCallPattern = RegExp(
          r'<tool_call>(.*?)</tool_call>',
          dotAll: true,
        );
        final matches = toolCallPattern.allMatches(bufferedContent);

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

      if (delta.thinking != null && delta.thinking!.isNotEmpty) {
        yield ChatResult(
          output: ChatMessage(
            role: ChatMessageRole.model,
            parts: [ThinkingPart(delta.thinking!)],
          ),
        );
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

  ToolPart _parseToolCall(String content, String callId) {
    try {
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

  LlamaChatMessage _toLlamaMessage(ChatMessage msg) {
    return LlamaChatMessage.withContent(
      role: _toLlamaRole(msg.role),
      content: _toLlamaContentParts(msg),
    );
  }

  List<LlamaContentPart> _toLlamaContentParts(ChatMessage msg) {
    return msg.parts.map((part) {
      if (part is TextPart) {
        return LlamaTextContent(part.text);
      }
      return LlamaTextContent(part.toString());
    }).toList();
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
