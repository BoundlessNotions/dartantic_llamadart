import 'dart:async';
import 'dart:convert';
import 'package:dartantic_interface/dartantic_interface.dart';
import 'package:llamadart/llamadart.dart';
import 'package:meta/meta.dart';

import 'llama_engine_cache.dart';
import 'llamadart_chat_options.dart';
import 'llamadart_provider.dart';
import 'tool_call_scanner.dart';

class LlamadartChatModel extends ChatModel<LlamadartChatOptions> {
  final LlamadartProvider provider;
  // ignore: overridden_fields, annotate_overrides
  final List<Tool<Object>>? tools;

  final Set<_GenerationState> _generations = {};

  /// Whether the model may produce reasoning before its answer, on models and
  /// templates that support it.
  final bool enableThinking;

  LlamadartChatModel({
    required this.provider,
    required super.name,
    this.tools,
    required super.defaultOptions,
    this.enableThinking = false,
  });

  /// Load-time parameters; together with the model path they pick the shared
  /// engine in [LlamaEngineCache].
  ModelParams _modelParams() {
    // When a GGUF MTP drafter is configured, llama.cpp requires the context to
    // reserve at least `draftTokenMax` recurrent-state rollback snapshots
    // (n_rs_seq) — otherwise generation fails with "MTP speculative decoding is
    // not available for this model/context". Reserve them at load time.
    final mtpDraft = defaultOptions.mtpDraftModelPath;
    final ggufMtpOn = mtpDraft != null && mtpDraft.isNotEmpty;
    final draftTokenMax = defaultOptions.mtpDraftTokenMax ?? 1;

    return ModelParams(
      contextSize: defaultOptions.nCtx ?? 8192,
      gpuLayers: defaultOptions.nGpuLayers ?? ModelParams.maxGpuLayers,
      preferredBackend: defaultOptions.preferredBackend,
      liteRtLmBackend: defaultOptions.liteRtLmBackend,
      chatTemplate: defaultOptions.chatTemplate,
      speculativeRollbackTokenMax: ggufMtpOn ? draftTokenMax : 0,
    );
  }

  /// The `responseFormat` that constrains output to [schema], or null.
  ///
  /// llamadart turns a JSON-schema response format into a grammar on
  /// llama.cpp and throws on schema keywords it can't convert. LiteRT-LM has
  /// no grammar constraints and llamadart rejects a strict response format
  /// there, so on that backend output is best effort: the schema is dropped.
  @visibleForTesting
  static Map<String, dynamic>? responseFormatFor(
    Schema? schema, {
    required bool isLiteRtLm,
  }) {
    if (schema == null || isLiteRtLm) return null;
    return {
      'type': 'json_schema',
      'json_schema': {'schema': jsonDecode(schema.toJson())},
    };
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
    final useGgufMtp = !isLiteRtLm && mtpDraft != null && mtpDraft.isNotEmpty;

    return GenerationParams(
      temp: options.temp ?? 0.8,
      topK: options.topK ?? 40,
      topP: options.topP ?? 0.9,
      penalty: isLiteRtLm
          ? genDefaults.penalty
          : (options.repeatPenalty ?? 1.1),
      minP: isLiteRtLm ? genDefaults.minP : (options.minP ?? 0.05),
      maxTokens: options.maxTokens ?? 0,
      reusePromptPrefix:
          options.reusePromptPrefix ?? genDefaults.reusePromptPrefix,
      streamBatchTokenThreshold:
          options.streamBatchTokenThreshold ??
          genDefaults.streamBatchTokenThreshold,
      streamBatchByteThreshold:
          options.streamBatchByteThreshold ??
          genDefaults.streamBatchByteThreshold,
      // LiteRT-LM honours the legacy bool; GGUF self-MTP also uses it. GGUF with
      // a separate drafter uses the explicit config below instead. draftTokenMax
      // must match the rollback snapshots reserved at model load (see
      // _modelParams).
      speculativeDecoding: useGgufMtp
          ? false
          : (options.speculativeDecoding ?? false),
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
  }) {
    final state = _GenerationState();
    final results = _send(state, messages, options, outputSchema);

    // A generator waiting on the next token only sees a cancel once that
    // token arrives, and prompt processing can take seconds. Stopping the
    // native generation on cancel (what a `.timeout` on this stream does)
    // ends the engine stream now, so the lock is released promptly.
    late final StreamController<ChatResult<ChatMessage>> controller;
    controller = StreamController(
      onListen: () {
        _generations.add(state);
        final subscription = results.listen(
          controller.add,
          onError: controller.addError,
          onDone: () {
            _generations.remove(state);
            controller.close();
          },
        );
        controller
          ..onPause = subscription.pause
          ..onResume = subscription.resume
          ..onCancel = () {
            _generations.remove(state);
            state.cancel();
            return subscription.cancel();
          };
      },
    );
    return controller.stream;
  }

  Stream<ChatResult<ChatMessage>> _send(
    _GenerationState state,
    List<ChatMessage> messages,
    LlamadartChatOptions? options,
    Schema? outputSchema,
  ) async* {
    if (messages.isEmpty) return;
    final effectiveOptions = defaultOptions.mergedWith(options);

    // Generations on a shared engine are serialized; the lock is released
    // however this stream ends.
    final (handle, release) = await _lockEngine();
    try {
      if (state.cancelled) return;
      yield* _generate(state, handle, messages, effectiveOptions, outputSchema);
    } finally {
      release();
    }
  }

  /// Whether [error], raised by the engine's `create` stream, can mean the
  /// native state is corrupt, so the engine has to be reloaded.
  @visibleForTesting
  static bool errorCorruptsEngine(Object error) => switch (error) {
    // Request-shape errors from llamadart's request planner, raised before
    // generation starts.
    LlamaUnsupportedException() || LlamaContextException() => false,
    // llama_cpp_service throws this from tokenization, before any decode.
    // Checked against SmolLM2 with nCtx 256: after the overflow, the same
    // engine produced output identical to a pre-overflow baseline. Matched on
    // the wrapped exception's message since llamadart has no type for it.
    LlamaInferenceException(:final details)
        when '$details'.contains('Tokenization failed or prompt too long') =>
      false,
    _ => true,
  };

  /// Acquires the shared engine and waits for exclusive use of it.
  Future<(LlamaEngineHandle, void Function())> _lockEngine() async {
    while (true) {
      // Acquire per call rather than holding the engine: another model
      // sharing it may have evicted it, and the cache then loads a fresh one.
      final handle = await LlamaEngineCache.instance.acquire(
        provider.modelPath,
        _modelParams(),
      );
      final release = await handle.lock();
      // The holder ahead of us may have evicted it after a failed generation.
      if (!handle.isEvicted) return (handle, release);
      release();
    }
  }

  Stream<ChatResult<ChatMessage>> _generate(
    _GenerationState state,
    LlamaEngineHandle handle,
    List<ChatMessage> messages,
    LlamadartChatOptions effectiveOptions,
    Schema? outputSchema,
  ) async* {
    final engine = handle.engine;
    final format = await handle.chatFormat();
    final hasTools =
        outputSchema != null || (tools != null && tools!.isNotEmpty);

    // The whole history goes straight to engine.create. llamadart's
    // ChatSession drops system-role history messages and has no
    // responseFormat, and dartantic already owns the conversation.
    final llamaMessages = [
      for (final msg in messages)
        toLlamaMessage(msg, format: format, hasTools: hasTools),
    ];

    var toolCallIdCounter = 0;
    String nextCallId() => 'call_${toolCallIdCounter++}';

    final llamadartTools = tools?.map(toolDefinitionFor).toList();

    // With tools, llamadart's template handler parses calls into
    // delta.toolCalls. Scanning the content too would parse them twice, so
    // the text fallback only runs for tool-less requests (e.g. prompt-
    // instructed calls).
    final scanner = ToolCallScanner(
      llamadartTools == null || llamadartTools.isEmpty
          ? toolCallEnvelopes(format)
          : const [],
    );

    // LiteRT-LM rejects llama.cpp-only sampling knobs (minP, penalty) when they
    // differ from GenerationParams defaults. Route on the model path the same
    // way llamadart does so those fields stay at their defaults for .litertlm
    // bundles while GGUF keeps the full sampler controls.
    final isLiteRtLm = provider.modelPath.toLowerCase().endsWith('.litertlm');

    final params = buildGenerationParams(
      effectiveOptions,
      isLiteRtLm: isLiteRtLm,
    );

    // Tag errors that come out of the engine, so a bug in this adapter's own
    // chunk handling never costs a model reload.
    Object? engineError;
    final chunks = engine
        .create(
          llamaMessages,
          enableThinking: enableThinking,
          params: params,
          responseFormat: responseFormatFor(
            outputSchema,
            isLiteRtLm: isLiteRtLm,
          ),
          tools: llamadartTools,
          toolChoice: llamadartTools != null && llamadartTools.isNotEmpty
              ? ToolChoice.auto
              : null,
        )
        .handleError((Object error, StackTrace stackTrace) {
          engineError = error;
          Error.throwWithStackTrace(error, stackTrace);
        });

    var completed = false;
    var finishReason = FinishReason.unspecified;
    state.engine = engine;
    try {
      await for (final chunk in chunks) {
        final choice = chunk.choices.firstOrNull;
        if (choice == null) continue;
        final delta = choice.delta;
        final parts = <Part>[];

        final thinking = delta.thinking;
        if (thinking != null && thinking.isNotEmpty) {
          parts.add(ThinkingPart(thinking));
        }

        final toolCalls = delta.toolCalls;
        final content = delta.content;
        if (toolCalls != null && toolCalls.isNotEmpty) {
          parts.addAll(toolCalls.map((tc) => _toolCallPart(tc, nextCallId)));
        } else if (content != null && content.isNotEmpty) {
          parts.addAll(_partsFrom(scanner.add(content), nextCallId));
        }

        if (choice.finishReason != null) {
          finishReason = _toFinishReason(choice.finishReason);
        }
        if (parts.isNotEmpty || choice.finishReason != null) {
          yield ChatResult(
            output: ChatMessage(role: ChatMessageRole.model, parts: parts),
            finishReason: choice.finishReason != null
                ? finishReason
                : FinishReason.unspecified,
          );
        }
      }
      completed = true;

      // Callers read the finish reason off the last result, so the flushed
      // tail repeats it.
      final rest = _partsFrom(scanner.close(), nextCallId);
      if (rest.isNotEmpty) {
        yield ChatResult(
          output: ChatMessage(role: ChatMessageRole.model, parts: rest),
          finishReason: finishReason,
        );
      }
    } catch (error) {
      // A failed native generation can corrupt the engine, and reusing it
      // segfaults on a worker thread. Evict it so the next call, from this or
      // any model sharing it, reloads a clean one.
      if (identical(error, engineError) && errorCorruptsEngine(error)) {
        await LlamaEngineCache.instance.evict(handle);
      }
      rethrow;
    } finally {
      state.engine = null;
      // Don't leave a generation running on the engine the next caller gets.
      if (!completed && !handle.isEvicted) engine.cancelGeneration();
    }
  }

  ToolPart _toolCallPart(
    LlamaCompletionChunkToolCall toolCall,
    String Function() nextCallId,
  ) {
    final function = toolCall.function;
    final rawArguments = function?.arguments;
    final arguments = <String, dynamic>{};
    if (rawArguments != null) {
      try {
        arguments.addAll(jsonDecode(rawArguments) as Map<String, dynamic>);
      } catch (_) {
        arguments['raw'] = rawArguments;
      }
    }
    return ToolPart.call(
      callId: toolCall.id ?? nextCallId(),
      toolName: function?.name ?? 'unknown',
      arguments: arguments,
    );
  }

  /// llamadart only reports 'stop' and 'tool_calls'.
  static FinishReason _toFinishReason(String? reason) => switch (reason) {
    'stop' => FinishReason.stop,
    'tool_calls' => FinishReason.toolCalls,
    _ => FinishReason.unspecified,
  };

  @visibleForTesting
  ToolDefinition toolDefinitionFor(Tool<Object> tool) {
    final examples = _extractExamples(tool.inputSchema);
    final fullDescription = examples.isNotEmpty
        ? '${tool.description}\n\nExamples:\n${examples.map((e) => '- $e').join('\n')}'
        : tool.description;

    return ToolDefinition(
      name: tool.name,
      description: fullDescription,
      parameters: _convertSchemaToParams(tool.inputSchema?.value),
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
    final examples = schema?['examples'];
    if (examples is! List) return [];
    return [for (final e in examples) e is String ? e : jsonEncode(e)];
  }

  List<ToolParam> _convertSchemaToParams(Map<String, dynamic>? schema) {
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
    final notes = <String>[];
    final values = prop['enum'] as List<dynamic>?;
    if (values != null && values.every((v) => v is String)) {
      return ToolParam.enumType(
        name,
        values: values.cast<String>(),
        description: prop['description'] as String?,
        required: isRequired,
      );
    }
    // ToolParam.enumType only takes strings, and stringifying would have the
    // model send "1" to a tool expecting 1. Keep the declared type instead
    // and list the values for the model.
    if (values != null) {
      notes.add('Allowed values: ${values.map(jsonEncode).join(', ')}.');
    }

    final paramType = switch (prop['type']) {
      final String type => type,
      // ToolParam has no union type. A nullable type is its non-null member;
      // anything wider is a string, with the JSON types noted. A null-only
      // property is also a string: ToolParam.nullType needs llamadart 0.8.21.
      final List<dynamic> types => switch (types.where((t) => t != 'null')) {
        final nonNull when nonNull.length == 1 => nonNull.single as String,
        final nonNull => () {
          notes.add(
            nonNull.isEmpty
                ? 'Must be null.'
                : 'JSON type: one of ${nonNull.join(', ')}.',
          );
          return 'string';
        }(),
      },
      _ => 'string',
    };

    final description = [
      prop['description'] as String?,
      ...notes,
    ].nonNulls.join(' ');
    final desc = description.isEmpty ? null : description;

    switch (paramType) {
      case 'integer':
        return ToolParam.integer(name, description: desc, required: isRequired);
      case 'number':
        return ToolParam.number(name, description: desc, required: isRequired);
      case 'boolean':
        return ToolParam.boolean(name, description: desc, required: isRequired);
      case 'array':
        final itemSchema = prop['items'] as Map<String, dynamic>?;
        final itemParam = itemSchema != null
            ? _schemaPropertyToToolParam(
                name: 'item',
                prop: itemSchema,
                isRequired: false,
              )
            : ToolParam.string('item');
        return ToolParam.array(
          name,
          itemType: itemParam,
          description: desc,
          required: isRequired,
        );
      case 'object':
        return ToolParam.object(
          name,
          properties: _convertSchemaToParams(prop),
          description: desc,
          required: isRequired,
        );
      default:
        return ToolParam.string(name, description: desc, required: isRequired);
    }
  }

  static const _hermesEnvelope = ToolCallEnvelope(
    '<tool_call>',
    '</tool_call>',
  );

  // Gemma4Handler's envelope (gemma4_handler.dart in llamadart).
  static const _gemmaEnvelope = ToolCallEnvelope(
    '<|tool_call>',
    '<tool_call|>',
  );

  /// Tool-call envelopes each format may write into plain content.
  static final Map<ChatFormat, List<ToolCallEnvelope>> _envelopesByFormat = {
    ChatFormat.gemma4: const [_gemmaEnvelope],
    ChatFormat.functionGemma: const [
      ToolCallEnvelope('<start_function_call>', '<end_function_call>'),
    ],
    ChatFormat.hermes: const [_hermesEnvelope],
    ChatFormat.deepseekV3: const [_hermesEnvelope],
  };

  static const _defaultEnvelopes = [_hermesEnvelope, _gemmaEnvelope];

  @visibleForTesting
  static List<ToolCallEnvelope> toolCallEnvelopes(ChatFormat format) =>
      _envelopesByFormat[format] ?? _defaultEnvelopes;

  List<Part> _partsFrom(
    List<ScanSegment> segments,
    String Function() nextCallId,
  ) => [
    for (final segment in segments)
      switch (segment) {
        TextSegment(:final text) => TextPart(text),
        // An envelope that doesn't parse stays text: yielding a fake tool
        // call would have dartantic try to run a tool that doesn't exist.
        EnvelopeSegment(:final envelope, :final body) =>
          parseToolCallBody(body, nextCallId) ??
              TextPart('${envelope.open}$body${envelope.close}'),
      },
  ];

  static final _gemmaCall = RegExp(r'^call:(\w+)\{(.*)\}$', dotAll: true);
  static final _gemmaArg = RegExp(
    r'(\w+):(?:<\|\\?"\|>(.*?)<\|\\?"\|>|([^,}]+))',
    dotAll: true,
  );

  /// Parses the body of a tool-call envelope, or returns null when it isn't a
  /// tool call.
  ///
  /// Accepts Gemma's `call:name{key:value,...}` and these JSON shapes, in
  /// order: `{"name", "arguments": {...} | "<json>"}` (Hermes/Qwen),
  /// `{"name", "parameters": {...}}`, and the single-key `{"<tool>": {...}}`.
  @visibleForTesting
  static ToolPart? parseToolCallBody(
    String body,
    String Function() nextCallId,
  ) {
    final trimmed = body.trim();
    final gemma = _gemmaCall.firstMatch(trimmed);
    if (gemma != null) {
      return ToolPart.call(
        callId: nextCallId(),
        toolName: gemma.group(1)!,
        arguments: {
          for (final arg in _gemmaArg.allMatches(gemma.group(2)!))
            arg.group(1)!: arg.group(2) ?? _castValue(arg.group(3)!.trim()),
        },
      );
    }

    final Object? json;
    try {
      json = jsonDecode(trimmed);
    } on FormatException {
      return null;
    }
    if (json is! Map<String, dynamic>) return null;

    final (name, arguments) = switch (json) {
      {'name': final String name, 'arguments': final Map<String, dynamic> a} =>
        (name, a),
      {'name': final String name, 'arguments': final String a} => (
        name,
        _decodeObject(a),
      ),
      {'name': final String name, 'parameters': final Map<String, dynamic> a} =>
        (name, a),
      _ when json.length == 1 && json.values.single is Map<String, dynamic> => (
        json.keys.single,
        json.values.single as Map<String, dynamic>,
      ),
      _ => (null, null),
    };
    if (name == null || arguments == null) return null;
    return ToolPart.call(
      callId: nextCallId(),
      toolName: name,
      arguments: arguments,
    );
  }

  static Map<String, dynamic>? _decodeObject(String source) {
    try {
      final decoded = jsonDecode(source);
      return decoded is Map<String, dynamic> ? decoded : null;
    } on FormatException {
      return null;
    }
  }

  static Object _castValue(String v) {
    if (v == 'true') return true;
    if (v == 'false') return false;
    return int.tryParse(v) ?? double.tryParse(v) ?? v;
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

  List<LlamaContentPart> _toLlamaContentPartsFromList(List<Part> parts) =>
      parts.map(_toLlamaContentPart).toList();

  // dartantic's Part is genai_primitives' sealed StandardPart, so this switch
  // is exhaustive and a new part type fails to compile rather than being
  // stringified into the prompt.
  LlamaContentPart _toLlamaContentPart(Part part) {
    return switch (part) {
      TextPart(:final text) => LlamaTextContent(text),
      // Each chat template decides whether to render or strip prior
      // reasoning, so pass it through rather than dropping it here.
      ThinkingPart(:final text) => LlamaThinkingContent(text),
      ToolPart(kind: ToolPartKind.call) => LlamaToolCallContent(
        id: part.callId,
        name: part.toolName,
        arguments: Map<String, dynamic>.from(part.arguments ?? {}),
        rawJson: part.argumentsRaw,
      ),
      ToolPart(:final result) => LlamaToolResultContent(
        id: part.callId,
        name: part.toolName,
        result: result is Map || result is List
            ? jsonEncode(result)
            : result?.toString() ?? '',
      ),
      DataPart(:final bytes, :final mimeType) => switch (_mediaKind(mimeType)) {
        _MediaKind.image => LlamaImageContent(bytes: bytes),
        _MediaKind.audio => LlamaAudioContent(bytes: bytes),
        null => throw UnsupportedError(
          'llamadart has no content type for $mimeType data',
        ),
      },
      LinkPart(:final url, :final mimeType) => switch ((
        url.scheme,
        _mediaKind(mimeType),
      )) {
        ('file', _MediaKind.image) => LlamaImageContent(path: url.toFilePath()),
        ('file', _MediaKind.audio) => LlamaAudioContent(path: url.toFilePath()),
        ('http' || 'https', _MediaKind.image) => LlamaImageContent(
          url: url.toString(),
        ),
        _ => throw UnsupportedError(
          'llamadart has no content type for a ${mimeType ?? 'untyped'} '
          'link to $url',
        ),
      },
    };
  }

  static _MediaKind? _mediaKind(String? mimeType) => switch (mimeType) {
    final String m when m.startsWith('image/') => _MediaKind.image,
    final String m when m.startsWith('audio/') => _MediaKind.audio,
    _ => null,
  };

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
    // The engine is owned by LlamaEngineCache and shared across models — do
    // not dispose it here. Use LlamaEngineCache.instance.disposeAll() at app
    // shutdown to release native resources. Stop this model's own generations
    // so they don't hold the engine for other models.
    for (final state in List.of(_generations)) {
      state.cancel();
    }
  }
}

/// Lets a subscriber's cancel reach an in-flight [LlamadartChatModel.sendStream].
class _GenerationState {
  bool cancelled = false;

  /// The engine while its `create` stream is being consumed.
  LlamaEngine? engine;

  void cancel() {
    cancelled = true;
    engine?.cancelGeneration();
  }
}

enum _MediaKind { image, audio }
