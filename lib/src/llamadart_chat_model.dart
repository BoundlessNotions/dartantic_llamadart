import 'dart:async';
import 'dart:convert';
import 'package:dartantic_interface/dartantic_interface.dart';
import 'package:llamadart/llamadart.dart';
// Not exported from llamadart's public barrel; the exact version pin in
// pubspec makes the src import stable. TODO(llamadart): ask upstream to
// export JsonSchemaConverter.
// ignore: implementation_imports
import 'package:llamadart/src/core/grammar/json_schema_converter.dart';
import 'package:meta/meta.dart';

import 'llama_engine_cache.dart';
import 'llamadart_chat_options.dart';
import 'llamadart_provider.dart';
import 'tool_call_scanner.dart';

class LlamadartChatModel extends ChatModel<LlamadartChatOptions> {
  final LlamadartProvider provider;
  // ignore: overridden_fields, annotate_overrides
  final List<Tool<Object>>? tools;

  LlamaEngine? _engine;
  String? _engineCacheKey;

  LlamadartChatModel({
    required this.provider,
    required super.name,
    this.tools,
    required super.defaultOptions,
  });

  Future<void> _ensureInitialized() async {
    if (_engine != null) return;

    // When a GGUF MTP drafter is configured, llama.cpp requires the context to
    // reserve at least `draftTokenMax` recurrent-state rollback snapshots
    // (n_rs_seq) — otherwise generation fails with "MTP speculative decoding is
    // not available for this model/context". Reserve them at load time.
    final mtpDraft = defaultOptions.mtpDraftModelPath;
    final ggufMtpOn = mtpDraft != null && mtpDraft.isNotEmpty;
    final draftTokenMax = defaultOptions.mtpDraftTokenMax ?? 1;

    final params = ModelParams(
      contextSize: defaultOptions.nCtx ?? 8192,
      gpuLayers: defaultOptions.nGpuLayers ?? ModelParams.maxGpuLayers,
      preferredBackend: defaultOptions.preferredBackend,
      liteRtLmBackend: defaultOptions.liteRtLmBackend,
      chatTemplate: defaultOptions.chatTemplate,
      speculativeRollbackTokenMax: ggufMtpOn ? draftTokenMax : 0,
    );

    // Engines are cached process-wide: model load (weights + graph compile +
    // context allocation) is paid once per (path, params), not per chat model.
    _engineCacheKey = LlamaEngineCache.keyFor(provider.modelPath, params);
    _engine = await LlamaEngineCache.instance.acquire(
      provider.modelPath,
      params,
    );
  }

  /// Evicts the shared engine so the next call reloads a clean one.
  ///
  /// The native runtime can be left in a corrupted state after a failed
  /// generation; reusing it then crashes (SIGSEGV) on a worker thread.
  /// Evicting here converts that fatal native crash into a recoverable
  /// per-call error.
  Future<void> _resetEngine() async {
    final key = _engineCacheKey;
    if (key != null) {
      await LlamaEngineCache.instance.evict(key);
    }
    _engine = null;
    _engineCacheKey = null;
  }

  Future<ChatFormat> _getChatFormat() async {
    await _ensureInitialized();
    final metadata = await _engine!.getMetadata();
    final template = metadata['tokenizer.chat_template'];
    return ChatTemplateEngine.detectFormat(template);
  }

  /// Converts a dartantic [Schema] into a GBNF grammar for constrained
  /// decoding, or null when the schema can't be converted or the backend
  /// (LiteRT-LM) doesn't support grammar constraints.
  @visibleForTesting
  static String? grammarForSchema(Schema? schema, {required bool isLiteRtLm}) {
    if (schema == null || isLiteRtLm) return null;
    try {
      final map = jsonDecode(schema.toJson()) as Map<String, dynamic>;
      return JsonSchemaConverter.convert(map);
    } catch (_) {
      // Unconvertible schema — generate unconstrained rather than fail.
      return null;
    }
  }

  /// Builds [GenerationParams] from [options], honoring backend capabilities.
  ///
  /// The LiteRT-LM backend only supports a subset of sampling controls and
  /// throws on llama.cpp-specific knobs (`minP`, `penalty`) whose values differ
  /// from the [GenerationParams] defaults. When [isLiteRtLm] is true those
  /// fields are left at their defaults so the runtime accepts the request;
  /// GGUF/llama.cpp receives the full set. [grammar] (from [grammarForSchema])
  /// constrains decoding to schema-shaped output on grammar-capable backends.
  @visibleForTesting
  GenerationParams buildGenerationParams(
    LlamadartChatOptions options, {
    required bool isLiteRtLm,
    String? grammar,
  }) {
    const genDefaults = GenerationParams();

    // On llama.cpp/GGUF, a non-empty MTP draft path enables draft-mtp
    // speculative decoding via a separate drafter; the LiteRT-LM backend instead
    // carries its MTP heads in the bundle and is driven by the legacy bool.
    final mtpDraft = options.mtpDraftModelPath;
    final useGgufMtp = !isLiteRtLm && mtpDraft != null && mtpDraft.isNotEmpty;

    return GenerationParams(
      grammar: grammar,
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
  }) async* {
    await _ensureInitialized();

    // The engine is shared across chat models. A caller that timed out on a
    // previous call cannot cancel the underlying native generation via
    // `.timeout` — it keeps running. Interrupt it here so this call doesn't
    // race (or queue behind) an abandoned zombie generation. Safe when idle:
    // the cancel token is per-generation and null between runs.
    _engine!.cancelGeneration();

    final format = await _getChatFormat();
    final hasTools =
        outputSchema != null || (tools != null && tools!.isNotEmpty);

    if (messages.isEmpty) return;

    // The whole history goes straight to engine.create. llamadart's
    // ChatSession drops system-role history messages and has no
    // responseFormat, and dartantic already owns the conversation.
    final llamaMessages = [
      for (final msg in messages)
        toLlamaMessage(msg, format: format, hasTools: hasTools),
    ];

    final effectiveOptions = options ?? defaultOptions;
    var toolCallIdCounter = 0;
    String nextCallId() => 'call_${toolCallIdCounter++}';

    final llamadartTools = tools
        ?.map((t) => _convertToolToDefinition(t))
        .toList();

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

    // Structured output: constrain decoding to the caller's schema via GBNF
    // grammar sampling (llama.cpp). Template-generated tool-call grammars take
    // precedence inside the engine, so this applies when the caller wants raw
    // schema-shaped JSON rather than a tool call.
    final grammar = grammarForSchema(outputSchema, isLiteRtLm: isLiteRtLm);

    try {
      await for (final chunk in _engine!.create(
        llamaMessages,
        enableThinking: true,
        params: buildGenerationParams(
          effectiveOptions,
          isLiteRtLm: isLiteRtLm,
          grammar: grammar,
        ),
        tools: llamadartTools,
        toolChoice: llamadartTools != null && llamadartTools.isNotEmpty
            ? ToolChoice.auto
            : null,
      )) {
        final delta = chunk.choices.firstOrNull?.delta;
        if (delta == null) continue;

        if (delta.thinking != null && delta.thinking!.isNotEmpty) {
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
                callId: tc.id ?? nextCallId(),
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

        final content = delta.content;
        if (content != null && content.isNotEmpty) {
          final parts = _partsFrom(scanner.add(content), nextCallId, format);
          if (parts.isNotEmpty) {
            yield ChatResult(
              output: ChatMessage(role: ChatMessageRole.model, parts: parts),
            );
          }
        }
      }

      final rest = _partsFrom(scanner.close(), nextCallId, format);
      if (rest.isNotEmpty) {
        yield ChatResult(
          output: ChatMessage(role: ChatMessageRole.model, parts: rest),
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
        return ToolParam.integer(
          name,
          description: description,
          required: isRequired,
        );
      case 'number':
        return ToolParam.number(
          name,
          description: description,
          required: isRequired,
        );
      case 'boolean':
        return ToolParam.boolean(
          name,
          description: description,
          required: isRequired,
        );
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
          description: description,
          required: isRequired,
        );
      case 'object':
        final nestedProps = prop['properties'] as Map<String, dynamic>?;
        final nestedRequired = prop['required'] as List<dynamic>?;
        final nestedParams =
            nestedProps?.entries
                .map(
                  (e) => _schemaPropertyToToolParam(
                    name: e.key,
                    prop: e.value as Map<String, dynamic>,
                    isRequired: nestedRequired?.contains(e.key) ?? false,
                  ),
                )
                .toList() ??
            [];
        return ToolParam.object(
          name,
          properties: nestedParams,
          description: description,
          required: isRequired,
        );
      default:
        return ToolParam.string(
          name,
          description: description,
          required: isRequired,
        );
    }
  }

  static const _hermesEnvelope = ToolCallEnvelope(
    '<tool_call>',
    '</tool_call>',
  );

  /// Tool-call envelopes each format may write into plain content.
  static final Map<ChatFormat, List<ToolCallEnvelope>> _envelopesByFormat = {
    // Gemma4Handler extracts tool calls from the full output itself.
    ChatFormat.gemma4: const [],
    ChatFormat.functionGemma: const [
      ToolCallEnvelope('<start_function_call>', '<end_function_call>'),
    ],
    ChatFormat.hermes: const [_hermesEnvelope],
    ChatFormat.deepseekV3: const [_hermesEnvelope],
  };

  static const _defaultEnvelopes = [
    _hermesEnvelope,
    ToolCallEnvelope('<|tool_call>', '<|tool_call|>'),
  ];

  @visibleForTesting
  static List<ToolCallEnvelope> toolCallEnvelopes(ChatFormat format) =>
      _envelopesByFormat[format] ?? _defaultEnvelopes;

  List<Part> _partsFrom(
    List<ScanSegment> segments,
    String Function() nextCallId,
    ChatFormat format,
  ) => [
    for (final segment in segments)
      switch (segment) {
        TextSegment(:final text) => TextPart(text),
        EnvelopeSegment(:final body) => _parseToolCall(
          body,
          nextCallId(),
          format,
        ),
      },
  ];

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
    // shutdown to release native resources.
    _engine = null;
  }
}
