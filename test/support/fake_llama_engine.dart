import 'dart:async';
import 'dart:collection';

import 'package:dartantic_llamadart/dartantic_llamadart.dart';
import 'package:llamadart/llamadart.dart';

/// A ChatML-style template that llamadart detects as [ChatFormat.hermes].
const hermesTemplate =
    "{% for m in messages %}<|im_start|>{{ m['role'] }}\n"
    "{{ m['content'] }}<|im_end|>\n{% endfor %}<tool_call>";

/// One scripted `create` call: [chunks] are replayed in order, then [error]
/// (if any) is thrown. The stream waits on [gate] before the first chunk and
/// on [hold] after the last, so tests can interleave concurrent generations.
class FakeGeneration {
  FakeGeneration(this.chunks, {this.error, this.gate, this.hold});

  final List<LlamaCompletionChunk> chunks;
  final Object? error;
  final Completer<void>? gate;
  final Completer<void>? hold;
}

/// Everything [FakeLlamaEngine.create] was called with.
class CreateCall {
  CreateCall({
    required this.messages,
    required this.params,
    required this.tools,
    required this.toolChoice,
    required this.enableThinking,
    required this.responseFormat,
  });

  final List<LlamaChatMessage> messages;
  final GenerationParams? params;
  final List<ToolDefinition>? tools;
  final ToolChoice? toolChoice;
  final bool enableThinking;
  final Map<String, dynamic>? responseFormat;
}

/// A [LlamaEngine] that replays scripted chunk streams instead of running a
/// native model. Any engine member not overridden here fails the test.
class FakeLlamaEngine implements LlamaEngine {
  FakeLlamaEngine({Map<String, String>? metadata})
    : metadata = metadata ?? {'tokenizer.chat_template': hermesTemplate};

  final Map<String, String> metadata;
  final Queue<FakeGeneration> _script = Queue();
  final List<CreateCall> createCalls = [];
  final Set<void Function()> _activeStops = {};

  final List<String> embedCalls = [];
  final List<List<String>> embedBatchCalls = [];

  /// The vector [embed] and [embedBatch] return for a text.
  List<double> Function(String text) vectorFor = (text) => [
    text.length.toDouble(),
    1,
  ];

  String? loadedPath;
  ModelParams? loadedParams;
  String? loadedMmproj;

  /// What the loaded projector can take, once one is loaded.
  bool visionCapable = true;
  bool audioCapable = true;
  int getMetadataCalls = 0;
  int cancelCalls = 0;
  int disposeCalls = 0;
  bool _ready = false;

  bool get isDisposed => disposeCalls > 0;

  /// Queues the stream the next `create` call replays.
  void enqueue(FakeGeneration generation) => _script.add(generation);

  void _checkNotDisposed(String member) {
    if (isDisposed) {
      throw StateError('FakeLlamaEngine.$member called after dispose');
    }
  }

  @override
  bool get isReady => _ready;

  @override
  Future<void> loadModel(
    String path, {
    ModelParams modelParams = const ModelParams(),
  }) async {
    loadedPath = path;
    loadedParams = modelParams;
    _ready = true;
  }

  @override
  Stream<LlamaCompletionChunk> create(
    List<LlamaChatMessage> messages, {
    GenerationParams? params,
    List<ToolDefinition>? tools,
    ToolChoice? toolChoice,
    bool parallelToolCalls = false,
    bool enableThinking = true,
    Map<String, dynamic>? responseFormat,
    String? sourceLangCode,
    String? targetLangCode,
    Map<String, dynamic>? chatTemplateKwargs,
    DateTime? templateNow,
  }) {
    _checkNotDisposed('create');
    createCalls.add(
      CreateCall(
        messages: List.of(messages),
        params: params,
        tools: tools,
        toolChoice: toolChoice,
        enableThinking: enableThinking,
        responseFormat: responseFormat,
      ),
    );
    final generation = _script.isEmpty
        ? FakeGeneration(const [])
        : _script.removeFirst();
    return _replay(generation);
  }

  // A controller rather than async*, like llama.cpp's backend, so cancelling
  // the subscription returns at once instead of waiting on a pending gate.
  Stream<LlamaCompletionChunk> _replay(FakeGeneration generation) {
    var stopped = false;
    late final StreamController<LlamaCompletionChunk> controller;
    // Like llama.cpp, cancelGeneration ends the stream without an error.
    void stop() {
      stopped = true;
      _activeStops.remove(stop);
      if (!controller.isClosed) controller.close();
    }

    controller = StreamController(
      onListen: () async {
        _activeStops.add(stop);
        await generation.gate?.future;
        for (final chunk in generation.chunks) {
          if (stopped) return;
          controller.add(chunk);
        }
        await generation.hold?.future;
        if (stopped) return;
        final error = generation.error;
        if (error != null) controller.addError(error);
        stop();
      },
      onCancel: () {
        stopped = true;
        _activeStops.remove(stop);
      },
    );
    return controller.stream;
  }

  @override
  Future<void> loadMultimodalProjector(String mmProjPath) async {
    _checkNotDisposed('loadMultimodalProjector');
    loadedMmproj = mmProjPath;
  }

  @override
  Future<bool> get supportsVision async =>
      loadedMmproj != null && visionCapable;

  @override
  Future<bool> get supportsAudio async => loadedMmproj != null && audioCapable;

  @override
  Future<List<double>> embed(String text, {bool normalize = true}) async {
    _checkNotDisposed('embed');
    embedCalls.add(text);
    return vectorFor(text);
  }

  @override
  Future<List<List<double>>> embedBatch(
    List<String> texts, {
    bool normalize = true,
  }) async {
    _checkNotDisposed('embedBatch');
    embedBatchCalls.add(List.of(texts));
    return texts.map(vectorFor).toList();
  }

  @override
  Future<Map<String, String>> getMetadata() async {
    _checkNotDisposed('getMetadata');
    getMetadataCalls++;
    return metadata;
  }

  @override
  void cancelGeneration() {
    cancelCalls++;
    for (final stop in List.of(_activeStops)) {
      stop();
    }
  }

  @override
  Future<void> dispose() async {
    disposeCalls++;
    _ready = false;
  }

  @override
  dynamic noSuchMethod(Invocation invocation) {
    throw UnimplementedError(
      'FakeLlamaEngine does not implement ${invocation.memberName}',
    );
  }
}

/// Points [LlamaEngineCache] at fresh [FakeLlamaEngine]s and records every
/// engine it builds, in order.
class FakeEngineFactory {
  /// Chat template metadata for new engines; null uses [hermesTemplate].
  Map<String, String>? metadata;

  /// Called with each new engine, e.g. to script its generations.
  void Function(FakeLlamaEngine engine)? onCreate;
  final List<FakeLlamaEngine> engines = [];

  FakeLlamaEngine get last => engines.last;

  void install() {
    LlamaEngineCache.instance.engineFactory = () {
      final engine = FakeLlamaEngine(metadata: metadata);
      onCreate?.call(engine);
      engines.add(engine);
      return engine;
    };
  }

  static Future<void> uninstall() async {
    await LlamaEngineCache.instance.disposeAll();
    LlamaEngineCache.instance.engineFactory =
        LlamaEngineCache.defaultEngineFactory;
  }
}

LlamaCompletionChunk _chunk(
  LlamaCompletionChunkDelta delta, {
  String? finishReason,
}) => LlamaCompletionChunk(
  id: 'fake',
  object: 'chat.completion.chunk',
  created: 0,
  model: 'fake',
  choices: [
    LlamaCompletionChunkChoice(
      index: 0,
      delta: delta,
      finishReason: finishReason,
    ),
  ],
);

LlamaCompletionChunk textChunk(String content) =>
    _chunk(LlamaCompletionChunkDelta(content: content));

LlamaCompletionChunk thinkingChunk(String thinking) =>
    _chunk(LlamaCompletionChunkDelta(thinking: thinking));

LlamaCompletionChunk toolCallChunk({
  required String name,
  required String arguments,
  String? id,
  int index = 0,
}) => _chunk(
  LlamaCompletionChunkDelta(
    toolCalls: [
      LlamaCompletionChunkToolCall(
        index: index,
        id: id,
        type: 'function',
        function: LlamaCompletionChunkFunction(
          name: name,
          arguments: arguments,
        ),
      ),
    ],
  ),
);

LlamaCompletionChunk finishChunk(String reason) =>
    _chunk(LlamaCompletionChunkDelta(), finishReason: reason);
