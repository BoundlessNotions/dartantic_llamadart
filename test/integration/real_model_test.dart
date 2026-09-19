@Tags(['integration'])
library;

import 'dart:async';
import 'dart:convert';
import 'dart:io';

import 'package:dartantic_interface/dartantic_interface.dart';
import 'package:dartantic_llamadart/dartantic_llamadart.dart';
import 'package:llamadart/llamadart.dart';
import 'package:test/test.dart';

/// A real engine that records what `create` received and streamed, so tests
/// can compare the adapter's output with the engine's.
class RecordingEngine extends LlamaEngine {
  RecordingEngine() : super(LlamaBackend());

  final List<List<LlamaChatMessage>> requests = [];
  final StringBuffer streamedContent = StringBuffer();

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
  }) async* {
    requests.add(List.of(messages));
    await for (final chunk in super.create(
      messages,
      params: params,
      tools: tools,
      toolChoice: toolChoice,
      parallelToolCalls: parallelToolCalls,
      enableThinking: enableThinking,
      responseFormat: responseFormat,
      sourceLangCode: sourceLangCode,
      targetLangCode: targetLangCode,
      chatTemplateKwargs: chatTemplateKwargs,
      templateNow: templateNow,
    )) {
      streamedContent.write(chunk.choices.firstOrNull?.delta.content ?? '');
      yield chunk;
    }
  }
}

/// Runs against a small instruct GGUF (SmolLM2-135M-Instruct Q4_K_M works) set
/// in `LLAMADART_TEST_MODEL`. Run with `dart test -P integration`.
void main() {
  final modelPath = Platform.environment['LLAMADART_TEST_MODEL'];
  final skip = modelPath == null || modelPath.isEmpty
      ? 'LLAMADART_TEST_MODEL is not set'
      : null;

  late RecordingEngine engine;

  setUp(() {
    LlamaEngineCache.instance.engineFactory = () => engine = RecordingEngine();
  });

  tearDown(() async {
    await LlamaEngineCache.instance.disposeAll();
    LlamaEngineCache.instance.engineFactory =
        LlamaEngineCache.defaultEngineFactory;
  });

  LlamadartChatModel model() =>
      LlamadartProvider(
            name: 'llamadart',
            displayName: 'Local Llama',
            modelPath: modelPath!,
          ).createChatModel(
            options: const LlamadartChatOptions(
              nCtx: 1024,
              nGpuLayers: 0,
              maxTokens: 24,
              temp: 0,
            ),
          )
          as LlamadartChatModel;

  test('streams a reply from a real model', () async {
    final texts = <String>[];
    await for (final result in model().sendStream([
      ChatMessage.user('Say hello in five words.'),
    ])) {
      texts.addAll(
        result.output.parts.whereType<TextPart>().map((p) => p.text),
      );
    }

    expect(texts.join(), isNotEmpty);
    expect(texts.join(), engine.streamedContent.toString());
  }, skip: skip);

  test('a system prompt reaches the real engine', () async {
    await model().sendStream([
      ChatMessage.system('Answer in French.'),
      ChatMessage.user('Say hello.'),
    ]).drain<void>();

    expect(engine.requests.single.map((m) => m.role), [
      LlamaChatRole.system,
      LlamaChatRole.user,
    ]);
  }, skip: skip);

  test('cancelling mid-stream frees the engine for the next call', () async {
    final firstChunk = Completer<void>();
    final subscription = model()
        .sendStream([ChatMessage.user('Count from one to fifty.')])
        .listen((_) {
          if (!firstChunk.isCompleted) firstChunk.complete();
        });
    await firstChunk.future;
    await subscription.cancel();

    final texts = <String>[];
    await for (final result in model().sendStream([
      ChatMessage.user('Say hello.'),
    ])) {
      texts.addAll(
        result.output.parts.whereType<TextPart>().map((p) => p.text),
      );
    }
    expect(texts.join(), isNotEmpty);
  }, skip: skip);

  test('outputSchema constrains a real model to the schema', () async {
    final texts = <String>[];
    await for (final result in model().sendStream(
      [ChatMessage.user('Is the sky blue? Answer in JSON.')],
      outputSchema: Schema.fromMap({
        'type': 'object',
        'properties': {
          'ok': {'type': 'boolean'},
        },
        'required': ['ok'],
      }),
    )) {
      texts.addAll(
        result.output.parts.whereType<TextPart>().map((p) => p.text),
      );
    }

    final decoded = jsonDecode(texts.join()) as Map<String, dynamic>;
    expect(decoded['ok'], isA<bool>());
  }, skip: skip);
}
