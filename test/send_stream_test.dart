import 'package:dartantic_interface/dartantic_interface.dart';
import 'package:dartantic_llamadart/dartantic_llamadart.dart';
import 'package:llamadart/llamadart.dart'
    show LlamaChatRole, LlamaInferenceException;
import 'package:test/test.dart';

import 'support/fake_llama_engine.dart';

LlamadartChatModel _model({
  String modelPath = '/models/fake.gguf',
  List<Tool<Object>>? tools,
  LlamadartChatOptions options = const LlamadartChatOptions(),
}) {
  final provider = LlamadartProvider(
    name: 'llamadart',
    displayName: 'Local Llama',
    modelPath: modelPath,
  );
  return provider.createChatModel(tools: tools, options: options)
      as LlamadartChatModel;
}

Future<List<Part>> _collectParts(Stream<ChatResult<ChatMessage>> stream) async {
  final parts = <Part>[];
  await for (final result in stream) {
    parts.addAll(result.output.parts);
  }
  return parts;
}

String _text(List<Part> parts) =>
    parts.whereType<TextPart>().map((p) => p.text).join();

void main() {
  late FakeEngineFactory factory;

  setUp(() {
    factory = FakeEngineFactory()..install();
  });

  tearDown(FakeEngineFactory.uninstall);

  test('loads the model through the engine factory', () async {
    final model = _model();
    await _collectParts(model.sendStream([ChatMessage.user('hi')]));

    expect(factory.engines, hasLength(1));
    expect(factory.last.loadedPath, '/models/fake.gguf');
    expect(factory.last.createCalls, hasLength(1));
  });

  test('streams plain text', () async {
    factory.onCreate = (engine) =>
        engine.enqueue(FakeGeneration([textChunk('Hel'), textChunk('lo!')]));
    final model = _model();

    final parts = await _collectParts(
      model.sendStream([ChatMessage.user('hi')]),
    );

    expect(_text(parts), 'Hello!');
  });

  group('text tool-call fallback', () {
    Future<List<Part>> run(List<String> chunks, {List<Tool<Object>>? tools}) {
      factory.onCreate = (engine) => engine.enqueue(
        FakeGeneration([for (final c in chunks) textChunk(c)]),
      );
      return _collectParts(
        _model(tools: tools).sendStream([ChatMessage.user('hi')]),
      );
    }

    test(
      'an envelope split mid-tag yields the text once and one call',
      () async {
        final parts = await run([
          'Sure. <tool_',
          'call>{"get_weather": {"city": ',
          '"Paris"}}</tool',
          '_call> Done.',
        ]);

        expect(_text(parts), 'Sure.  Done.');
        final call = parts.whereType<ToolPart>().single;
        expect(call.toolName, 'get_weather');
        expect(call.arguments, {'city': 'Paris'});
      },
    );

    test('a trailing < that never becomes a tag is flushed once', () async {
      final parts = await run(['a <', 'b <']);

      expect(_text(parts), 'a <b <');
      expect(parts.whereType<ToolPart>(), isEmpty);
    });

    test('an unterminated envelope is flushed as text once', () async {
      final parts = await run(['x <tool_call>{"a"', ': {}}']);

      expect(_text(parts), 'x <tool_call>{"a": {}}');
      expect(parts.whereType<ToolPart>(), isEmpty);
    });

    test('does not scan content when tools went to the engine', () async {
      final parts = await run(
        ['<tool_call>{"get_weather": {}}</tool_call>'],
        tools: [
          Tool<Map<String, dynamic>>(
            name: 'get_weather',
            description: 'Weather for a city',
            onCall: (_) => {},
          ),
        ],
      );

      expect(_text(parts), '<tool_call>{"get_weather": {}}</tool_call>');
      expect(parts.whereType<ToolPart>(), isEmpty);
    });
  });

  test('streams thinking deltas as ThinkingParts', () async {
    factory.onCreate = (engine) =>
        engine.enqueue(FakeGeneration([thinkingChunk('hmm'), textChunk('ok')]));
    final model = _model();

    final parts = await _collectParts(
      model.sendStream([ChatMessage.user('hi')]),
    );

    expect(parts.whereType<ThinkingPart>().map((p) => p.text), ['hmm']);
  });

  test('maps native toolCalls deltas to ToolParts', () async {
    factory.onCreate = (engine) => engine.enqueue(
      FakeGeneration([
        toolCallChunk(
          id: 'call_1',
          name: 'get_weather',
          arguments: '{"city":"Paris"}',
        ),
        finishChunk('tool_calls'),
      ]),
    );
    final model = _model();

    final parts = await _collectParts(
      model.sendStream([ChatMessage.user('weather?')]),
    );

    final calls = parts.whereType<ToolPart>().toList();
    expect(calls, hasLength(1));
    expect(calls.single.kind, ToolPartKind.call);
    expect(calls.single.callId, 'call_1');
    expect(calls.single.toolName, 'get_weather');
    expect(calls.single.arguments, {'city': 'Paris'});
  });

  group('message list', () {
    test('system and user messages both reach engine.create', () async {
      final model = _model();
      await _collectParts(
        model.sendStream([
          ChatMessage.system('Be terse.'),
          ChatMessage.user('hi'),
        ]),
      );

      final call = factory.last.createCalls.single;
      expect(call.messages.map((m) => m.role), [
        LlamaChatRole.system,
        LlamaChatRole.user,
      ]);
      expect(call.messages.first.content, 'Be terse.');
    });

    test('a trailing tool result is the last message sent', () async {
      final model = _model();
      await _collectParts(
        model.sendStream([
          ChatMessage.user('weather?'),
          ChatMessage(
            role: ChatMessageRole.model,
            parts: [
              ToolPart.call(
                callId: 'call_1',
                toolName: 'get_weather',
                arguments: {'city': 'Paris'},
              ),
            ],
          ),
          ChatMessage(
            role: ChatMessageRole.user,
            parts: [
              ToolPart.result(
                callId: 'call_1',
                toolName: 'get_weather',
                result: {'temp': 20},
              ),
            ],
          ),
        ]),
      );

      final call = factory.last.createCalls.single;
      expect(call.messages.map((m) => m.role), [
        LlamaChatRole.user,
        LlamaChatRole.assistant,
        LlamaChatRole.tool,
      ]);
    });

    test('the FunctionGemma trigger reaches the system message', () async {
      factory.metadata = {
        'tokenizer.chat_template': '<start_function_call>{{ messages }}',
      };
      final model = _model(
        tools: [
          Tool<Map<String, dynamic>>(
            name: 'get_weather',
            description: 'Weather for a city',
            onCall: (_) => {},
          ),
        ],
      );
      await _collectParts(
        model.sendStream([
          ChatMessage.system('Be terse.'),
          ChatMessage.user('hi'),
        ]),
      );

      final system = factory.last.createCalls.single.messages.first;
      expect(system.role, LlamaChatRole.system);
      expect(
        system.content,
        startsWith(
          'You are a model that can do function calling with the following '
          'functions',
        ),
      );
    });
  });

  group('shared engine', () {
    test('two models on one key share an engine', () async {
      final a = _model();
      final b = _model();
      await _collectParts(a.sendStream([ChatMessage.user('hi')]));
      await _collectParts(b.sendStream([ChatMessage.user('hi')]));

      expect(factory.engines, hasLength(1));
      expect(factory.last.createCalls, hasLength(2));
    });

    test('an eviction by one model is seen by the others', () async {
      final a = _model();
      final b = _model();
      await _collectParts(b.sendStream([ChatMessage.user('warm up')]));
      final first = factory.last;
      first.enqueue(
        FakeGeneration(const [], error: LlamaInferenceException('boom')),
      );

      await expectLater(
        _collectParts(a.sendStream([ChatMessage.user('hi')])),
        throwsA(isA<LlamaInferenceException>()),
      );
      expect(first.isDisposed, isTrue);

      // The disposed fake throws if touched, so this passing means b
      // acquired the replacement.
      await _collectParts(b.sendStream([ChatMessage.user('again')]));
      expect(factory.engines, hasLength(2));
      expect(factory.last.createCalls, hasLength(1));
    });
  });
}
