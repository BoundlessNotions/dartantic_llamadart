import 'dart:async';
import 'dart:typed_data';

import 'package:dartantic_interface/dartantic_interface.dart';
import 'package:dartantic_llamadart/dartantic_llamadart.dart';
import 'package:llamadart/llamadart.dart'
    show
        LlamaChatRole,
        LlamaImageContent,
        LlamaInferenceException,
        LlamaUnsupportedException;
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

    test('parses the Hermes {name, arguments} shape', () async {
      final parts = await run([
        '<tool_call>{"name": "get_weather", "arguments": {"city": "Paris"}}'
            '</tool_call>',
      ]);

      final call = parts.whereType<ToolPart>().single;
      expect(call.toolName, 'get_weather');
      expect(call.arguments, {'city': 'Paris'});
    });

    test('decodes arguments given as a JSON string', () async {
      final parts = await run([
        r'<tool_call>{"name": "get_weather", "arguments": "{\"city\": \"Paris\"}"}'
            '</tool_call>',
      ]);

      expect(parts.whereType<ToolPart>().single.arguments, {'city': 'Paris'});
    });

    test('parses a Gemma call expression', () async {
      factory.metadata = {
        'tokenizer.chat_template': '<|turn>{{ messages }}<turn|>',
      };
      final parts = await run([
        'Checking. <|tool_call>call:get_weather{city:<|"|>Paris<|"|>,',
        'days:3}<tool_call|>',
      ]);

      expect(_text(parts), 'Checking. ');
      final call = parts.whereType<ToolPart>().single;
      expect(call.toolName, 'get_weather');
      expect(call.arguments, {'city': 'Paris', 'days': 3});
    });

    test('keeps an envelope that is not a tool call as text', () async {
      final parts = await run(['<tool_call>not json</tool_call>']);

      expect(_text(parts), '<tool_call>not json</tool_call>');
      expect(parts.whereType<ToolPart>(), isEmpty);
    });

    test('with tools, yields a native call once despite a raw envelope in '
        'content', () async {
      factory.onCreate = (engine) => engine.enqueue(
        FakeGeneration([
          textChunk('<tool_call>{"name": "get_weather", "arguments": {}}'),
          textChunk('</tool_call>'),
          toolCallChunk(id: 'call_1', name: 'get_weather', arguments: '{}'),
        ]),
      );
      final parts = await _collectParts(
        _model(
          tools: [
            Tool<Map<String, dynamic>>(
              name: 'get_weather',
              description: 'Weather for a city',
              onCall: (_) => {},
            ),
          ],
        ).sendStream([ChatMessage.user('hi')]),
      );

      expect(parts.whereType<ToolPart>().single.callId, 'call_1');
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

  group('generation lock', () {
    test('a second generation waits for the first to finish', () async {
      final gate = Completer<void>();
      factory.onCreate = (engine) => engine
        ..enqueue(FakeGeneration([textChunk('one')], gate: gate))
        ..enqueue(FakeGeneration([textChunk('two')]));

      final first = _collectParts(_model().sendStream([ChatMessage.user('1')]));
      await pumpEventQueue();
      final second = _collectParts(
        _model().sendStream([ChatMessage.user('2')]),
      );
      await pumpEventQueue();

      expect(factory.last.createCalls, hasLength(1));

      gate.complete();
      expect(_text(await first), 'one');
      expect(_text(await second), 'two');
      expect(factory.last.createCalls, hasLength(2));
      expect(factory.last.cancelCalls, 0);
    });

    test('cancelling mid-stream cancels the native generation once and '
        'lets the next caller in', () async {
      final hold = Completer<void>();
      factory.onCreate = (engine) => engine
        ..enqueue(FakeGeneration([textChunk('one')], hold: hold))
        ..enqueue(FakeGeneration([textChunk('two')]));

      final firstChunk = Completer<void>();
      final subscription = _model()
          .sendStream([ChatMessage.user('1')])
          .listen((_) => firstChunk.complete());
      await firstChunk.future;
      final second = _collectParts(
        _model().sendStream([ChatMessage.user('2')]),
      );
      await pumpEventQueue();
      expect(factory.last.createCalls, hasLength(1));

      await subscription.cancel();

      expect(_text(await second), 'two');
      expect(factory.last.cancelCalls, 1);
    });

    test('disposing the model stops its generation', () async {
      final hold = Completer<void>();
      factory.onCreate = (engine) =>
          engine.enqueue(FakeGeneration([textChunk('one')], hold: hold));
      final model = _model();

      final parts = _collectParts(model.sendStream([ChatMessage.user('1')]));
      await pumpEventQueue();
      model.dispose();

      expect(_text(await parts), 'one');
      expect(factory.last.cancelCalls, 1);
    });

    test(
      'a request queued behind a failure gets the reloaded engine',
      () async {
        final gate = Completer<void>();
        factory.onCreate = (engine) {
          if (factory.engines.isEmpty) {
            engine.enqueue(
              FakeGeneration(
                const [],
                gate: gate,
                error: LlamaInferenceException('boom'),
              ),
            );
          }
        };

        final first = _collectParts(
          _model().sendStream([ChatMessage.user('1')]),
        );
        await pumpEventQueue();
        final second = _collectParts(
          _model().sendStream([ChatMessage.user('2')]),
        );
        await pumpEventQueue();
        gate.complete();

        await expectLater(first, throwsA(isA<LlamaInferenceException>()));
        await second;
        expect(factory.engines, hasLength(2));
        expect(factory.engines.first.createCalls, hasLength(1));
        expect(factory.last.createCalls, hasLength(1));
      },
    );

    test('different engines generate concurrently', () async {
      final gate = Completer<void>();
      factory.onCreate = (engine) {
        if (factory.engines.isEmpty) {
          engine.enqueue(FakeGeneration([textChunk('slow')], gate: gate));
        }
      };

      final slow = _collectParts(
        _model(
          options: const LlamadartChatOptions(nCtx: 1024),
        ).sendStream([ChatMessage.user('1')]),
      );
      await pumpEventQueue();
      await _collectParts(
        _model(
          options: const LlamadartChatOptions(nCtx: 2048),
        ).sendStream([ChatMessage.user('2')]),
      );

      expect(factory.engines, hasLength(2));
      gate.complete();
      expect(_text(await slow), 'slow');
    });
  });

  group('eviction', () {
    Future<void> failWith(Object error) async {
      factory.onCreate = (engine) {
        if (factory.engines.isEmpty) {
          engine.enqueue(FakeGeneration(const [], error: error));
        }
      };
      await expectLater(
        _collectParts(_model().sendStream([ChatMessage.user('1')])),
        throwsA(same(error)),
      );
      await _collectParts(_model().sendStream([ChatMessage.user('2')]));
    }

    test('keeps the engine on a request-shape error', () async {
      await failWith(LlamaUnsupportedException('no'));

      expect(factory.engines, hasLength(1));
      expect(factory.last.createCalls, hasLength(2));
    });

    test('keeps the engine when the prompt overflows the context', () async {
      await failWith(
        LlamaInferenceException(
          'Generation failed',
          Exception('Tokenization failed or prompt too long'),
        ),
      );

      expect(factory.engines, hasLength(1));
    });

    test('reloads the engine after an inference failure', () async {
      await failWith(LlamaInferenceException('Generation failed'));

      expect(factory.engines, hasLength(2));
      expect(factory.engines.first.isDisposed, isTrue);
    });

    test('reloads the engine after a raw backend error', () async {
      await failWith(StateError('native'));

      expect(factory.engines, hasLength(2));
    });

    test('keeps the engine when our own conversion throws', () async {
      final model = _model(
        tools: [
          Tool<Map<String, dynamic>>(
            name: 'broken',
            description: 'A boolean property schema',
            inputSchema: Schema.fromMap({
              'type': 'object',
              'properties': {'x': true},
            }),
            onCall: (_) => {},
          ),
        ],
      );

      await expectLater(
        _collectParts(model.sendStream([ChatMessage.user('1')])),
        throwsA(isA<TypeError>()),
      );
      await _collectParts(_model().sendStream([ChatMessage.user('2')]));

      expect(factory.engines, hasLength(1));
    });
  });

  group('per-engine state', () {
    test('detects the chat format once per engine', () async {
      await _collectParts(_model().sendStream([ChatMessage.user('1')]));
      await _collectParts(_model().sendStream([ChatMessage.user('2')]));

      expect(factory.last.getMetadataCalls, 1);
    });

    test('carries finishReason from the chunk that reports it', () async {
      factory.onCreate = (engine) => engine.enqueue(
        FakeGeneration([
          toolCallChunk(name: 'get_weather', arguments: '{}'),
          finishChunk('tool_calls'),
        ]),
      );

      final results = await _model().sendStream([
        ChatMessage.user('1'),
      ]).toList();

      expect(results.last.finishReason, FinishReason.toolCalls);
    });

    test('repeats finishReason on a flushed tail', () async {
      factory.onCreate = (engine) => engine.enqueue(
        FakeGeneration([textChunk('a <'), finishChunk('stop')]),
      );

      final results = await _model().sendStream([
        ChatMessage.user('1'),
      ]).toList();

      expect(results.last.output.text, '<');
      expect(results.last.finishReason, FinishReason.stop);
    });
  });

  group('structured output', () {
    final schema = Schema.fromMap({
      'type': 'object',
      'properties': {
        'ok': {'type': 'boolean'},
      },
      'required': ['ok'],
    });

    test('sends a json_schema responseFormat on GGUF', () async {
      await _collectParts(
        _model().sendStream([ChatMessage.user('1')], outputSchema: schema),
      );

      final call = factory.last.createCalls.single;
      expect(call.responseFormat, {
        'type': 'json_schema',
        'json_schema': {
          'schema': {
            'type': 'object',
            'properties': {
              'ok': {'type': 'boolean'},
            },
            'required': ['ok'],
          },
        },
      });
      expect(call.params!.grammar, isNull);
    });

    test('sends no responseFormat on LiteRT-LM', () async {
      await _collectParts(
        _model(
          modelPath: '/models/fake.litertlm',
        ).sendStream([ChatMessage.user('1')], outputSchema: schema),
      );

      expect(factory.last.createCalls.single.responseFormat, isNull);
    });
  });

  group('options', () {
    test('thinking is off unless the provider was asked for it', () async {
      await _collectParts(_model().sendStream([ChatMessage.user('1')]));
      expect(factory.last.createCalls.last.enableThinking, isFalse);

      final thinking = LlamadartProvider(
        name: 'llamadart',
        displayName: 'Local Llama',
        modelPath: '/models/fake.gguf',
      ).createChatModel(enableThinking: true);
      await _collectParts(thinking.sendStream([ChatMessage.user('2')]));
      expect(factory.last.createCalls.last.enableThinking, isTrue);
    });

    test('per-call options merge over the defaults', () async {
      await _collectParts(
        _model(
          options: const LlamadartChatOptions(topK: 7, maxTokens: 64),
        ).sendStream([
          ChatMessage.user('1'),
        ], options: const LlamadartChatOptions(temp: 0.2)),
      );

      final params = factory.last.createCalls.single.params!;
      expect(params.temp, 0.2);
      expect(params.topK, 7);
      expect(params.maxTokens, 64);
    });

    test('a per-call load-time field fails the call', () async {
      await expectLater(
        _collectParts(
          _model().sendStream([
            ChatMessage.user('1'),
          ], options: const LlamadartChatOptions(nCtx: 1024)),
        ),
        throwsArgumentError,
      );
    });
  });

  group('media parts', () {
    final image = ChatMessage(
      role: ChatMessageRole.user,
      parts: [
        const TextPart('What is this?'),
        DataPart(Uint8List.fromList([1, 2, 3]), mimeType: 'image/png'),
      ],
    );

    test('a GGUF model without a projector rejects an image', () async {
      await expectLater(
        _collectParts(_model().sendStream([image])),
        throwsA(
          isA<UnsupportedError>().having(
            (e) => e.message,
            'message',
            contains('mmprojPath'),
          ),
        ),
      );
    });

    test('a loaded projector takes the image', () async {
      final model = _model(
        options: const LlamadartChatOptions(mmprojPath: '/models/mmproj.gguf'),
      );

      await _collectParts(model.sendStream([image]));

      expect(factory.last.loadedMmproj, '/models/mmproj.gguf');
      final parts = factory.last.createCalls.single.messages.single.parts;
      expect(parts.last, isA<LlamaImageContent>());
    });

    test('a projector without audio support rejects audio', () async {
      factory.onCreate = (engine) => engine.audioCapable = false;
      final model = _model(
        options: const LlamadartChatOptions(mmprojPath: '/models/mmproj.gguf'),
      );

      await expectLater(
        _collectParts(
          model.sendStream([
            ChatMessage(
              role: ChatMessageRole.user,
              parts: [DataPart(Uint8List(2), mimeType: 'audio/wav')],
            ),
          ]),
        ),
        throwsUnsupportedError,
      );
    });

    test('LiteRT-LM bundles handle media without a projector', () async {
      final model = _model(modelPath: '/models/fake.litertlm');

      await _collectParts(model.sendStream([image]));

      expect(factory.last.loadedMmproj, isNull);
      expect(factory.last.createCalls, hasLength(1));
    });

    test('the projector is part of the engine cache key', () async {
      await _collectParts(
        _model(
          options: const LlamadartChatOptions(mmprojPath: '/models/a.gguf'),
        ).sendStream([ChatMessage.user('hi')]),
      );
      await _collectParts(
        _model(
          options: const LlamadartChatOptions(mmprojPath: '/models/b.gguf'),
        ).sendStream([ChatMessage.user('hi')]),
      );

      expect(factory.engines, hasLength(2));
      expect(factory.engines.map((e) => e.loadedMmproj), [
        '/models/a.gguf',
        '/models/b.gguf',
      ]);
    });
  });
}
