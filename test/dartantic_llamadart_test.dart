import 'dart:typed_data';

import 'package:dartantic_interface/dartantic_interface.dart';
import 'package:dartantic_llamadart/dartantic_llamadart.dart';

import 'package:llamadart/llamadart.dart';
import 'package:test/test.dart';

void main() {
  group('LlamadartChatOptions', () {
    test('can be created with options', () {
      const options = LlamadartChatOptions(temp: 0.5, nCtx: 1024);
      expect(options.temp, 0.5);
      expect(options.nCtx, 1024);
    });

    test('copyWith works', () {
      const options = LlamadartChatOptions(temp: 0.5);
      final updated = options.copyWith(temp: 0.8, nCtx: 512);
      expect(updated.temp, 0.8);
      expect(updated.nCtx, 512);
    });

    test('speculativeDecoding defaults to null and round-trips', () {
      const options = LlamadartChatOptions(temp: 0.5);
      expect(options.speculativeDecoding, isNull);

      final updated = options.copyWith(speculativeDecoding: true);
      expect(updated.speculativeDecoding, isTrue);
      // unrelated fields are preserved
      expect(updated.temp, 0.5);
    });
  });

  group('LlamadartChatOptions.mergedWith', () {
    const defaults = LlamadartChatOptions(
      nCtx: 4096,
      temp: 0.7,
      topK: 20,
      maxTokens: 256,
      preferredBackend: GpuBackend.vulkan,
    );

    test('keeps defaults the per-call options leave unset', () {
      final merged = defaults.mergedWith(const LlamadartChatOptions(temp: 0.2));

      expect(merged.temp, 0.2);
      expect(merged.topK, 20);
      expect(merged.maxTokens, 256);
      expect(merged.nCtx, 4096);
      expect(merged.preferredBackend, GpuBackend.vulkan);
    });

    test('rejects a per-call load-time field that differs', () {
      expect(
        () => defaults.mergedWith(const LlamadartChatOptions(nCtx: 1024)),
        throwsArgumentError,
      );
      expect(
        () => defaults.mergedWith(
          const LlamadartChatOptions(preferredBackend: GpuBackend.cpu),
        ),
        throwsArgumentError,
      );
    });

    test('accepts a per-call load-time field equal to the default', () {
      expect(
        defaults.mergedWith(const LlamadartChatOptions(nCtx: 4096)).nCtx,
        4096,
      );
    });
  });

  group('LlamadartProvider', () {
    test('initializes correctly', () {
      final provider = LlamadartProvider(
        name: 'llamadart',
        displayName: 'Local Llama',
        modelPath: '/path/to/model.gguf',
      );

      expect(provider.name, 'llamadart');
      expect(provider.modelPath, '/path/to/model.gguf');
    });

    test('createChatModel returns a model', () {
      final provider = LlamadartProvider(
        name: 'llamadart',
        displayName: 'Local Llama',
        modelPath: '/path/to/model.gguf',
      );

      final model = provider.createChatModel();
      expect(model.name, 'default');
    });
  });

  group('LlamadartChatModel', () {
    late LlamadartProvider provider;
    late LlamadartChatModel model;

    setUp(() {
      provider = LlamadartProvider(
        name: 'llamadart',
        displayName: 'Local Llama',
        modelPath: '/path/to/model.gguf',
      );
      model = provider.createChatModel() as LlamadartChatModel;
    });

    test('toLlamaMessage prepends trigger for FunctionGemma with tools', () {
      final msg = ChatMessage.system('You are a helpful assistant.');
      final llamaMsg = model.toLlamaMessage(
        msg,
        format: ChatFormat.functionGemma,
        hasTools: true,
      );

      expect(
        llamaMsg.content,
        startsWith(
          'You are a model that can do function calling with the following functions',
        ),
      );
      expect(llamaMsg.content, contains('You are a helpful assistant.'));
    });

    test('toLlamaMessage does not prepend trigger if already present', () {
      const trigger =
          'You are a model that can do function calling with the following functions';
      final msg = ChatMessage.system('$trigger\n\nExisting system prompt.');
      final llamaMsg = model.toLlamaMessage(
        msg,
        format: ChatFormat.functionGemma,
        hasTools: true,
      );

      // Should not duplicate the trigger
      final occurrences = trigger.allMatches(llamaMsg.content).length;
      expect(occurrences, 1);
    });

    test('toLlamaMessage does not prepend trigger for non-FunctionGemma', () {
      final msg = ChatMessage.system('You are a helpful assistant.');
      final llamaMsg = model.toLlamaMessage(
        msg,
        format: ChatFormat.llama3,
        hasTools: true,
      );

      expect(
        llamaMsg.content,
        isNot(
          startsWith(
            'You are a model that can do function calling with the following functions',
          ),
        ),
      );
      expect(llamaMsg.content, 'You are a helpful assistant.');
    });

    test('toLlamaMessage does not prepend trigger if no tools', () {
      final msg = ChatMessage.system('You are a helpful assistant.');
      final llamaMsg = model.toLlamaMessage(
        msg,
        format: ChatFormat.functionGemma,
        hasTools: false,
      );

      expect(
        llamaMsg.content,
        isNot(
          startsWith(
            'You are a model that can do function calling with the following functions',
          ),
        ),
      );
      expect(llamaMsg.content, 'You are a helpful assistant.');
    });

    group('tool schema conversion', () {
      Tool<Map<String, dynamic>> tool(Map<String, Object?> schema) =>
          Tool<Map<String, dynamic>>(
            name: 't',
            description: 'A tool.',
            inputSchema: Schema.fromMap(schema),
            onCall: (_) => {},
          );

      Map<String, dynamic> param(Map<String, Object?> prop) => model
          .toolDefinitionFor(
            tool({
              'type': 'object',
              'properties': {'p': prop},
            }),
          )
          .parameters
          .single
          .toJsonSchema();

      test('JSON-encodes non-string examples', () {
        final definition = model.toolDefinitionFor(
          tool({
            'type': 'object',
            'examples': [
              'plain',
              {'city': 'Paris'},
            ],
          }),
        );

        expect(definition.description, contains('- plain'));
        expect(definition.description, contains('- {"city":"Paris"}'));
      });

      test('keeps the declared type for a non-string enum', () {
        final schema = param({
          'type': 'integer',
          'description': 'Days.',
          'enum': [1, 2, 3],
        });

        expect(schema['type'], 'integer');
        expect(schema['description'], 'Days. Allowed values: 1, 2, 3.');
      });

      test('uses the non-null member of a nullable type', () {
        expect(
          param({
            'type': ['string', 'null'],
          })['type'],
          'string',
        );
        expect(
          param({
            'type': ['null', 'integer'],
          })['type'],
          'integer',
        );
      });

      test('maps a null-only type to a described string', () {
        final schema = param({
          'type': ['null'],
        });

        expect(schema['type'], 'string');
        expect(schema['description'], 'Must be null.');
      });

      test('maps a union type to a string noting the JSON types', () {
        final schema = param({
          'type': ['string', 'integer'],
        });

        expect(schema['type'], 'string');
        expect(schema['description'], 'JSON type: one of string, integer.');
      });

      test('converts nested object properties', () {
        final schema = param({
          'type': 'object',
          'properties': {
            'n': {'type': 'integer'},
          },
          'required': ['n'],
        });

        expect(schema['properties'], {
          'n': {'type': 'integer'},
        });
        expect(schema['required'], ['n']);
      });
    });

    group('toLlamaMessage part mapping', () {
      List<LlamaContentPart> convert(List<Part> parts) => model
          .toLlamaMessage(
            ChatMessage(role: ChatMessageRole.model, parts: parts),
            format: ChatFormat.hermes,
            hasTools: false,
          )
          .parts;

      test('passes thinking through as LlamaThinkingContent', () {
        final parts = convert([
          const ThinkingPart('secret'),
          const TextPart('hi'),
        ]);

        expect(parts, [isA<LlamaThinkingContent>(), isA<LlamaTextContent>()]);
        expect((parts.first as LlamaThinkingContent).thinking, 'secret');
      });

      test('maps image and audio data to media content', () {
        final image = Uint8List.fromList([1, 2, 3]);
        final audio = Uint8List.fromList([4, 5]);
        final parts = convert([
          DataPart(image, mimeType: 'image/png'),
          DataPart(audio, mimeType: 'audio/wav'),
        ]);

        expect((parts[0] as LlamaImageContent).bytes, image);
        expect((parts[1] as LlamaAudioContent).bytes, audio);
      });

      test('rejects data llamadart has no content type for', () {
        expect(
          () => convert([DataPart(Uint8List(1), mimeType: 'application/pdf')]),
          throwsUnsupportedError,
        );
      });

      test('maps file and http links by mime type', () {
        final parts = convert([
          LinkPart(Uri.file('/tmp/cat.png'), mimeType: 'image/png'),
          LinkPart(Uri.file('/tmp/meow.wav'), mimeType: 'audio/wav'),
          LinkPart(
            Uri.parse('https://example.com/cat.png'),
            mimeType: 'image/png',
          ),
        ]);

        expect((parts[0] as LlamaImageContent).path, '/tmp/cat.png');
        expect((parts[1] as LlamaAudioContent).path, '/tmp/meow.wav');
        expect(
          (parts[2] as LlamaImageContent).url,
          'https://example.com/cat.png',
        );
      });

      test('rejects links it cannot load', () {
        expect(
          () => convert([
            LinkPart(
              Uri.parse('https://example.com/meow.wav'),
              mimeType: 'audio/wav',
            ),
          ]),
          throwsUnsupportedError,
        );
        expect(
          () => convert([LinkPart(Uri.file('/tmp/notes'))]),
          throwsUnsupportedError,
        );
      });
    });

    test('buildGenerationParams drops llama.cpp-only knobs for LiteRT-LM', () {
      const options = LlamadartChatOptions(
        temp: 0.3,
        topK: 20,
        topP: 0.95,
        repeatPenalty: 1.15,
        minP: 0.05,
        maxTokens: 256,
        speculativeDecoding: true,
      );

      final litert = model.buildGenerationParams(options, isLiteRtLm: true);
      // Supported knobs pass through.
      expect(litert.temp, 0.3);
      expect(litert.topK, 20);
      expect(litert.topP, 0.95);
      expect(litert.maxTokens, 256);
      expect(litert.speculativeDecoding, isTrue);
      // llama.cpp-only knobs are forced to GenerationParams defaults so the
      // LiteRT-LM backend does not reject the request.
      const defaults = GenerationParams();
      expect(litert.minP, defaults.minP);
      expect(litert.penalty, defaults.penalty);

      // GGUF/llama.cpp keeps the caller's values.
      final gguf = model.buildGenerationParams(options, isLiteRtLm: false);
      expect(gguf.minP, 0.05);
      expect(gguf.penalty, 1.15);
    });

    test('buildGenerationParams leaves maxTokens at llamadart\'s default', () {
      final params = model.buildGenerationParams(
        const LlamadartChatOptions(),
        isLiteRtLm: false,
      );

      expect(params.maxTokens, const GenerationParams().maxTokens);
      expect(params.maxTokens, greaterThan(0));
    });

    test('buildGenerationParams keeps default topK under a per-call temp', () {
      final params = model.buildGenerationParams(
        const LlamadartChatOptions(
          topK: 7,
        ).mergedWith(const LlamadartChatOptions(temp: 0.2)),
        isLiteRtLm: false,
      );

      expect(params.temp, 0.2);
      expect(params.topK, 7);
    });

    test('buildGenerationParams passes the streaming knobs through', () {
      const defaults = GenerationParams();
      final unset = model.buildGenerationParams(
        const LlamadartChatOptions(),
        isLiteRtLm: false,
      );
      expect(unset.reusePromptPrefix, defaults.reusePromptPrefix);

      final set = model.buildGenerationParams(
        LlamadartChatOptions(
          reusePromptPrefix: !defaults.reusePromptPrefix,
          streamBatchTokenThreshold: 3,
          streamBatchByteThreshold: 5,
        ),
        isLiteRtLm: false,
      );
      expect(set.reusePromptPrefix, !defaults.reusePromptPrefix);
      expect(set.streamBatchTokenThreshold, 3);
      expect(set.streamBatchByteThreshold, 5);
    });

    test('buildGenerationParams maps an MTP draft path to a GGUF spec config', () {
      const options = LlamadartChatOptions(
        maxTokens: 128,
        mtpDraftModelPath: '/models/mtp-draft.gguf',
      );

      // GGUF: the draft path produces an explicit draft-mtp speculative config.
      final gguf = model.buildGenerationParams(options, isLiteRtLm: false);
      expect(gguf.speculativeDecodingConfig, isNotNull);
      expect(
        gguf.speculativeDecodingConfig!.strategy,
        SpeculativeDecodingStrategy.mtp,
      );
      expect(
        gguf.speculativeDecodingConfig!.draftModelPath,
        '/models/mtp-draft.gguf',
      );
      // Defaults to a draft token max of 1 (must match the reserved rollback).
      expect(gguf.speculativeDecodingConfig!.draftTokenMax, 1);
      // The explicit config supersedes the legacy bool on the GGUF path.
      expect(gguf.speculativeDecoding, isFalse);

      // LiteRT-LM ignores the GGUF draft path entirely.
      final litert = model.buildGenerationParams(options, isLiteRtLm: true);
      expect(litert.speculativeDecodingConfig, isNull);
    });
  });
}
