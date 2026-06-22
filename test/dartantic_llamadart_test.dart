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
