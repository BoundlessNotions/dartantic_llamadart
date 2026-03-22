import 'package:dartantic_llamadart/dartantic_llamadart.dart';
import 'package:test/test.dart';

void main() {
  group('LlamadartChatOptions', () {
    test('can be created with options', () {
      const options = LlamadartChatOptions(
        temp: 0.5,
        nCtx: 1024,
      );
      expect(options.temp, 0.5);
      expect(options.nCtx, 1024);
    });

    test('copyWith works', () {
      const options = LlamadartChatOptions(temp: 0.5);
      final updated = options.copyWith(temp: 0.8, nCtx: 512);
      expect(updated.temp, 0.8);
      expect(updated.nCtx, 512);
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
      expect(model.modelName, 'default');
    });
  });
}
