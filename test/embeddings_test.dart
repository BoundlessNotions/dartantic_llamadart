import 'package:dartantic_interface/dartantic_interface.dart';
import 'package:dartantic_llamadart/dartantic_llamadart.dart';
import 'package:test/test.dart';

import 'support/fake_llama_engine.dart';

LlamadartProvider _provider({
  String? embeddingsModelPath = '/models/embed.gguf',
}) => LlamadartProvider(
  name: 'llamadart',
  displayName: 'Local Llama',
  modelPath: '/models/chat.gguf',
  embeddingsModelPath: embeddingsModelPath,
);

void main() {
  late FakeEngineFactory factory;

  setUp(() {
    factory = FakeEngineFactory()..install();
  });

  tearDown(FakeEngineFactory.uninstall);

  test('embedQuery returns the engine vector', () async {
    final model = _provider().createEmbeddingsModel();

    final result = await model.embedQuery('hello');

    expect(result.embeddings, [5.0, 1.0]);
    expect(result.finishReason, FinishReason.stop);
    expect(factory.last.loadedPath, '/models/embed.gguf');
    expect(factory.last.embedCalls, ['hello']);
  });

  test('embedDocuments chunks by batchSize, in order', () async {
    final model = _provider().createEmbeddingsModel(
      options: const EmbeddingsModelOptions(batchSize: 2),
    );

    final result = await model.embedDocuments(['a', 'bb', 'ccc', 'dddd', 'e']);

    expect(factory.last.embedBatchCalls, [
      ['a', 'bb'],
      ['ccc', 'dddd'],
      ['e'],
    ]);
    expect(result.embeddings.map((v) => v.first), [1, 2, 3, 4, 1]);
    expect(factory.last.loadedParams!.maxParallelSequences, 2);
  });

  test('rejects a dimensions request the model cannot meet', () async {
    final model = _provider().createEmbeddingsModel(
      options: const EmbeddingsModelOptions(dimensions: 768),
    );

    await expectLater(model.embedQuery('hello'), throwsArgumentError);
  });

  test('chat and embeddings on one path load separate engines', () async {
    final provider = LlamadartProvider(
      name: 'llamadart',
      displayName: 'Local Llama',
      modelPath: '/models/both.gguf',
      embeddingsModelPath: '/models/both.gguf',
    );

    await provider.createChatModel().sendStream([
      ChatMessage.user('hi'),
    ]).drain<void>();
    await provider.createEmbeddingsModel().embedQuery('hi');

    expect(factory.engines, hasLength(2));
  });

  test('a provider without an embeddings model refuses to make one', () {
    expect(
      () => _provider(embeddingsModelPath: null).createEmbeddingsModel(),
      throwsStateError,
    );
  });

  test('listModels advertises embeddings only when configured', () async {
    final kinds = await _provider().listModels().map((m) => m.kinds).toList();
    expect(kinds, [
      {ModelKind.chat},
      {ModelKind.embeddings},
    ]);

    final chatOnly = await _provider(
      embeddingsModelPath: null,
    ).listModels().map((m) => m.kinds).toList();
    expect(chatOnly, [
      {ModelKind.chat},
    ]);
  });
}
