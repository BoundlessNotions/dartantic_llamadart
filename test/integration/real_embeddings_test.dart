@Tags(['integration'])
library;

import 'dart:io';

import 'package:dartantic_interface/dartantic_interface.dart';
import 'package:dartantic_llamadart/dartantic_llamadart.dart';
import 'package:test/test.dart';

/// Runs against an embedding GGUF (EmbeddingGemma 300M works) set in
/// `LLAMADART_TEST_EMBEDDING_MODEL`. Run with `dart test -P integration`.
void main() {
  final modelPath = Platform.environment['LLAMADART_TEST_EMBEDDING_MODEL'];
  final skip = modelPath == null || modelPath.isEmpty
      ? 'LLAMADART_TEST_EMBEDDING_MODEL is not set'
      : null;

  tearDown(LlamaEngineCache.instance.disposeAll);

  test('embeds queries and batches with a real model', () async {
    final model =
        LlamadartProvider(
          name: 'llamadart',
          displayName: 'Local Llama',
          modelPath: modelPath!,
          embeddingsModelPath: modelPath,
        ).createEmbeddingsModel(
          options: const EmbeddingsModelOptions(batchSize: 2),
        );

    final query = await model.embedQuery('A cat sat on the mat.');
    final docs = await model.embedDocuments([
      'A kitten rested on the rug.',
      'Quarterly revenue rose four percent.',
      'The dog slept by the door.',
    ]);

    expect(query.embeddings, isNotEmpty);
    expect(docs.count, 3);
    expect(docs.dimensions, query.embeddings.length);
    final scores = EmbeddingsModel.calculateSimilarity(
      query.embeddings,
      docs.embeddings,
    );
    expect(scores[0], greaterThan(scores[1]));
  }, skip: skip);
}
