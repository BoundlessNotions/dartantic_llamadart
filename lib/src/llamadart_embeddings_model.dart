import 'package:dartantic_interface/dartantic_interface.dart';
import 'package:llamadart/llamadart.dart';

import 'llama_engine_cache.dart';

/// An embeddings model backed by an embedding GGUF loaded through llamadart.
///
/// The engine is shared through [LlamaEngineCache] like a chat model's, keyed
/// by [modelPath] and load parameters that differ from any chat engine's.
class LlamadartEmbeddingsModel extends EmbeddingsModel<EmbeddingsModelOptions> {
  LlamadartEmbeddingsModel({
    required this.modelPath,
    required super.name,
    required super.defaultOptions,
  }) : super(
         dimensions: defaultOptions.dimensions,
         batchSize: defaultOptions.batchSize,
       );

  final String modelPath;

  ModelParams _modelParams() {
    const defaults = ModelParams();
    return ModelParams(
      // Encoder-only embedding models decode the whole input as one batch.
      batchSize: defaults.contextSize,
      // Lets llama.cpp embed a batch's texts as parallel sequences.
      maxParallelSequences: defaultOptions.batchSize ?? 1,
    );
  }

  @override
  Future<EmbeddingsResult> embedQuery(
    String query, {
    EmbeddingsModelOptions? options,
  }) async {
    final vector = await _withEngine((engine) => engine.embed(query));
    _checkDimensions(vector, options);
    return EmbeddingsResult(
      output: vector,
      finishReason: FinishReason.stop,
      metadata: const {},
      usage: const LanguageModelUsage(),
    );
  }

  @override
  Future<BatchEmbeddingsResult> embedDocuments(
    List<String> texts, {
    EmbeddingsModelOptions? options,
  }) async {
    final chunkSize = options?.batchSize ?? defaultOptions.batchSize;
    final vectors = await _withEngine((engine) async {
      if (chunkSize == null || chunkSize <= 0) return engine.embedBatch(texts);
      return [
        for (var start = 0; start < texts.length; start += chunkSize)
          ...await engine.embedBatch(
            texts.sublist(
              start,
              start + chunkSize < texts.length
                  ? start + chunkSize
                  : texts.length,
            ),
          ),
      ];
    });
    for (final vector in vectors) {
      _checkDimensions(vector, options);
    }
    return BatchEmbeddingsResult(
      output: vectors,
      finishReason: FinishReason.stop,
      metadata: const {},
      usage: const LanguageModelUsage(),
    );
  }

  Future<T> _withEngine<T>(
    Future<T> Function(LlamaEngine engine) action,
  ) async {
    final (handle, release) = await LlamaEngineCache.instance.acquireExclusive(
      modelPath,
      _modelParams(),
    );
    try {
      return await action(handle.engine);
    } finally {
      release();
    }
  }

  // A GGUF embedding has one native size, so a requested size that differs
  // can't be honored by truncation without changing what the vector means.
  void _checkDimensions(List<double> vector, EmbeddingsModelOptions? options) {
    final requested = options?.dimensions ?? defaultOptions.dimensions;
    if (requested != null && requested != vector.length) {
      throw ArgumentError.value(
        requested,
        'dimensions',
        'does not match the model, which produces ${vector.length}',
      );
    }
  }

  @override
  void dispose() {
    // The engine is shared through LlamaEngineCache; see
    // LlamaEngineCache.instance.disposeAll().
  }
}
