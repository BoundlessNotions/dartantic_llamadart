import 'package:dartantic_interface/dartantic_interface.dart';

/// An embeddings model implementation using the Llamadart engine.
class LlamadartEmbeddingsModel extends EmbeddingsModel<EmbeddingsModelOptions> {
  LlamadartEmbeddingsModel({
    required super.name,
    required super.defaultOptions,
  });

  @override
  Future<EmbeddingsResult> embedQuery(
    String query, {
    EmbeddingsModelOptions? options,
  }) async {
    // TODO: Implement embeddings using llamadart native API.
    throw UnimplementedError('Embeddings are not yet implemented.');
  }

  @override
  Future<BatchEmbeddingsResult> embedDocuments(
    List<String> documents, {
    EmbeddingsModelOptions? options,
  }) async {
    // TODO: Implement embeddings using llamadart native API.
    throw UnimplementedError('Embeddings are not yet implemented.');
  }

  @override
  void dispose() {
    // Cleanup if needed
  }
}
