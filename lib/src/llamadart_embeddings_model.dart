import 'package:dartantic_interface/dartantic_interface.dart';
import 'llamadart_provider.dart';

/// An embeddings model implementation using the Llamadart engine.
class LlamadartEmbeddingsModel extends EmbeddingsModel<EmbeddingsModelOptions> {
  @override
  final LlamadartProvider provider;

  @override
  final String modelName;

  @override
  final EmbeddingsModelOptions defaultOptions;

  LlamadartEmbeddingsModel({
    required this.provider,
    required this.modelName,
    required this.defaultOptions,
  });

  @override
  Future<List<double>> embed(
    String input, {
    EmbeddingsModelOptions? options,
  }) async {
    // TODO: Implement embeddings using llamadart native API.
    // LlamaModel/Context needs to be configured with embedding: true
    throw UnimplementedError('Embeddings are not yet implemented.');
  }

  @override
  void dispose() {
    // Cleanup if needed
  }
}
