import 'package:dartantic_interface/dartantic_interface.dart';
import 'llamadart_chat_options.dart';
import 'llamadart_chat_model.dart';
import 'llamadart_embeddings_model.dart';

/// A provider for the Llamadart engine.
class LlamadartProvider
    implements
        Provider<LlamadartChatOptions, EmbeddingsModelOptions,
            MediaGenerationModelOptions> {
  @override
  final String name;

  @override
  final String displayName;

  /// The local path to the GGUF model.
  final String modelPath;

  /// Default model names for different kinds of tasks.
  final Map<ModelKind, String> defaultModelNames;

  LlamadartProvider({
    required this.name,
    required this.displayName,
    required this.modelPath,
    this.defaultModelNames = const {},
  });

  @override
  ChatModel<LlamadartChatOptions> createChatModel({
    String? modelName,
    LlamadartChatOptions? defaultOptions,
  }) {
    return LlamadartChatModel(
      provider: this,
      modelName: modelName ?? defaultModelNames[ModelKind.chat] ?? 'default',
      defaultOptions: defaultOptions ?? const LlamadartChatOptions(),
    );
  }

  @override
  EmbeddingsModel<EmbeddingsModelOptions> createEmbeddingsModel({
    String? modelName,
    EmbeddingsModelOptions? defaultOptions,
  }) {
    return LlamadartEmbeddingsModel(
      provider: this,
      modelName: modelName ?? defaultModelNames[ModelKind.embeddings] ?? 'default',
      defaultOptions: defaultOptions ?? const EmbeddingsModelOptions(),
    );
  }

  @override
  MediaGenerationModel<MediaGenerationModelOptions> createMediaGenerationModel({
    String? modelName,
    MediaGenerationModelOptions? defaultOptions,
  }) {
    throw UnimplementedError('Media generation is not supported by Llamadart.');
  }
}
