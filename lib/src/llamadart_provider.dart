import 'package:dartantic_interface/dartantic_interface.dart';
import 'llamadart_chat_options.dart';
import 'llamadart_chat_model.dart';
import 'llamadart_embeddings_model.dart';

/// A dartantic provider for local models run through llamadart.
class LlamadartProvider
    extends
        Provider<
          LlamadartChatOptions,
          EmbeddingsModelOptions,
          MediaGenerationModelOptions
        > {
  /// The chat model: a `.gguf` file or a `.litertlm` bundle.
  final String modelPath;

  /// An embedding GGUF for [createEmbeddingsModel], or null when this
  /// provider doesn't do embeddings.
  final String? embeddingsModelPath;

  LlamadartProvider({
    required super.name,
    required super.displayName,
    required this.modelPath,
    this.embeddingsModelPath,
    super.defaultModelNames = const {},
    super.headers = const {},
  });

  @override
  List<String> get aliases => [];

  @override
  String? get apiKey => null;

  @override
  String? get apiKeyName => null;

  @override
  Uri? get baseUrl => null;

  @override
  Stream<ModelInfo> listModels() async* {
    yield ModelInfo(
      name: defaultModelNames[ModelKind.chat] ?? 'default',
      providerName: name,
      kinds: {ModelKind.chat},
    );
    if (embeddingsModelPath != null) {
      yield ModelInfo(
        name: defaultModelNames[ModelKind.embeddings] ?? 'default',
        providerName: name,
        kinds: {ModelKind.embeddings},
      );
    }
  }

  @override
  ChatModel<LlamadartChatOptions> createChatModel({
    String? name,
    List<Tool<Object>>? tools,
    double? temperature,
    bool enableThinking = false,
    LlamadartChatOptions? options,
  }) {
    final modelName = name ?? defaultModelNames[ModelKind.chat] ?? 'default';
    return LlamadartChatModel(
      provider: this,
      name: modelName,
      tools: tools,
      temperature: temperature,
      defaultOptions: (options ?? const LlamadartChatOptions()).copyWith(
        temp: temperature,
      ),
      enableThinking: enableThinking,
    );
  }

  @override
  EmbeddingsModel<EmbeddingsModelOptions> createEmbeddingsModel({
    String? name,
    EmbeddingsModelOptions? options,
  }) {
    final path = embeddingsModelPath;
    if (path == null) {
      throw StateError(
        'LlamadartProvider "${this.name}" has no embeddingsModelPath',
      );
    }
    return LlamadartEmbeddingsModel(
      modelPath: path,
      name: name ?? defaultModelNames[ModelKind.embeddings] ?? 'default',
      defaultOptions: options ?? const EmbeddingsModelOptions(),
    );
  }

  @override
  MediaGenerationModel<MediaGenerationModelOptions> createMediaModel({
    String? name,
    List<Tool<Object>>? tools,
    MediaGenerationModelOptions? options,
  }) {
    throw UnimplementedError('Media generation is not supported.');
  }
}
