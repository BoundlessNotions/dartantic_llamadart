import 'package:dartantic_interface/dartantic_interface.dart';
import 'llamadart_chat_options.dart';
import 'llamadart_chat_model.dart';
import 'llamadart_embeddings_model.dart';

class LlamadartProvider
    extends
        Provider<
          LlamadartChatOptions,
          EmbeddingsModelOptions,
          MediaGenerationModelOptions
        > {
  final String modelPath;

  LlamadartProvider({
    required super.name,
    required super.displayName,
    required this.modelPath,
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
      defaultOptions: (options ?? const LlamadartChatOptions()).copyWith(
        temp: temperature,
      ),
    );
  }

  @override
  EmbeddingsModel<EmbeddingsModelOptions> createEmbeddingsModel({
    String? name,
    EmbeddingsModelOptions? options,
  }) {
    final modelName =
        name ?? defaultModelNames[ModelKind.embeddings] ?? 'default';
    return LlamadartEmbeddingsModel(
      name: modelName,
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
