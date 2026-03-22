import 'package:dartantic_interface/dartantic_interface.dart';
import 'llamadart_chat_options.dart';
import 'llamadart_chat_model.dart';
import 'llamadart_embeddings_model.dart';

/// A provider for the Llamadart engine.
class LlamadartProvider
    implements Provider<LlamadartChatOptions, EmbeddingsModelOptions> {
  @override
  final String name;

  @override
  final String displayName;

  /// The local path to the GGUF model.
  final String modelPath;

  @override
  final Map<ModelKind, String> defaultModelNames;

  LlamadartProvider({
    required this.name,
    required this.displayName,
    required this.modelPath,
    this.defaultModelNames = const {},
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
  Set<ProviderCaps> get caps => {ProviderCaps.chat, ProviderCaps.thinking};

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
    LlamadartChatOptions? options,
    double? temperature,
    List<Tool>? tools,
    bool enableThinking = false,
  }) {
    final modelName = name ?? defaultModelNames[ModelKind.chat] ?? 'default';
    return LlamadartChatModel(
      provider: this,
      name: modelName,
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
}
