import 'package:dartantic_interface/dartantic_interface.dart';
import 'package:llamadart/llamadart.dart';

/// Options for the Llamadart chat model.
class LlamadartChatOptions extends ChatModelOptions {
  /// The context size (number of tokens).
  final int? nCtx;

  /// The number of layers to offload to the GPU.
  final int? nGpuLayers;

  /// The preferred GPU backend.
  final GpuBackend preferredBackend;

  /// The temperature for sampling.
  final double? temp;

  /// The top-k value for sampling.
  final int? topK;

  /// The top-p value for sampling.
  final double? topP;

  /// The repeat penalty.
  final double? repeatPenalty;

  /// Minimum probability for nucleus sampling.
  final double? minP;

  /// Maximum number of tokens to generate.
  final int? maxTokens;

  /// Stream batch token threshold for native backends.
  final int? streamBatchTokenThreshold;

  /// Stream batch byte threshold for native backends.
  final int? streamBatchByteThreshold;

  /// Reuse prompt prefix for multi-turn chat optimization.
  final bool? reusePromptPrefix;

  const LlamadartChatOptions({
    this.nCtx,
    this.nGpuLayers,
    this.preferredBackend = GpuBackend.auto,
    this.temp,
    this.topK,
    this.topP,
    this.repeatPenalty,
    this.minP,
    this.maxTokens,
    this.streamBatchTokenThreshold,
    this.streamBatchByteThreshold,
    this.reusePromptPrefix,
  });

  LlamadartChatOptions copyWith({
    int? nCtx,
    int? nGpuLayers,
    GpuBackend? preferredBackend,
    double? temp,
    int? topK,
    double? topP,
    double? repeatPenalty,
    double? minP,
    int? maxTokens,
    int? streamBatchTokenThreshold,
    int? streamBatchByteThreshold,
    bool? reusePromptPrefix,
  }) {
    return LlamadartChatOptions(
      nCtx: nCtx ?? this.nCtx,
      nGpuLayers: nGpuLayers ?? this.nGpuLayers,
      preferredBackend: preferredBackend ?? this.preferredBackend,
      temp: temp ?? this.temp,
      topK: topK ?? this.topK,
      topP: topP ?? this.topP,
      repeatPenalty: repeatPenalty ?? this.repeatPenalty,
      minP: minP ?? this.minP,
      maxTokens: maxTokens ?? this.maxTokens,
      streamBatchTokenThreshold:
          streamBatchTokenThreshold ?? this.streamBatchTokenThreshold,
      streamBatchByteThreshold:
          streamBatchByteThreshold ?? this.streamBatchByteThreshold,
      reusePromptPrefix: reusePromptPrefix ?? this.reusePromptPrefix,
    );
  }
}
