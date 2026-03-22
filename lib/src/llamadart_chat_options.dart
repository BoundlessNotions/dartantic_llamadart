import 'package:dartantic_interface/dartantic_interface.dart';

/// Options for the Llamadart chat model.
class LlamadartChatOptions extends ChatModelOptions {
  /// The context size (number of tokens).
  final int? nCtx;

  /// The number of layers to offload to the GPU.
  final int? nGpuLayers;

  /// The temperature for sampling.
  final double? temp;

  /// The top-k value for sampling.
  final int? topK;

  /// The top-p value for sampling.
  final double? topP;

  /// The repeat penalty.
  final double? repeatPenalty;

  const LlamadartChatOptions({
    this.nCtx,
    this.nGpuLayers,
    this.temp,
    this.topK,
    this.topP,
    this.repeatPenalty,
  });

  LlamadartChatOptions copyWith({
    int? nCtx,
    int? nGpuLayers,
    double? temp,
    int? topK,
    double? topP,
    double? repeatPenalty,
  }) {
    return LlamadartChatOptions(
      nCtx: nCtx ?? this.nCtx,
      nGpuLayers: nGpuLayers ?? this.nGpuLayers,
      temp: temp ?? this.temp,
      topK: topK ?? this.topK,
      topP: topP ?? this.topP,
      repeatPenalty: repeatPenalty ?? this.repeatPenalty,
    );
  }
}
