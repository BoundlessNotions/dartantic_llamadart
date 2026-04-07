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

  const LlamadartChatOptions({
    this.nCtx,
    this.nGpuLayers,
    this.preferredBackend = GpuBackend.auto,
    this.temp,
    this.topK,
    this.topP,
    this.repeatPenalty,
    this.minP,
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
    );
  }
}
