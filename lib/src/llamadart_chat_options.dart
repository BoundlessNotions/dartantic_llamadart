import 'package:dartantic_interface/dartantic_interface.dart';
import 'package:llamadart/llamadart.dart';

/// Options for the Llamadart chat model.
class LlamadartChatOptions extends ChatModelOptions {
  /// The context size (number of tokens).
  final int? nCtx;

  /// The number of layers to offload to the GPU.
  final int? nGpuLayers;

  /// The preferred GPU backend (for GGUF/llama.cpp models).
  final GpuBackend preferredBackend;

  /// The preferred LiteRT-LM runtime backend (for .litertlm models).
  ///
  /// Defaults to [LiteRtLmBackendPreference.auto], which selects GPU on
  /// Android/macOS and CPU elsewhere.
  final LiteRtLmBackendPreference liteRtLmBackend;

  /// Override the chat template detected from the model file.
  ///
  /// Useful when the model does not embed a recognisable chat template or when
  /// you want to force a specific format (e.g. `'gemma'`).
  final String? chatTemplate;

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

  /// Enables backend-native speculative decoding (multi-token prediction).
  ///
  /// On the LiteRT-LM backend this requires a `.litertlm` bundle that ships MTP
  /// draft heads (e.g. the post-MTP Gemma 4 E2B revision). On llama.cpp/GGUF it
  /// enables the backend-default speculative path for a model that carries its
  /// own MTP layers; to drive GGUF speculation with a *separate* drafter model,
  /// set [mtpDraftModelPath] instead. WebGPU and LiteRT-LM web reject it.
  /// Defaults to disabled.
  final bool? speculativeDecoding;

  /// Path to a separate MTP draft model for llama.cpp/GGUF speculative decoding
  /// (the `draft-mtp` path, equivalent to llama.cpp `--model-draft`).
  ///
  /// When set on a GGUF model, multi-token-prediction speculative decoding is
  /// enabled using this drafter. Ignored on the LiteRT-LM backend (which carries
  /// its MTP heads inside the `.litertlm` bundle and is driven by
  /// [speculativeDecoding]). Leave null for a self-MTP GGUF.
  final String? mtpDraftModelPath;

  /// Maximum draft tokens proposed per MTP step on the GGUF path (null → 1).
  ///
  /// Drives both the generation-time `draftTokenMax` and the model-load-time
  /// `ModelParams.speculativeRollbackTokenMax` (which llama.cpp requires to be
  /// >= the draft token max for architectures needing bounded rollback). Only
  /// relevant when [mtpDraftModelPath] is set.
  final int? mtpDraftTokenMax;

  const LlamadartChatOptions({
    this.nCtx,
    this.nGpuLayers,
    this.preferredBackend = GpuBackend.auto,
    this.liteRtLmBackend = LiteRtLmBackendPreference.auto,
    this.chatTemplate,
    this.temp,
    this.topK,
    this.topP,
    this.repeatPenalty,
    this.minP,
    this.maxTokens,
    this.streamBatchTokenThreshold,
    this.streamBatchByteThreshold,
    this.reusePromptPrefix,
    this.speculativeDecoding,
    this.mtpDraftModelPath,
    this.mtpDraftTokenMax,
  });

  LlamadartChatOptions copyWith({
    int? nCtx,
    int? nGpuLayers,
    GpuBackend? preferredBackend,
    LiteRtLmBackendPreference? liteRtLmBackend,
    String? chatTemplate,
    bool clearChatTemplate = false,
    double? temp,
    int? topK,
    double? topP,
    double? repeatPenalty,
    double? minP,
    int? maxTokens,
    int? streamBatchTokenThreshold,
    int? streamBatchByteThreshold,
    bool? reusePromptPrefix,
    bool? speculativeDecoding,
    String? mtpDraftModelPath,
    int? mtpDraftTokenMax,
  }) {
    return LlamadartChatOptions(
      nCtx: nCtx ?? this.nCtx,
      nGpuLayers: nGpuLayers ?? this.nGpuLayers,
      preferredBackend: preferredBackend ?? this.preferredBackend,
      liteRtLmBackend: liteRtLmBackend ?? this.liteRtLmBackend,
      chatTemplate: clearChatTemplate ? null : (chatTemplate ?? this.chatTemplate),
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
      speculativeDecoding: speculativeDecoding ?? this.speculativeDecoding,
      mtpDraftModelPath: mtpDraftModelPath ?? this.mtpDraftModelPath,
      mtpDraftTokenMax: mtpDraftTokenMax ?? this.mtpDraftTokenMax,
    );
  }
}
