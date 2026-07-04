import 'package:llamadart/llamadart.dart';

/// Process-wide cache of loaded [LlamaEngine]s, keyed by model path plus the
/// load-time parameters that shape the native model/context.
///
/// Loading a GGUF/LiteRT model pays for weight mapping, graph compilation, and
/// context allocation — seconds of CPU work and hundreds of MB of native
/// memory. Callers that construct a fresh [LlamadartChatModel] per request
/// (e.g. fresh-agent-per-phase orchestrators) would otherwise reload the model
/// on every call and leak the previous engine's native handles.
///
/// Engines are shared, not pooled: two chat models with the same key get the
/// same engine. Generations must therefore never overlap — callers interrupt
/// any in-flight generation via [LlamaEngine.cancelGeneration] before starting
/// a new one (the chat model does this automatically).
class LlamaEngineCache {
  LlamaEngineCache._();

  static final LlamaEngineCache instance = LlamaEngineCache._();

  final Map<String, Future<LlamaEngine>> _engines = {};

  static String keyFor(String modelPath, ModelParams params) => [
    modelPath,
    params.contextSize,
    params.gpuLayers,
    params.preferredBackend,
    params.liteRtLmBackend,
    params.chatTemplate,
    params.speculativeRollbackTokenMax,
  ].join('|');

  /// Returns the cached engine for ([modelPath], [params]), loading the model
  /// on first use. A failed load is not cached — the next call retries.
  Future<LlamaEngine> acquire(String modelPath, ModelParams params) {
    final key = keyFor(modelPath, params);
    final existing = _engines[key];
    if (existing != null) return existing;

    final future = () async {
      final engine = LlamaEngine(LlamaBackend());
      try {
        await engine.loadModel(modelPath, modelParams: params);
        return engine;
      } catch (_) {
        _engines.remove(key);
        try {
          await engine.dispose();
        } catch (_) {
          // Best effort — the engine never finished loading.
        }
        rethrow;
      }
    }();
    _engines[key] = future;
    return future;
  }

  /// Removes and disposes the engine under [key]. Used when a failed native
  /// generation may have corrupted the engine — the next [acquire] reloads.
  Future<void> evict(String key) async {
    final future = _engines.remove(key);
    if (future == null) return;
    try {
      final engine = await future;
      engine.cancelGeneration();
      await engine.dispose();
    } catch (_) {
      // Best effort — the engine may already be unusable.
    }
  }

  /// Disposes every cached engine. For app shutdown and tests.
  Future<void> disposeAll() async {
    final keys = _engines.keys.toList();
    for (final key in keys) {
      await evict(key);
    }
  }
}
