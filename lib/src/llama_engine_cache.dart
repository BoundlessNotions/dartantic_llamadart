import 'dart:async';

import 'package:llamadart/llamadart.dart';
import 'package:meta/meta.dart';

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
/// same engine. Generations on one engine must not overlap, so callers take
/// [LlamaEngineHandle.lock] around each generation (the chat model does this)
/// and queue behind whoever holds it.
class LlamaEngineCache {
  LlamaEngineCache._();

  static final LlamaEngineCache instance = LlamaEngineCache._();

  static LlamaEngine defaultEngineFactory() => LlamaEngine(LlamaBackend());

  /// Builds the engine [acquire] loads. Tests swap in a fake engine so
  /// generation can be scripted without a native model.
  @visibleForTesting
  LlamaEngine Function() engineFactory = defaultEngineFactory;

  final Map<String, _EngineEntry> _entries = {};

  static String keyFor(String modelPath, ModelParams params) => [
    modelPath,
    params.contextSize,
    params.gpuLayers,
    params.preferredBackend,
    params.liteRtLmBackend,
    params.chatTemplate,
    params.speculativeRollbackTokenMax,
    params.batchSize,
    params.maxParallelSequences,
  ].join('|');

  /// Returns a handle on the cached engine for ([modelPath], [params]),
  /// loading the model on first use. A failed load is not cached; the next
  /// call retries.
  ///
  /// Callers should acquire per use rather than hold the engine: after
  /// [evict], the next acquire loads a fresh engine, while a held one is
  /// disposed.
  Future<LlamaEngineHandle> acquire(
    String modelPath,
    ModelParams params,
  ) async {
    final key = keyFor(modelPath, params);
    final entry = _entries[key] ??= _EngineEntry(_load(modelPath, params));
    try {
      return LlamaEngineHandle._(key, entry, await entry.engine);
    } catch (_) {
      if (identical(_entries[key], entry)) _entries.remove(key);
      rethrow;
    }
  }

  Future<LlamaEngine> _load(String modelPath, ModelParams params) async {
    final engine = engineFactory();
    try {
      await engine.loadModel(modelPath, modelParams: params);
      return engine;
    } catch (_) {
      try {
        await engine.dispose();
      } catch (_) {
        // Best effort — the engine never finished loading.
      }
      rethrow;
    }
  }

  /// Acquires the engine for ([modelPath], [params]) and waits for exclusive
  /// use of it. Returns the handle and the function that releases it.
  Future<(LlamaEngineHandle, void Function())> acquireExclusive(
    String modelPath,
    ModelParams params,
  ) async {
    while (true) {
      final handle = await acquire(modelPath, params);
      final release = await handle.lock();
      // The holder ahead of us may have evicted it after a failed generation.
      if (!handle.isEvicted) return (handle, release);
      release();
    }
  }

  /// Removes and disposes the engine behind [handle]. Used when a failed
  /// native generation may have corrupted the engine; the next [acquire]
  /// reloads. A no-op if that engine was already evicted.
  Future<void> evict(LlamaEngineHandle handle) async {
    if (!identical(_entries[handle.key], handle._entry)) return;
    _entries.remove(handle.key);
    await _dispose(handle._entry);
  }

  /// Disposes every cached engine. For app shutdown and tests.
  Future<void> disposeAll() async {
    final entries = _entries.values.toList();
    _entries.clear();
    for (final entry in entries) {
      await _dispose(entry);
    }
  }

  Future<void> _dispose(_EngineEntry entry) async {
    entry.evicted = true;
    try {
      final engine = await entry.engine;
      engine.cancelGeneration();
      await engine.dispose();
    } catch (_) {
      // Best effort — the engine may already be unusable.
    }
  }
}

/// A loaded engine plus the per-engine state [LlamaEngineCache] keeps for it.
class LlamaEngineHandle {
  LlamaEngineHandle._(this.key, this._entry, this.engine);

  /// The cache key, from [LlamaEngineCache.keyFor].
  final String key;
  final LlamaEngine engine;
  final _EngineEntry _entry;

  /// Whether [engine] was evicted (and disposed) after this handle was
  /// acquired. Acquire again for a live engine.
  bool get isEvicted => _entry.evicted;

  /// The chat format detected from the model's template, computed once per
  /// loaded engine.
  Future<ChatFormat> chatFormat() => _entry.format ??= _detectFormat();

  Future<ChatFormat> _detectFormat() async {
    try {
      final metadata = await engine.getMetadata();
      return ChatTemplateEngine.detectFormat(
        metadata['tokenizer.chat_template'],
      );
    } catch (_) {
      // Retry on the next call rather than caching the failure.
      _entry.format = null;
      rethrow;
    }
  }

  /// Waits, in FIFO order, for exclusive use of [engine] and returns the
  /// function that releases it. Call the release exactly once.
  Future<void Function()> lock() => _entry.lock();
}

class _EngineEntry {
  _EngineEntry(this.engine);

  final Future<LlamaEngine> engine;
  Future<ChatFormat>? format;
  bool evicted = false;
  Future<void> _lockTail = Future.value();

  Future<void Function()> lock() async {
    final previous = _lockTail;
    final released = Completer<void>();
    _lockTail = released.future;
    await previous;
    return released.complete;
  }
}
