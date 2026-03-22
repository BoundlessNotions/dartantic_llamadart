# Handoff Plan: dartantic_llamadart Plugin Development & Publication

## 1. Project Overview
**Goal:** Create a standalone Dart/Flutter package (`dartantic_llamadart`) that implements the `dartantic_interface` to provide local GGUF model support via the `llamadart` engine. This allows `dartantic_ai` agents to run fully offline within AetherForge AI and other Flutter applications.

---

## 2. Technical Specification

### 2.1 Dependencies
The `pubspec.yaml` for the new repository must include:
- `dartantic_interface`: ^1.0.0 (The core abstraction layer)
- `llamadart`: ^0.6.7 (The native GGUF inference engine)
- `meta`: ^1.11.0 (For @protected and @internal annotations)

### 2.2 Core Implementations Required
1.  **`LlamadartChatOptions`**: Extend `ChatModelOptions` to include `n_ctx`, `n_gpu_layers`, `temp`, `top_k`, `top_p`, and `repeat_penalty`.
2.  **`LlamadartProvider`**: 
    - Implement `Provider<LlamadartChatOptions, EmbeddingsModelOptions, MediaGenerationModelOptions>`.
    - Support a constructor that accepts a local `modelPath`.
    - Implement `createChatModel` and `createEmbeddingsModel`.
3.  **`LlamadartChatModel`**:
    - Wrap `llamadart.ChatSession`.
    - **Streaming:** Map `llamadart` tokens to `Stream<ChatResult<ChatMessage>>`.
    - **Tool Calling:** Implement a GBNF grammar or XML-tag wrapper (e.g., `<tool_call>`) to ensure the local model outputs structured tool calls compatible with `dartantic.ToolPart`.
4.  **`LlamadartEmbeddingsModel`**:
    - Implement the `EmbeddingsModel` interface to allow local vector RAG.

---

## 3. Publication & Distribution Plan

### 3.1 Repository Setup
- **Name:** `dartantic_llamadart`
- **Structure:** Standard Dart package structure with an `example/` folder showing basic agent integration.
- **CI/CD:** GitHub Actions for `dart analyze`, `dart test`, and `pana` (pub.dev score check).

### 3.2 Pub.dev Publication
1.  **Verification:** Run `dart pub publish --dry-run` to ensure no warnings.
2.  **Documentation:** 
    - Detailed `README.md` explaining how to download/bundle GGUF models.
    - `CHANGELOG.md` following Semantic Versioning (SemVer).
    - `LICENSE` (MIT or BSD-3 recommended).
3.  **Release:** `dart pub publish`.

---

## 4. Integration & Registration (AetherForge AI)

Once the plugin is published or available via Git, integrate it into the main app:

### 4.1 Dependency Addition
```yaml
dependencies:
  dartantic_ai: ^1.0.0
  dartantic_llamadart: ^1.0.0 # From pub.dev or git
```

### 4.2 Bootstrap Registration
In the main application entry point (or a dedicated service provider), register the provider into the `Agent` factory:

```dart
import 'package:dartantic_ai/dartantic_ai.dart';
import 'package:dartantic_llamadart/dartantic_llamadart.dart';

void initializeAiEnvironment() {
  // Register llamadart as a first-class provider in the Agent ecosystem
  Agent.providerFactories['llamadart'] = () => LlamadartProvider(
    name: 'llamadart',
    displayName: 'Local Llama (GGUF)',
    // Configuration for local paths and context
    defaultModelNames: {
      ModelKind.chat: 'llama3.2-1b',
      ModelKind.embeddings: 'nomic-embed-text',
    },
  );
}
```

---

## 5. Phase-by-Phase Execution

### Phase 1: Local Implementation (Days 1-3) ✅ COMPLETED
- [x] Scaffolding the repo and implementing the `Provider` and `ChatModel` classes.
- [x] Internal testing with the `llama3.2-1b-it-int4.gguf` model already in AetherForge.

### Phase 2: Tool Call Logic (Days 4-5) ✅ COMPLETED
- [x] Developing the GBNF grammar or prompt-wrapper to ensure 100% JSON reliability for tool calls from small models.
- [x] Implementing the `ToolPart` parser.

### Phase 3: Publication (Day 6) ❌ NOT STARTED
- [ ] Writing documentation and publishing to `pub.dev`.

### Phase 4: AetherForge Migration (Day 7) ❌ NOT STARTED
- [ ] Replacing the custom `DeckArchitect` loops with `dartantic.Agent` calls using the new `llamadart` provider.

---

## 6. Success Criteria
- [ ] Plugin passes `pana` with 130/140+ score.
- [ ] Agent can successfully execute a multi-turn tool call (e.g., Search -> Add -> Search -> Add) using ONLY the local `llamadart` provider.
- [ ] Memory usage remains stable across multiple agent sessions.
