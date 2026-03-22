# dartantic_llamadart

Local GGUF model support for [Dartantic AI](https://pub.dev/packages/dartantic_ai) using the [llamadart](https://pub.dev/packages/llamadart) engine.

This package allows you to run fully offline AI agents in Flutter and Dart applications by leveraging local GGUF models.

## Features

- **Local Inference:** Run models entirely on-device without an internet connection.
- **Dartantic Interface:** Seamlessly integrates with the Dartantic AI ecosystem.
- **GPU Acceleration:** Inherits hardware acceleration support from `llamadart`.
- **Tool Calling:** Supports structured tool calls using a prompt-wrapped XML approach (GBNF support coming soon).

## Getting Started

### 1. Add Dependencies

Add `dartantic_llamadart` to your `pubspec.yaml`:

```yaml
dependencies:
  dartantic_ai: ^1.0.0
  dartantic_llamadart: ^0.1.0
```

### 2. Download a GGUF Model

Download a GGUF model (e.g., Llama 3.2 1B) from Hugging Face and place it in your project or a local directory.

### 3. Initialize the Provider

```dart
import 'package:dartantic_ai/dartantic_ai.dart';
import 'package:dartantic_llamadart/dartantic_llamadart.dart';

void main() async {
  final provider = LlamadartProvider(
    name: 'llamadart',
    displayName: 'Local Llama',
    modelPath: 'path/to/your/model.gguf',
  );

  final agent = Agent(
    provider: provider,
    instructions: 'You are a helpful assistant running locally.',
  );

  final response = await agent.prompt('Hello! Who are you?');
  print(response);
}
```

## Advanced Configuration

You can customize inference parameters via `LlamadartChatOptions`:

```dart
final model = provider.createChatModel(
  defaultOptions: LlamadartChatOptions(
    temp: 0.7,
    nCtx: 4096,
    nGpuLayers: 35, // Offload layers to GPU
  ),
);
```

## License

BSD-3-Clause - See [LICENSE](LICENSE) for details.
