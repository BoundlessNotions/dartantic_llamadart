# dartantic_llamadart

Local GGUF model support for [Dartantic AI](https://pub.dev/packages/dartantic_ai) using the [llamadart](https://pub.dev/packages/llamadart) engine.

This package allows you to run fully offline AI agents in Flutter and Dart applications by leveraging local GGUF models.

## Features

- **Local Inference:** Run models entirely on-device without an internet connection.
- **Dartantic Interface:** Seamlessly integrates with the Dartantic AI ecosystem.
- **GPU Acceleration:** Inherits hardware acceleration support from `llamadart`.
- **Tool Calling:** Supports structured tool calls using XML-tag wrapper format (`<tool_call>`).

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

## Tool Calling

To use tool calling with local GGUF models, include instructions in your system prompt to use the `<tool_call>` XML format:

```dart
final messages = [
  ChatMessage(
    role: ChatMessageRole.system,
    parts: [TextPart('''You have access to tools. When you need to call a tool, output:
<tool_call>{"tool_name": {"param1": "value1"}}</tool_call>

Example: <tool_call>{"search": {"query": "weather"}}</tool_call>
''')],
  ),
  ChatMessage(role: ChatMessageRole.user, parts: [TextPart('Search for AI news')]),
];
```

The model will output tool calls wrapped in `<tool_call>` tags, which are automatically parsed into `ToolPart` objects.

## License

BSD-3-Clause - See [LICENSE](LICENSE) for details.
