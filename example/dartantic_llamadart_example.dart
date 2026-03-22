import 'package:dartantic_interface/dartantic_interface.dart';
import 'package:dartantic_llamadart/dartantic_llamadart.dart';

void main() async {
  // 1. Setup the Llamadart provider with a path to a GGUF model.
  final provider = LlamadartProvider(
    name: 'llamadart',
    displayName: 'Local Llama',
    modelPath: 'models/llama3.2-1b.gguf', // Replace with your actual model path
  );

  // 2. Create a chat model.
  final chatModel = provider.createChatModel(
    defaultOptions: const LlamadartChatOptions(
      temp: 0.7,
      nCtx: 2048,
    ),
  );

  // 3. Define messages.
  final messages = [
    ChatMessage(role: ChatRole.system, parts: [TextPart('You are a helpful assistant.')]),
    ChatMessage(role: ChatRole.user, parts: [TextPart('Hello! Who are you?')]),
  ];

  // 4. Generate a response.
  print('Generating response (offline)...');
  final result = await chatModel.generate(messages);

  print('Response:');
  for (final part in result.message.parts) {
    if (part is TextPart) {
      print(part.text);
    }
  }

  // 5. Cleanup
  chatModel.dispose();
}
