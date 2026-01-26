# LiteLLM Connector for GitHub Copilot Chat

<!-- ALL-CONTRIBUTORS-BADGE:START - Do not remove or modify this section -->
[![All Contributors](https://img.shields.io/badge/all_contributors-3-orange.svg?style=flat-square)](#contributors-)
<!-- ALL-CONTRIBUTORS-BADGE:END -->

Integrate LiteLLM proxies into GitHub Copilot Chat through the \`litellm-connector\` vendor. This extension surfaces hundreds of LiteLLM-backed models directly inside VS Code, allowing you to use any model supported by LiteLLM within the Copilot Chat interface.

## ⚠️ Important - Prerequisites ⚠️

To use this extension, **YOU MUST** have an active GitHub Copilot plan (the Free plan works). This extension utilizes the VS Code Language Model Chat Provider API, which currently requires a Copilot subscription. For more details, see the [VS Code documentation](https://code.visualstudio.com/api/extension-guides/ai/language-model-chat-provider).

## ✨ Features

* **Hundreds of Models**: Access any model configured in your LiteLLM proxy (OpenAI, Anthropic, Google, Mistral, etc.) directly from the Copilot model picker.
* **Streaming Support**: Real-time response streaming for a smooth chat experience.
* **Tool Calling**: Built-in support for tool/function calling when supported by the underlying LiteLLM model.
* **Vision Capabilities**: Support for image-capable models when the LiteLLM endpoint advertises vision capabilities.
* **Smart Parameter Handling**: Automatically detects and strips incompatible parameters (like \`temperature\` for O1 models) to ensure reliable operation across different providers.
* **Secure Configuration**: API keys and base URLs are stored securely using VS Code's \`SecretStorage\`.
* **Flexible Deployment**: Works with self-hosted LiteLLM proxies or hosted instances.

## ⚡ Quick Start

1. **Install Prerequisites**: Ensure [GitHub Copilot Chat](https://marketplace.visualstudio.com/items?itemName=GitHub.copilot-chat) is installed.
2. **Install Extension**: Install "LiteLLM Connector for Copilot" from the VS Code Marketplace.
3. **Configure Provider**:
   * Open the Command Palette (\`Ctrl+Shift+P\` or \`Cmd+Shift+P\`).
   * Run the command: \`Manage LiteLLM Provider\`.
   * Enter your LiteLLM **Base URL** (e.g., \`http://localhost:4000\` or your hosted proxy URL).
   * Enter your **API Key** (optional, depending on your LiteLLM setup).
4. **Select Model**:
   * Open the Copilot Chat view.
   * Click the model picker (usually at the bottom of the chat input).
   * Select a model under the **LiteLLM** section.
5. **Start Chatting!**

## 🛠️ Development

If you want to contribute or build from source:

### Prerequisites
* [Node.js](https://nodejs.org/) (v18 or higher)
* [npm](https://www.npmjs.com/)

### Setup
1. Clone the repository.
2. Run \`npm install\` to install dependencies and download the latest VS Code Chat API definitions.
3. Press \`F5\` to launch the "Extension Development Host" window.

### Common Scripts
* \`npm run compile\`: Build the TypeScript source.
* \`npm run watch\`: Build and watch for changes.
* \`npm run lint\`: Run ESLint.
* \`npm run test\`: Run unit tests.
* \`npm run test:coverage\`: Run tests and generate coverage reports.
* \`npm run bump-version\`: Update version in \`package.json\`.

## 📚 Learn More

* [LiteLLM Documentation](https://docs.litellm.ai)
* [VS Code Language Model API](https://code.visualstudio.com/api/extension-guides/ai/language-model-chat-provider)

## Support & Contributions

* **Issues**: Report bugs or request features on [GitHub Issues](https://github.com/gethnet/litellm-vscode-chat/issues).
* **License**: Apache-2.0

---
*Maintained by [amwdrizz](https://github.com/amwdrizz)*
