# LiteLLM Connector for GitHub Copilot Chat

Integrate LiteLLM proxies into GitHub Copilot Chat through the `litellm-connector` vendor that surfaces hundreds of LiteLLM-backed models directly inside VS Code.

## ⚡ Quick Start
1. Install [GitHub Copilot Chat](https://marketplace.visualstudio.com/items?itemName=GitHub.copilot-chat) if you haven't already.
2. Install LiteLLM Connector for GitHub Copilot Chat ([marketplace listing](https://marketplace.visualstudio.com/items?itemName=Gethnet.litellm-connector-copilot)).
3. Open the Copilot Chat interface and select "Manage Models..." from the model picker.
4. Choose the "LiteLLM" provider (vendor `litellm-connector`) and run the `Manage LiteLLM Provider` command (`litellm-connector.manage`) from the Command Palette.
5. Enter your LiteLLM base URL (e.g., `http://localhost:4000` or your hosted proxy) and optionally your API key. Values are stored securely in VS Code secret storage.
6. Pick the LiteLLM models you want to enable. Each entry retains the `cheapest` and `fastest` selection options to favor cost or throughput.

Each model entry also offers `cheapest` and `fastest` mode for each model. `fastest` selects the provider with highest throughput and `cheapest` selects the provider with lowest price per output token.

## ✨ Features
* Exposes hundreds of LiteLLM-backed models inside Copilot Chat through the `litellm-connector` vendor registration and `litellm-connector.manage` configuration command.
* Handles streaming responses, built-in tool calling, and image-capable vision models when the LiteLLM endpoint advertises those capabilities.
* Detects LiteLLM-openAI parameter limitations, strips incompatible arguments to avoid errors, and shows friendly warnings to reconfigure when authorization or connectivity issues occur.
* Works with self-hosted LiteLLM proxies or hosted instances, so you can keep data inside your network or use managed APIs.

## Requirements
* VS Code 1.107.0 or higher (match the `engines.vscode` constraint).
* [GitHub Copilot Chat](https://marketplace.visualstudio.com/items?itemName=GitHub.copilot-chat) (`github.copilot-chat`) must be installed since we register a chat provider within that experience.
* A LiteLLM endpoint (self-hosted or cloud) plus its base URL and optional API key.

## 🛠️ Development
```bash
git clone https://github.com/gethnet/litellm-vscode-chat
cd litellm-vscode-chat
pnpm install       # downloads @vscode/dts (download-api) via postinstall
pnpm run compile
```
Press F5 to launch an Extension Development Host.

Common scripts:
* Build: `pnpm run compile`
* Download API metadata: `pnpm run download-api` (also triggered automatically after install and updates)
* Watch: `pnpm run watch`
* Lint and fix: `pnpm run lint`
* Format: `pnpm run format`
* Test: `pnpm run test` (runs compile + `vscode-test`)
* Bump version: `pnpm run bump-version` (updates package metadata via `scripts/bump-version.js`)
* Package/publish: use `vsce` (`pnpm exec vsce package` / `pnpm exec vsce publish`).

## 📚 Learn more
* LiteLLM documentation: https://docs.litellm.ai
* VS Code Chat Provider API: https://code.visualstudio.com/api/extension-guides/ai/language-model-chat-provider

## Support

* Open issues: https://github.com/gethnet/litellm-vscode-chat/issues
