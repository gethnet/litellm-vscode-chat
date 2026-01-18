import * as vscode from "vscode";
import { LiteLLMChatModelProvider } from "./provider";

export function activate(context: vscode.ExtensionContext) {
	// Build a descriptive User-Agent to help quantify API usage
	const ext = vscode.extensions.getExtension("gethnet.litellm-vscode-chat");
	const extVersion = ext?.packageJSON?.version ?? "unknown";
	const vscodeVersion = vscode.version;
	// Keep UA minimal: only extension version and VS Code version
	const ua = `litellm-vscode-chat/${extVersion} VSCode/${vscodeVersion}`;

	const provider = new LiteLLMChatModelProvider(context.secrets, ua);
	// Register the LiteLLM provider under the vendor id used in package.json
	vscode.lm.registerLanguageModelChatProvider("litellm-connector", provider);

	// Management command to configure base URL and API key
	context.subscriptions.push(
		vscode.commands.registerCommand("litellm-connector.manage", async () => {
			// First, prompt for base URL
			const existingBaseUrl = await context.secrets.get("litellm-connector.baseUrl");
			const baseUrl = await vscode.window.showInputBox({
				title: "LiteLLM Base URL",
				prompt: existingBaseUrl
					? "Update your LiteLLM base URL"
					: "Enter your LiteLLM base URL (e.g., http://localhost:4000 or https://api.litellm.ai)",
				ignoreFocusOut: true,
				value: existingBaseUrl ?? "",
				placeHolder: "http://localhost:4000",
			});
			if (baseUrl === undefined) {
				return; // user canceled
			}

			// Then, prompt for API key
			const existingApiKey = await context.secrets.get("litellm-connector.apiKey");
			const apiKey = await vscode.window.showInputBox({
				title: "LiteLLM API Key",
				prompt: existingApiKey
					? "Update your LiteLLM API key"
					: "Enter your LiteLLM API key (leave empty if not required)",
				ignoreFocusOut: true,
				password: false,
				value: existingApiKey ?? "",
			});
			if (apiKey === undefined) {
				return; // user canceled
			}

			// Save or clear the values
			if (!baseUrl.trim()) {
				await context.secrets.delete("litellm-connector.baseUrl");
			} else {
				await context.secrets.store("litellm-connector.baseUrl", baseUrl.trim());
			}

			if (!apiKey.trim()) {
				await context.secrets.delete("litellm-connector.apiKey");
			} else {
				await context.secrets.store("litellm-connector.apiKey", apiKey.trim());
			}

			vscode.window.showInformationMessage("LiteLLM configuration saved.");
		})
	);
}

export function deactivate() {}
