import * as vscode from "vscode";
import { ConfigManager } from "../config/configManager";

export function registerManageConfigCommand(context: vscode.ExtensionContext, configManager: ConfigManager) {
	return vscode.commands.registerCommand("litellm-connector.manage", async () => {
		const config = await configManager.getConfig();

		const baseUrl = await vscode.window.showInputBox({
			title: "LiteLLM Base URL",
			prompt: config.url
				? "Update your LiteLLM base URL"
				: "Enter your LiteLLM base URL (e.g., http://localhost:4000 or https://api.litellm.ai)",
			ignoreFocusOut: true,
			value: config.url,
			placeHolder: "http://localhost:4000",
		});

		if (baseUrl === undefined) {
			return;
		}

		const apiKey = await vscode.window.showInputBox({
			title: "LiteLLM API Key",
			prompt: config.key ? "Update your LiteLLM API key" : "Enter your LiteLLM API key (leave empty if not required)",
			ignoreFocusOut: true,
			password: true,
			value: config.key ?? "",
		});

		if (apiKey === undefined) {
			return;
		}

		await configManager.setConfig({
			url: baseUrl.trim(),
			key: apiKey.trim() || undefined,
		});

		vscode.window.showInformationMessage("LiteLLM configuration saved.");
	});
}
