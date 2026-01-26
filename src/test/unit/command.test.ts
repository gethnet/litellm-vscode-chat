import * as assert from "assert";
import * as vscode from "vscode";
import * as sinon from "sinon";
import { registerManageConfigCommand } from "../../commands/manageConfig";
import { ConfigManager } from "../../config/configManager";

suite("ManageConfig Command Unit Tests", () => {
	let sandbox: sinon.SinonSandbox;
	let mockConfigManager: sinon.SinonStubbedInstance<ConfigManager>;
	let mockContext: vscode.ExtensionContext;

	setup(() => {
		sandbox = sinon.createSandbox();
		mockConfigManager = sandbox.createStubInstance(ConfigManager);
		mockContext = { subscriptions: [] } as unknown as vscode.ExtensionContext;
	});

	teardown(() => {
		sandbox.restore();
	});

	test("registers command correctly", () => {
		const registerStub = sandbox.stub(vscode.commands, "registerCommand");
		registerManageConfigCommand(mockContext, mockConfigManager as unknown as ConfigManager);
		assert.strictEqual(registerStub.calledWith("litellm-connector.manage"), true);
	});

	test("updates config when input is provided", async () => {
		mockConfigManager.getConfig.resolves({ url: "old-url", key: "old-key" });
		const showInputBoxStub = sandbox.stub(vscode.window, "showInputBox");
		showInputBoxStub.onFirstCall().resolves("new-url");
		showInputBoxStub.onSecondCall().resolves("new-key");
		const showInfoStub = sandbox.stub(vscode.window, "showInformationMessage");

		// Get the registered command handler
		let commandHandler: (() => Promise<void>) | undefined;
		sandbox.stub(vscode.commands, "registerCommand").callsFake((id, handler) => {
			if (id === "litellm-connector.manage") {
				commandHandler = handler as () => Promise<void>;
			}
			return { dispose: () => {} } as vscode.Disposable;
		});

		registerManageConfigCommand(mockContext, mockConfigManager as unknown as ConfigManager);

		if (commandHandler) {
			await commandHandler();
		}

		assert.strictEqual(
			mockConfigManager.setConfig.calledWith({
				url: "new-url",
				key: "new-key",
			}),
			true
		);
		assert.strictEqual(showInfoStub.calledOnce, true);
	});

	test("aborts if URL input is cancelled", async () => {
		mockConfigManager.getConfig.resolves({ url: "", key: "" });
		const showInputBoxStub = sandbox.stub(vscode.window, "showInputBox");
		showInputBoxStub.onFirstCall().resolves(undefined);

		let commandHandler: (() => Promise<void>) | undefined;
		sandbox.stub(vscode.commands, "registerCommand").callsFake((id, handler) => {
			if (id === "litellm-connector.manage") {
				commandHandler = handler as () => Promise<void>;
			}
			return { dispose: () => {} } as vscode.Disposable;
		});

		registerManageConfigCommand(mockContext, mockConfigManager as unknown as ConfigManager);

		if (commandHandler) {
			await commandHandler();
		}

		assert.strictEqual(mockConfigManager.setConfig.called, false);
	});
});
