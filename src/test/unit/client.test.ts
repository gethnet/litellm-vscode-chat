import * as assert from "assert";
import { LiteLLMClient } from "../../adapters/litellmClient";

suite("LiteLLM Client Unit Tests", () => {
	const config = { url: "http://localhost:4000", key: "test-key" };
	const userAgent = "test-ua";
	const client = new LiteLLMClient(config, userAgent);

	test("transformToResponsesFormat normalizes tool call IDs", () => {
		const body = client.transformToResponsesFormat({
			model: "m",
			messages: [
				{
					role: "assistant",
					tool_calls: [{ id: "call1", type: "function", function: { name: "do", arguments: "{}" } }],
				},
				{ role: "tool", tool_call_id: "call1", content: "ok" },
			],
		});

		const input = body.input as Record<string, unknown>[];
		const functionCall = input.find((i) => i.type === "function_call");
		const functionOutput = input.find((i) => i.type === "function_call_output");

		assert.strictEqual(functionCall?.id, "fc_call1");
		assert.strictEqual(functionOutput?.call_id, "fc_call1");
	});

	test("transformToResponsesFormat skips tool outputs with no matching function call", () => {
		const body = client.transformToResponsesFormat({
			model: "m",
			messages: [
				{ role: "user", content: "hello" },
				{ role: "tool", tool_call_id: "fc_missing", content: "result" },
			],
		});

		const input = body.input as Record<string, unknown>[];
		const functionOutputs = input.filter((item) => item.type === "function_call_output");
		assert.strictEqual(functionOutputs.length, 0);
	});
});
