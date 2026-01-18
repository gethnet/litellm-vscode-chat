import * as vscode from "vscode";
import {
	CancellationToken,
	LanguageModelChatInformation,
	LanguageModelChatRequestMessage,
	LanguageModelChatProvider,
	LanguageModelResponsePart,
	Progress,
	ProvideLanguageModelChatResponseOptions,
} from "vscode";

import type { LiteLLMModelInfoResponse, LiteLLMModelInfo, TransformedModelItem } from "./types";

import { convertTools, convertMessages, tryParseJSONObject, validateRequest } from "./utils";

const DEFAULT_MAX_OUTPUT_TOKENS = 16000;
const DEFAULT_CONTEXT_LENGTH = 128000;

/**
 * Known models that don't support certain OpenAI parameters,
 * even if LiteLLM claims they do.
 */
const KNOWN_PARAMETER_LIMITATIONS: Record<string, Set<string>> = {
	// Anthropic models don't support temperature in the same way
	"claude-3-5-sonnet": new Set(["temperature"]),
	"claude-3-5-haiku": new Set(["temperature"]),
	"claude-3-opus": new Set(["temperature"]),
	"claude-3-sonnet": new Set(["temperature"]),
	"claude-3-haiku": new Set(["temperature"]),
	"claude-haiku-4-5": new Set(["temperature"]),
	// OpenAI Codex models don't support temperature, frequency_penalty, or presence_penalty
	"gpt-5.1-codex": new Set(["temperature", "frequency_penalty", "presence_penalty"]),
	"gpt-5.1-codex-mini": new Set(["temperature", "frequency_penalty", "presence_penalty"]),
	"gpt-5.1-codex-max": new Set(["temperature", "frequency_penalty", "presence_penalty"]),
	"codex-mini-latest": new Set(["temperature", "frequency_penalty", "presence_penalty"]),
	// Add more known limitations as discovered
};

/**
 * VS Code Chat provider backed by LiteLLM.
 */
export class LiteLLMChatModelProvider implements LanguageModelChatProvider {
	/** Cache of probed model parameter support to avoid repeated tests */
	private _parameterProbeCache: Map<string, Set<string>> = new Map<string, Set<string>>();
	private _chatEndpoints: { model: string; modelMaxPromptTokens: number }[] = [];
	/** Buffer for assembling streamed tool calls by index. */
	private _toolCallBuffers: Map<number, { id?: string; name?: string; args: string }> = new Map<
		number,
		{ id?: string; name?: string; args: string }
	>();

	/** Indices for which a tool call has been fully emitted. */
	private _completedToolCallIndices = new Set<number>();

	/** Track if we emitted any assistant text before seeing tool calls (SSE-like begin-tool-calls hint). */
	private _hasEmittedAssistantText = false;

	/** Track if we emitted the begin-tool-calls whitespace flush. */
	private _emittedBeginToolCallsHint = false;

	/**
	 * Buffer of assistant text already emitted during the current streaming request.
	 * Used to provide partial output when a request ultimately fails due to rate limiting.
	 */
	private _partialAssistantText = "";

	// Lightweight tokenizer state for tool calls embedded in text
	private _textToolParserBuffer = "";
	private _textToolActive:
		| undefined
		| {
				name?: string;
				index?: number;
				argBuffer: string;
				emitted?: boolean;
		  };
	private _emittedTextToolCallKeys = new Set<string>();
	private _emittedTextToolCallIds = new Set<string>();

	/** Cache of model information by model ID for use during chat requests */
	private _modelInfoCache: Map<string, LiteLLMModelInfo | undefined> = new Map<string, LiteLLMModelInfo | undefined>();

	/**
	 * Create a provider using the given secret storage for the API key.
	 * @param secrets VS Code secret storage.
	 */
	constructor(
		private readonly secrets: vscode.SecretStorage,
		private readonly userAgent: string
	) {}

	/** Roughly estimate tokens for VS Code chat messages (text only) */
	private estimateMessagesTokens(msgs: readonly vscode.LanguageModelChatRequestMessage[]): number {
		let total = 0;
		for (const m of msgs) {
			for (const part of m.content) {
				if (part instanceof vscode.LanguageModelTextPart) {
					total += Math.ceil(part.value.length / 4);
				}
			}
		}
		return total;
	}

	/** Rough token estimate for tool definitions by JSON size */
	private estimateToolTokens(
		tools: { type: string; function: { name: string; description?: string; parameters?: object } }[] | undefined
	): number {
		if (!tools || tools.length === 0) {
			return 0;
		}
		try {
			const json = JSON.stringify(tools);
			return Math.ceil(json.length / 4);
		} catch {
			return 0;
		}
	}

	/**
	 * Get the list of available language models contributed by this provider
	 * @param options Options which specify the calling context of this function
	 * @param token A cancellation token which signals if the user cancelled the request or not
	 * @returns A promise that resolves to the list of available language models
	 */
	async prepareLanguageModelChatInformation(
		options: { silent: boolean },
		_token: CancellationToken
	): Promise<LanguageModelChatInformation[]> {
		console.log("[LiteLLM Model Provider] prepareLanguageModelChatInformation called", { silent: options.silent });

		try {
			const config = await this.ensureConfig(options.silent);
			if (!config) {
				console.log("[LiteLLM Model Provider] No config found, returning empty array");
				return [];
			}
			console.log("[LiteLLM Model Provider] Config loaded", { baseUrl: config.baseUrl, hasApiKey: !!config.apiKey });

			const { models } = await this.fetchModels(config.apiKey, config.baseUrl);
			console.log("[LiteLLM Model Provider] Fetched models", {
				count: models.length,
				modelIds: models.map((m) => m.id),
			});

			const infos: LanguageModelChatInformation[] = models.map((m) => {
				console.log(`[LiteLLM Model Provider] Processing model: ${m.id}`);
				const modelInfo = m.model_info;

				// Cache the model info for later use during chat requests
				this._modelInfoCache.set(m.id, modelInfo);

				// Extract token constraints from model_info
				const maxInputTokens = modelInfo?.max_input_tokens ?? DEFAULT_CONTEXT_LENGTH;
				const maxOutputTokens = modelInfo?.max_output_tokens ?? DEFAULT_MAX_OUTPUT_TOKENS;

				// Build capabilities based on model_info flags
				const capabilities = this.buildCapabilities(modelInfo);

				console.log(`[LiteLLM Model Provider]   - model_info:`, {
					maxInputTokens,
					maxOutputTokens,
					capabilities,
				});

				return {
					id: m.id,
					name: m.model_name ?? m.id,
					tooltip: `${m.model_info?.litellm_provider ?? "LiteLLM"} (${m.model_info?.mode ?? "responses"})`,
					family: "litellm",
					version: "1.0.0",
					maxInputTokens: Math.max(1, maxInputTokens),
					maxOutputTokens: Math.max(1, maxOutputTokens),
					capabilities,
				} satisfies LanguageModelChatInformation;
			});

			this._chatEndpoints = infos.map((info) => ({
				model: info.id,
				modelMaxPromptTokens: info.maxInputTokens + info.maxOutputTokens,
			}));

			console.log("[LiteLLM Model Provider] Final model count:", infos.length);
			console.log(
				"[LiteLLM Model Provider] Model IDs:",
				infos.map((i) => i.id)
			);
			return infos;
		} catch (err) {
			console.error("[LiteLLM Model Provider] prepareLanguageModelChatInformation failed", {
				error: err instanceof Error ? { name: err.name, message: err.message, stack: err.stack } : String(err),
			});

			const errorMessage = err instanceof Error ? err.message : String(err);

			// Handle 403 Forbidden - API key issue
			if (errorMessage.includes("403") || errorMessage.includes("Authorization failed")) {
				vscode.window
					.showWarningMessage(
						"Your LiteLLM API key is invalid or does not have permission to access models. " +
							"The extension will use default models with limited functionality. " +
							"Please reconfigure with a 'default' type API key to enable full features.",
						"Reconfigure",
						"Continue with Defaults"
					)
					.then((selection) => {
						if (selection === "Reconfigure") {
							vscode.commands.executeCommand("litellm.manage");
						}
					});
				// Return empty array - caller will show no models available
				return [];
			}

			// Handle 401 Unauthorized and other auth errors
			if (errorMessage.includes("401") || errorMessage.includes("Authentication")) {
				vscode.window.showErrorMessage("Authentication failed: " + errorMessage, "Reconfigure").then((selection) => {
					if (selection === "Reconfigure") {
						vscode.commands.executeCommand("litellm.manage");
					}
				});
				return [];
			}

			// Handle connection errors
			if (errorMessage.includes("Failed to connect")) {
				vscode.window
					.showErrorMessage("Failed to connect to LiteLLM server: " + errorMessage, "Reconfigure")
					.then((selection) => {
						if (selection === "Reconfigure") {
							vscode.commands.executeCommand("litellm.manage");
						}
					});
				return [];
			}

			// Generic error
			console.error("[LiteLLM Model Provider] Unexpected error:", errorMessage);
			return [];
		}
	}

	/**
	 * Build capabilities object from model_info flags.
	 * Maps LiteLLM model capabilities to VSCode LanguageModelChatInformation capabilities.
	 */
	private buildCapabilities(modelInfo: LiteLLMModelInfo | undefined): vscode.LanguageModelChatCapabilities {
		if (!modelInfo) {
			// Default capabilities if no model_info available
			return {
				toolCalling: true,
				imageInput: false,
			};
		}

		// Map LiteLLM capabilities to VSCode capabilities
		const capabilities: vscode.LanguageModelChatCapabilities = {
			// Tool calling is supported if function_calling is supported
			toolCalling: modelInfo.supports_function_calling !== false,
			// Image input is supported if vision is supported
			imageInput: modelInfo.supports_vision === true,
		};

		return capabilities;
	}

	/**
	 * Record that a parameter failed for a model (discovered at runtime).
	 * Used to cache parameter incompatibilities discovered through failed API calls.
	 */
	private recordParameterUnsupported(modelId: string, param: string): void {
		if (!this._parameterProbeCache.has(modelId)) {
			this._parameterProbeCache.set(modelId, new Set());
		}
		this._parameterProbeCache.get(modelId)!.add(param);
		console.log(`[LiteLLM Model Provider] Cached parameter unsupported: ${modelId} does not support ${param}`);
	}

	/**
	 * Remove unsupported parameters from the request body as a final safety net.
	 */
	private stripUnsupportedParametersFromRequest(
		requestBody: Record<string, unknown>,
		modelInfo: LiteLLMModelInfo | undefined,
		modelId?: string
	): void {
		const paramsToCheck = ["temperature", "stop", "frequency_penalty", "presence_penalty"];
		for (const p of paramsToCheck) {
			if (!this.isParameterSupported(p, modelInfo, modelId) && p in requestBody) {
				delete requestBody[p];
				console.log(`[LiteLLM Model Provider] Removed unsupported parameter '${p}' for model ${modelId}`);
			}
		}
	}

	/**
	 * Check if a parameter is supported by the model.
	 * Uses a multi-level approach:
	 * 1. Check known limitations (hardcoded list of models that don't support certain params)
	 * 2. Check cached probe results (if model was tested at runtime)
	 * 3. Check supported_openai_params from LiteLLM (with skepticism)
	 */
	private isParameterSupported(param: string, modelInfo: LiteLLMModelInfo | undefined, modelId?: string): boolean {
		// 1. Check known limitations (do this BEFORE checking modelInfo, as it doesn't depend on it)
		if (modelId) {
			// Exact match
			if (KNOWN_PARAMETER_LIMITATIONS[modelId]?.has(param)) {
				console.log(
					`[LiteLLM Model Provider] Parameter '${param}' blocked for ${modelId} by exact match in KNOWN_PARAMETER_LIMITATIONS`
				);
				return false;
			}
			// Substring match for model families (e.g., "claude-haiku-4-5" matches "claude-haiku")
			for (const [knownModel, limitations] of Object.entries(KNOWN_PARAMETER_LIMITATIONS)) {
				if (modelId.includes(knownModel) && limitations.has(param)) {
					console.log(
						`[LiteLLM Model Provider] Parameter '${param}' blocked for ${modelId} by substring match with ${knownModel}`
					);
					return false;
				}
			}
		}

		if (!modelInfo) {
			return true; // No model info, assume supported
		}

		// 2. Check cached probe results
		if (modelId && this._parameterProbeCache.has(modelId)) {
			const unsupportedParams = this._parameterProbeCache.get(modelId);
			if (unsupportedParams?.has(param)) {
				console.log(`[LiteLLM Model Provider] Parameter '${param}' blocked for ${modelId} by probe cache`);
				return false;
			}
		}

		// 3. Check supported_openai_params (treat as soft requirement - model claims to support it)
		if (modelInfo?.supported_openai_params) {
			const supported = modelInfo.supported_openai_params.includes(param);
			console.log(
				`[LiteLLM Model Provider] Parameter '${param}' in supported_openai_params for ${modelId}: ${supported}`
			);
			return supported;
		}

		// 4. Default to true for backward compatibility
		console.log(`[LiteLLM Model Provider] Parameter '${param}' defaulting to true (no model info)`);
		return true;
	}

	/**
	 * Get the appropriate completions endpoint based on model mode.
	 * Uses "responses" endpoint if mode is "responses", otherwise defaults to "chat/completions".
	 */
	private getCompletionsEndpoint(mode: string | undefined): string {
		if (mode === "responses") {
			return "/responses";
		}
		return "/chat/completions";
	}

	/**
	 * Parse error response from LiteLLM API and extract human-readable message.
	 */
	private parseApiError(statusCode: number, errorText: string): string {
		try {
			const parsed = JSON.parse(errorText);
			if (parsed.error?.message) {
				return parsed.error.message;
			}
		} catch {
			/* ignore */
		}
		if (errorText) {
			return errorText.slice(0, 200);
		}
		return `API request failed with status ${statusCode}`;
	}

	async provideLanguageModelChatInformation(
		options: { silent: boolean },
		_token: CancellationToken
	): Promise<LanguageModelChatInformation[]> {
		return this.prepareLanguageModelChatInformation({ silent: options.silent ?? false }, _token);
	}

	/**
	 * Fetch the list of models and supplementary metadata from LiteLLM.
	 * @param apiKey The LiteLLM API key used to authenticate.
	 * @param baseUrl The LiteLLM base URL.
	 */
	private async fetchModels(apiKey: string, baseUrl: string): Promise<{ models: TransformedModelItem[] }> {
		console.log("[LiteLLM Model Provider] fetchModels called", { baseUrl, hasApiKey: !!apiKey });
		try {
			const headers: Record<string, string> = { "User-Agent": this.userAgent };
			if (apiKey) {
				// Try both authentication methods: standard Bearer and X-API-Key
				headers.Authorization = `Bearer ${apiKey}`;
				headers["X-API-Key"] = apiKey;
			}

			console.log("[LiteLLM Model Provider] Fetching from:", `${baseUrl}/model/info`);
			let resp: Response;
			try {
				resp = await this.fetchWithRateLimit(`${baseUrl}/model/info`, {
					method: "GET",
					headers,
				});
			} catch (fetchError) {
				console.error("[LiteLLM Model Provider] Fetch error:", fetchError);
				throw new Error(
					`Failed to connect to LiteLLM server at ${baseUrl}: ${fetchError instanceof Error ? fetchError.message : String(fetchError)}`
				);
			}

			console.log("[LiteLLM Model Provider] Response status:", resp.status, resp.statusText);
			if (!resp.ok) {
				let text = "";
				try {
					text = await resp.text();
				} catch (error) {
					console.error("[LiteLLM Model Provider] Failed to read response text", error);
				}

				// Provide helpful error message for authentication/authorization failures
				if (resp.status === 401) {
					const err = new Error(
						`Authentication failed (401): Your LiteLLM server requires a valid API key. Please run the "Manage LiteLLM Provider" command to update your API key.`
					);
					console.error("[LiteLLM Model Provider] Authentication error", err);
					throw err;
				}

				if (resp.status === 403) {
					const err = new Error(
						`Authorization failed (403): Your API key is invalid or does not have permission to access models. Please run the "Manage LiteLLM Provider" command to update your API key.`
					);
					console.error("[LiteLLM Model Provider] Authorization error", err);
					throw err;
				}

				const err = new Error(
					`Failed to fetch LiteLLM models: ${resp.status} ${resp.statusText}${text ? `\n${text}` : ""}`
				);
				console.error("[LiteLLM Model Provider] Failed to fetch LiteLLM models", err);
				throw err;
			}

			let parsed: LiteLLMModelInfoResponse;
			try {
				parsed = (await resp.json()) as LiteLLMModelInfoResponse;
			} catch (jsonError) {
				console.error("[LiteLLM Model Provider] Failed to parse response JSON", jsonError);
				throw new Error(
					`Failed to parse LiteLLM response: ${jsonError instanceof Error ? jsonError.message : String(jsonError)}`
				);
			}

			console.log("[LiteLLM Model Provider] Parsed response:", {
				modelCount: parsed.data?.length ?? 0,
			});
			if (parsed.data && parsed.data.length > 0) {
				console.log("[LiteLLM Model Provider] First model sample:", JSON.stringify(parsed.data[0], null, 2));
			}

			// Transform LiteLLM response format to internal format
			const transformed: TransformedModelItem[] = (parsed.data ?? []).map((entry, index) => ({
				id: entry.model_info?.key ?? entry.model_name ?? `model-${index}`,
				object: "model",
				created: Date.now(),
				owned_by: entry.model_info?.litellm_provider ?? "litellm",
				model_name: entry.model_name,
				litellm_params: entry.litellm_params,
				model_info: entry.model_info,
			}));

			console.log("[LiteLLM Model Provider] Successfully fetched models:", transformed.length);
			return { models: transformed };
		} catch (err) {
			console.error("[LiteLLM Model Provider] fetchModels failed", {
				error: err instanceof Error ? { name: err.name, message: err.message, stack: err.stack } : String(err),
			});
			throw err;
		}
	}

	/**
	 * Returns the response for a chat request, passing the results to the progress callback.
	 * The {@linkcode LanguageModelChatProvider} must emit the response parts to the progress callback as they are received from the language model.
	 * @param model The language model to use
	 * @param messages The messages to include in the request
	 * @param options Options for the request
	 * @param progress The progress to emit the streamed response chunks to
	 * @param token A cancellation token for the request
	 * @returns A promise that resolves when the response is complete. Results are actually passed to the progress callback.
	 */
	async provideLanguageModelChatResponse(
		model: LanguageModelChatInformation,
		messages: readonly LanguageModelChatRequestMessage[],
		options: ProvideLanguageModelChatResponseOptions,
		progress: Progress<LanguageModelResponsePart>,
		token: CancellationToken
	): Promise<void> {
		this._toolCallBuffers.clear();
		this._completedToolCallIndices.clear();
		this._hasEmittedAssistantText = false;
		this._emittedBeginToolCallsHint = false;
		this._textToolParserBuffer = "";
		this._textToolActive = undefined;
		this._emittedTextToolCallKeys.clear();
		this._emittedTextToolCallIds.clear();
		this._partialAssistantText = "";

		let requestBody: Record<string, unknown> | undefined;
		const trackingProgress: Progress<LanguageModelResponsePart> = {
			report: (part) => {
				try {
					if (part instanceof vscode.LanguageModelTextPart) {
						this._partialAssistantText += part.value;
					}
					progress.report(part);
				} catch (e) {
					console.error("[LiteLLM Model Provider] Progress.report failed", {
						modelId: model.id,
						error: e instanceof Error ? { name: e.name, message: e.message } : String(e),
					});
				}
			},
		};
		try {
			const config = await this.ensureConfig(true);
			if (!config) {
				throw new Error("LiteLLM configuration not found. Please configure the LiteLLM provider.");
			}

			// For responses API, trim messages to prevent overwhelming the endpoint
			// Keep first system message + last N user/assistant messages for context
			const messagesToUse = messages;

			const openaiMessages = convertMessages(messagesToUse);
			validateRequest(messagesToUse);
			const toolConfig = convertTools(options);

			if (options.tools && options.tools.length > 128) {
				throw new Error("Cannot have more than 128 tools per request.");
			}

			const inputTokenCount = this.estimateMessagesTokens(messages);
			const toolTokenCount = this.estimateToolTokens(toolConfig.tools);
			const tokenLimit = Math.max(1, model.maxInputTokens);
			if (inputTokenCount + toolTokenCount > tokenLimit) {
				console.error("[LiteLLM Model Provider] Message exceeds token limit", {
					total: inputTokenCount + toolTokenCount,
					tokenLimit,
				});
				throw new Error("Message exceeds token limit.");
			}

			requestBody = {
				model: model.id,
				messages: openaiMessages,
				stream: true,
				max_tokens: Math.min(options.modelOptions?.max_tokens || 4096, model.maxOutputTokens),
			};

			// Retrieve model info from cache
			const modelInfo = this._modelInfoCache.get(model.id);

			console.log(`[LiteLLM Model Provider] Model info supported_openai_params:`, modelInfo?.supported_openai_params);

			// Only include temperature if the model supports it
			const tempSupported = this.isParameterSupported("temperature", modelInfo, model.id);
			console.log(`[LiteLLM Model Provider] Temperature supported for ${model.id}: ${tempSupported}`);
			if (tempSupported) {
				(requestBody as Record<string, unknown>).temperature = options.modelOptions?.temperature ?? 0.7;
				console.log(
					`[LiteLLM Model Provider] Added temperature: ${(requestBody as Record<string, unknown>).temperature}`
				);
			}

			// Allow-list model options based on supported parameters
			if (options.modelOptions) {
				const mo = options.modelOptions as Record<string, unknown>;
				if (this.isParameterSupported("stop", modelInfo, model.id)) {
					if (typeof mo.stop === "string" || Array.isArray(mo.stop)) {
						(requestBody as Record<string, unknown>).stop = mo.stop;
					}
				}
				if (this.isParameterSupported("frequency_penalty", modelInfo, model.id)) {
					if (typeof mo.frequency_penalty === "number") {
						(requestBody as Record<string, unknown>).frequency_penalty = mo.frequency_penalty;
					}
				}
				if (this.isParameterSupported("presence_penalty", modelInfo, model.id)) {
					if (typeof mo.presence_penalty === "number") {
						(requestBody as Record<string, unknown>).presence_penalty = mo.presence_penalty;
					}
				}
			}

			console.log(`[LiteLLM Model Provider] Request body before strip:`, JSON.stringify(requestBody));
			// Final safety: strip any unsupported parameters that slipped through earlier checks
			this.stripUnsupportedParametersFromRequest(requestBody as Record<string, unknown>, modelInfo, model.id);
			console.log(`[LiteLLM Model Provider] Request body after strip:`, JSON.stringify(requestBody));

			if (toolConfig.tools) {
				(requestBody as Record<string, unknown>).tools = toolConfig.tools;
				console.log("[LiteLLM Model Provider] Added tools to request body:", {
					toolCount: toolConfig.tools.length,
					tools: JSON.stringify(toolConfig.tools, null, 2),
				});
			}
			if (toolConfig.tool_choice) {
				(requestBody as Record<string, unknown>).tool_choice = toolConfig.tool_choice;
				console.log("[LiteLLM Model Provider] Added tool_choice to request body:", toolConfig.tool_choice);
			}
			const headers: Record<string, string> = {
				"Content-Type": "application/json",
				"User-Agent": this.userAgent,
			};
			if (config.apiKey) {
				// Try both authentication methods: standard Bearer and X-API-Key
				headers.Authorization = `Bearer ${config.apiKey}`;
				headers["X-API-Key"] = config.apiKey;
			}
			const endpoint = this.getCompletionsEndpoint(modelInfo?.mode);

			// Transform request body if using responses endpoint
			let finalRequestBody: Record<string, unknown> = requestBody as Record<string, unknown>;
			if (endpoint === "/responses") {
				finalRequestBody = this.transformToResponsesFormat(finalRequestBody);
			}

			console.log("[LiteLLM Model Provider] Sending chat request", {
				url: `${config.baseUrl}${endpoint}`,
				modelId: model.id,
				messageCount: messages.length,
				endpoint: endpoint,
				requestBody: JSON.stringify(finalRequestBody, null, 2),
			});
			const response = await this.fetchWithRateLimit(`${config.baseUrl}${endpoint}`, {
				method: "POST",
				headers,
				body: JSON.stringify(finalRequestBody),
			});

			console.log("[LiteLLM Model Provider] Received response", {
				status: response.status,
				statusText: response.statusText,
				contentType: response.headers.get("content-type"),
				hasBody: !!response.body,
			});

			if (!response.ok) {
				const errorText = await response.text();
				console.error("[LiteLLM Model Provider] API error response", errorText);

				if (response.status === 429) {
					throw new Error("LiteLLM rate limit exceeded. Please try again later.");
				}

				// Provide helpful error message for authentication failures
				if (response.status === 401) {
					throw new Error(
						`Authentication failed: Your LiteLLM server requires an API key. Please run the "Manage LiteLLM Provider" command to configure your API key.`
					);
				}

				const parsedError = this.parseApiError(response.status, errorText);

				// Detect unsupported parameters and cache them
				const parameterErrors = [
					{ param: "temperature", patterns: ["temperature", "param temperature"] },
					{ param: "stop", patterns: ["stop_sequences", "stop"] },
					{ param: "frequency_penalty", patterns: ["frequency_penalty"] },
					{ param: "presence_penalty", patterns: ["presence_penalty"] },
				];

				for (const { param, patterns } of parameterErrors) {
					if (patterns.some((p) => errorText.toLowerCase().includes(p))) {
						this.recordParameterUnsupported(model.id, param);
					}
				}

				throw new Error(
					`LiteLLM API error: ${response.status} ${response.statusText}${parsedError ? `\n${parsedError}` : ""}`
				);
			}

			if (!response.body) {
				throw new Error("No response body from LiteLLM API");
			}
			await this.processStreamingResponse(response.body, trackingProgress, token);
		} catch (err) {
			console.error("[LiteLLM Model Provider] Chat request failed", {
				modelId: model.id,
				messageCount: messages.length,
				error: err instanceof Error ? { name: err.name, message: err.message } : String(err),
			});

			// If we hit a rate limit and have partial output, surface it back to the user.
			const errMsg = err instanceof Error ? err.message : String(err);
			if ((/\b429\b/.test(errMsg) || /\brate limit\b/i.test(errMsg)) && this._partialAssistantText.trim().length > 0) {
				try {
					progress.report(
						new vscode.LanguageModelTextPart(
							"\n\n[Rate limit exceeded; returned partial output. Please try again later.]\n"
						)
					);
				} catch {
					// ignore
				}
			}
			throw err;
		}
	}

	/**
	 * Fetch with automatic retries for transient errors (500).
	 */
	private async fetchWithRetry(
		url: string,
		init: RequestInit,
		options?: { retries?: number; delayMs?: number }
	): Promise<Response> {
		const maxRetries = options?.retries ?? 2;
		const delayMs = options?.delayMs ?? 1000;
		let attempt = 0;
		while (true) {
			try {
				const response = await fetch(url, init);
				if (response.ok || attempt >= maxRetries || response.status < 500 || response.status >= 600) {
					return response;
				}
				attempt++;
				console.warn(
					`[LiteLLM Model Provider] Retryable response (status ${response.status}). Attempt ${attempt}/${maxRetries}`
				);
				await new Promise((resolve) => setTimeout(resolve, delayMs));
			} catch (err) {
				if (attempt >= maxRetries) {
					throw err;
				}
				attempt++;
				console.warn("[LiteLLM Model Provider] Fetch failed, retrying", { attempt, error: err });
				await new Promise((resolve) => setTimeout(resolve, delayMs));
			}
		}
	}

	/**
	 * Fetch with exponential back-off for rate limiting (429).
	 * Retries with exponential delay up to a maximum cumulative delay of 2 minutes.
	 * For other transient errors, it delegates to {@link fetchWithRetry}.
	 */
	private async fetchWithRateLimit(
		url: string,
		init: RequestInit,
		options?: { maxTotalDelayMs?: number; initialDelayMs?: number }
	): Promise<Response> {
		const maxTotalDelayMs = options?.maxTotalDelayMs ?? 120_000;
		const initialDelayMs = options?.initialDelayMs ?? 500;
		let cumulativeDelayMs = 0;
		let attempt = 0;

		while (true) {
			// Use existing retry logic for transient 5xx errors.
			const response = await this.fetchWithRetry(url, init, { retries: 2, delayMs: 1000 });
			if (response.status !== 429) {
				return response;
			}

			const remaining = maxTotalDelayMs - cumulativeDelayMs;
			if (remaining <= 0) {
				return response;
			}

			const nextDelayMs = Math.min(initialDelayMs * Math.pow(2, attempt), remaining);
			attempt++;
			cumulativeDelayMs += nextDelayMs;
			console.warn(
				`[LiteLLM Model Provider] Rate limit (429). Retrying in ${nextDelayMs}ms (total backoff ${cumulativeDelayMs}ms/${maxTotalDelayMs}ms)`
			);
			await new Promise((resolve) => setTimeout(resolve, nextDelayMs));
		}
	}

	/**
	 * Returns the number of tokens for a given text using the model specific tokenizer logic
	 * @param model The language model to use
	 * @param text The text to count tokens for
	 * @param token A cancellation token for the request
	 * @returns A promise that resolves to the number of tokens
	 */
	async provideTokenCount(
		model: LanguageModelChatInformation,
		text: string | LanguageModelChatRequestMessage,
		_token: CancellationToken
	): Promise<number> {
		if (typeof text === "string") {
			return Math.ceil(text.length / 4);
		} else {
			let totalTokens = 0;
			for (const part of text.content) {
				if (part instanceof vscode.LanguageModelTextPart) {
					totalTokens += Math.ceil(part.value.length / 4);
				}
			}
			return totalTokens;
		}
	}

	/**
	 * Ensure base URL and API key exist in SecretStorage, optionally prompting the user when not silent.
	 * Validates the API key by testing the connection to the LiteLLM server.
	 * @param silent If true, do not prompt the user.
	 */
	private async ensureConfig(silent: boolean): Promise<{ baseUrl: string; apiKey: string } | undefined> {
		console.log("[LiteLLM Model Provider] ensureConfig called", { silent });
		let baseUrl = await this.secrets.get("litellm.baseUrl");
		let apiKey = await this.secrets.get("litellm.apiKey");
		console.log("[LiteLLM Model Provider] Retrieved from secrets:", { hasBaseUrl: !!baseUrl, hasApiKey: !!apiKey });

		if (!baseUrl && !silent) {
			const entered = await vscode.window.showInputBox({
				title: "LiteLLM Base URL",
				prompt: "Enter your LiteLLM base URL (e.g., http://localhost:4000 or https://api.litellm.ai)",
				ignoreFocusOut: true,
				placeHolder: "http://localhost:4000",
			});
			if (entered && entered.trim()) {
				baseUrl = entered.trim();
				await this.secrets.store("litellm.baseUrl", baseUrl);
			}
		}

		// Prompt for API key and validate it
		if (!silent) {
			let apiKeyValid = false;
			let promptCount = 0;
			const maxPrompts = 3;

			while (!apiKeyValid && promptCount < maxPrompts) {
				promptCount++;
				const entered = await vscode.window.showInputBox({
					title: "LiteLLM API Key",
					prompt: `Enter your LiteLLM API key as a 'default' type (leave empty to skip). Attempt ${promptCount}/${maxPrompts}`,
					ignoreFocusOut: true,
					password: true,
				});

				if (entered === undefined) {
					// User cancelled
					break;
				}

				if (!entered.trim()) {
					// User left it empty - allow
					apiKey = "";
					apiKeyValid = true;
					await this.secrets.delete("litellm.apiKey");
				} else if (baseUrl) {
					// Validate the API key by testing the connection (only if baseUrl is set)
					const isValid = await this.validateApiKey(baseUrl, entered.trim());
					if (isValid) {
						apiKey = entered.trim();
						await this.secrets.store("litellm.apiKey", apiKey);
						apiKeyValid = true;
						vscode.window.showInformationMessage("API key validated successfully!");
					} else {
						if (promptCount < maxPrompts) {
							vscode.window.showErrorMessage(
								`API key validation failed. Please check your key is a 'default' type. (${promptCount}/${maxPrompts})`
							);
						} else {
							vscode.window.showErrorMessage(
								`Could not validate API key after ${maxPrompts} attempts. Please reconfigure.`
							);
							return undefined;
						}
					}
				} else {
					// No baseUrl, can't validate
					apiKey = entered.trim();
					await this.secrets.store("litellm.apiKey", apiKey);
					apiKeyValid = true;
				}
			}
		}

		if (!baseUrl) {
			console.log("[LiteLLM Model Provider] No baseUrl configured, returning undefined");
			return undefined;
		}

		console.log("[LiteLLM Model Provider] Config ready:", { baseUrl, hasApiKey: !!apiKey });
		return { baseUrl, apiKey: apiKey ?? "" };
	}

	/**
	 * Validate the API key by attempting to connect to the LiteLLM server.
	 * @param baseUrl The LiteLLM base URL
	 * @param apiKey The API key to validate
	 * @returns true if the API key is valid, false otherwise
	 */
	private async validateApiKey(baseUrl: string, apiKey: string): Promise<boolean> {
		try {
			console.log("[LiteLLM Model Provider] Validating API key...");
			const headers: Record<string, string> = { "User-Agent": this.userAgent };
			if (apiKey) {
				headers.Authorization = `Bearer ${apiKey}`;
				headers["X-API-Key"] = apiKey;
			}

			const resp = await fetch(`${baseUrl}/model/info`, {
				method: "GET",
				headers,
			});

			console.log("[LiteLLM Model Provider] API key validation response:", resp.status, resp.statusText);

			// Accept 200 (success) or 401/403 (would mean key exists but has issues)
			// We're just checking if the server is reachable and responding
			if (resp.ok) {
				console.log("[LiteLLM Model Provider] API key validation succeeded");
				return true;
			}

			if (resp.status === 401 || resp.status === 403) {
				console.log("[LiteLLM Model Provider] API key invalid (401/403)");
				return false;
			}

			console.log("[LiteLLM Model Provider] API key validation failed with status:", resp.status);
			return false;
		} catch (err) {
			console.error("[LiteLLM Model Provider] API key validation error:", err);
			return false;
		}
	}

	/**
	 * Read and parse the LiteLLM streaming (SSE-like) response and report parts.
	 * @param responseBody The readable stream body.
	 * @param progress Progress reporter for streamed parts.
	 * @param token Cancellation token.
	 */
	private async processStreamingResponse(
		responseBody: ReadableStream<Uint8Array>,
		progress: vscode.Progress<vscode.LanguageModelResponsePart>,
		token: vscode.CancellationToken
	): Promise<void> {
		const reader = responseBody.getReader();
		const decoder = new TextDecoder();
		let buffer = "";

		try {
			while (!token.isCancellationRequested) {
				const { done, value } = await reader.read();
				if (done) {
					console.log("[LiteLLM Model Provider] Stream ended");
					break;
				}

				buffer += decoder.decode(value, { stream: true });
				const lines = buffer.split("\n");
				buffer = lines.pop() || "";

				for (const line of lines) {
					// Log non-empty lines for debugging
					if (line.length > 0) {
						console.log("[LiteLLM Model Provider] Received line:", line.slice(0, 200));
					}

					if (!line.startsWith("data: ")) {
						continue;
					}
					const data = line.slice(6);
					if (data === "[DONE]") {
						console.log("[LiteLLM Model Provider] Received [DONE] marker");
						// Do not throw on [DONE]; any incomplete/empty buffers are ignored.
						await this.flushToolCallBuffers(progress, /*throwOnInvalid*/ false);
						// Flush any in-progress text-embedded tool call (silent if incomplete)
						await this.flushActiveTextToolCall(progress);
						continue;
					}

					try {
						const parsed = JSON.parse(data);
						console.log("[LiteLLM Model Provider] Parsed JSON successfully");
						await this.processDelta(parsed, progress);
					} catch (e) {
						// Log parsing errors
						console.warn("[LiteLLM Model Provider] Failed to parse JSON:", {
							data: data.slice(0, 200),
							error: e instanceof Error ? e.message : String(e),
						});
					}
				}
			}
		} finally {
			reader.releaseLock();
			// Clean up any leftover tool call state
			this._toolCallBuffers.clear();
			this._completedToolCallIndices.clear();
			this._hasEmittedAssistantText = false;
			this._emittedBeginToolCallsHint = false;
			this._textToolParserBuffer = "";
			this._textToolActive = undefined;
			this._emittedTextToolCallKeys.clear();
		}
	}

	/**
	 * Handle a single streamed delta chunk, emitting text and tool call parts.
	 * Supports both /chat/completions format (with choices) and /responses format (with output).
	 * @param delta Parsed SSE chunk from LiteLLM.
	 * @param progress Progress reporter for parts.
	 */
	private async processDelta(
		delta: Record<string, unknown>,
		progress: vscode.Progress<vscode.LanguageModelResponsePart>
	): Promise<boolean> {
		let emitted = false;

		// Handle OpenAI Responses API format (event-based streaming)
		// Each delta has a "type" field indicating what kind of event it is
		const eventType = delta.type as string | undefined;

		// Log all event types for debugging
		if (eventType && !eventType.startsWith("response.in_progress") && eventType !== "response.created") {
			console.log("[LiteLLM Model Provider] Received event type:", {
				eventType,
				deltaKeys: Object.keys(delta),
				deltaPreview: JSON.stringify(delta).slice(0, 200),
			});
		}

		if (eventType === "response.output_text.delta") {
			// Text delta from responses API - try multiple field names for the text chunk
			// Different implementations may use: delta, text, or chunk
			let textDelta = delta.delta as string | undefined;
			if (!textDelta) {
				textDelta = delta.text as string | undefined;
			}
			if (!textDelta) {
				textDelta = delta.chunk as string | undefined;
			}
			if (textDelta) {
				console.log("[LiteLLM Model Provider] Processing text delta:", textDelta.slice(0, 50));
				progress.report(new vscode.LanguageModelTextPart(textDelta));
				return true;
			}
			console.log(
				"[LiteLLM Model Provider] Text delta event received but no text found in delta, text, or chunk fields"
			);
			return false;
		}

		if (eventType === "response.output_text.done") {
			// Complete text message - has "text" field with full message
			// We don't need to emit here since we already streamed the deltas
			console.log("[LiteLLM Model Provider] Text output complete");
			return false;
		}

		// Handle function call arguments completion from responses API
		if (eventType === "response.function_call_arguments.done") {
			// This event has the complete function call with all arguments assembled
			const argumentsStr = delta.arguments as string | undefined;
			const itemId = delta.item_id as string | undefined;

			if (argumentsStr && itemId) {
				try {
					console.log("[LiteLLM Model Provider] Processing function call arguments.done:", {
						itemId,
						argumentsPreview: argumentsStr.slice(0, 100),
					});

					const parsed = tryParseJSONObject(argumentsStr);
					if (!parsed.ok) {
						console.warn("[LiteLLM Model Provider] Failed to parse function arguments:", argumentsStr);
						return false;
					}

					// At this point we have the arguments but not the function name
					// The name should have come from the response.output_item.added event
					// For now, we'll need to track it separately or get it from another source
					console.log("[LiteLLM Model Provider] Have arguments for function call:", {
						itemId,
						arguments: parsed.value,
					});
					return false; // We'll emit when we have the name
				} catch (e) {
					console.warn("[LiteLLM Model Provider] Error processing function call arguments:", e);
					return false;
				}
			}
			return false;
		}

		// Handle function call item completion
		if (eventType === "response.output_item.done") {
			// This tells us the function call is complete
			const item = delta.item as Record<string, unknown> | undefined;

			if (item && item.type === "function_call") {
				const callId = item.call_id as string | undefined;
				const argumentsStr = item.arguments as string | undefined;
				const name = item.name as string | undefined;

				console.log("[LiteLLM Model Provider] Function call output_item.done:", {
					callId,
					name,
					hasArguments: !!argumentsStr,
					argumentsPreview: argumentsStr ? String(argumentsStr).slice(0, 100) : undefined,
				});

				if (callId && argumentsStr) {
					try {
						const parsed = tryParseJSONObject(argumentsStr);
						if (!parsed.ok) {
							console.warn("[LiteLLM Model Provider] Failed to parse function arguments:", argumentsStr);
							return false;
						}

						// Use the item_id as call_id if name is not available
						// The function name might be embedded in the arguments
						const toolName = name || "unknown_tool";

						console.log("[LiteLLM Model Provider] Emitting tool call:", {
							callId,
							name: toolName,
							argumentsKeys: Object.keys(parsed.value),
						});

						progress.report(new vscode.LanguageModelToolCallPart(callId, toolName, parsed.value));
						return true;
					} catch (e) {
						console.warn("[LiteLLM Model Provider] Error processing function call:", e);
						return false;
					}
				}
			}
			return false;
		}

		// Handle function call items being added (with partial data)
		if (eventType === "response.output_item.added") {
			const item = delta.item as Record<string, unknown> | undefined;

			if (item && item.type === "function_call") {
				const callId = item.call_id as string | undefined;
				const name = item.name as string | undefined;

				console.log("[LiteLLM Model Provider] Function call output_item.added:", {
					callId,
					name,
					status: item.status,
				});
			}
			return false; // Wait for output_item.done or arguments.done
		}

		// Handle /chat/completions format (choices array)
		let choice = (delta.choices as Record<string, unknown>[] | undefined)?.[0];

		// Handle /responses format (output array) - legacy support
		if (!choice) {
			const output = (delta.output as Record<string, unknown>[] | undefined)?.[0];
			if (output) {
				// Convert responses format to choices-like format for compatibility
				const content = output.content as Record<string, unknown>[] | undefined;
				if (content && content.length > 0) {
					// Extract text content from responses format
					const textContent = content.find((c) => (c as Record<string, unknown>).type === "output_text");
					if (textContent) {
						choice = {
							delta: { content: (textContent as Record<string, unknown>).text },
							finish_reason: output.finish_reason,
						};
					}
				}
			}
		}

		// Additional fallback: try to extract text directly if it exists at top level
		// Some responses might have "text" or "content" fields at the root level
		if (!choice && !eventType) {
			const content = delta.content as string | undefined;
			if (content) {
				console.log("[LiteLLM Model Provider] Found text content at root level:", content.slice(0, 50));
				choice = {
					delta: { content },
					finish_reason: undefined,
				};
			} else {
				const text = delta.text as string | undefined;
				if (text) {
					console.log("[LiteLLM Model Provider] Found text at root level:", text.slice(0, 50));
					choice = {
						delta: { content: text },
						finish_reason: undefined,
					};
				}
			}
		}

		if (!choice) {
			// Not a recognized format, skip
			return false;
		}

		const deltaObj = choice.delta as Record<string, unknown> | undefined;

		// report thinking progress if backend provides it and host supports it
		try {
			const maybeThinking =
				(choice as Record<string, unknown> | undefined)?.thinking ??
				(deltaObj as Record<string, unknown> | undefined)?.thinking;
			if (maybeThinking !== undefined) {
				const vsAny = vscode as unknown as Record<string, unknown>;
				const ThinkingCtor = vsAny["LanguageModelThinkingPart"] as
					| (new (text: string, id?: string, metadata?: unknown) => unknown)
					| undefined;
				if (ThinkingCtor) {
					let text = "";
					let id: string | undefined;
					let metadata: unknown;
					if (maybeThinking && typeof maybeThinking === "object") {
						const mt = maybeThinking as Record<string, unknown>;
						text = typeof mt["text"] === "string" ? (mt["text"] as string) : "";
						id = typeof mt["id"] === "string" ? (mt["id"] as string) : undefined;
						metadata = mt["metadata"];
					} else if (typeof maybeThinking === "string") {
						text = maybeThinking;
					}
					if (text) {
						progress.report(
							new (ThinkingCtor as new (text: string, id?: string, metadata?: unknown) => unknown)(
								text,
								id,
								metadata
							) as unknown as vscode.LanguageModelResponsePart
						);
						emitted = true;
					}
				}
			}
		} catch {
			// ignore errors here temporarily
		}
		if (deltaObj?.content) {
			const content = String(deltaObj.content);
			const res = this.processTextContent(content, progress);
			if (res.emittedText) {
				this._hasEmittedAssistantText = true;
			}
			if (res.emittedAny) {
				emitted = true;
			}
		}

		if (deltaObj?.tool_calls) {
			const toolCalls = deltaObj.tool_calls as Array<Record<string, unknown>>;

			// SSEProcessor-like: if first tool call appears after text, emit a whitespace
			// to ensure any UI buffers/linkifiers are flushed without adding visible noise.
			if (!this._emittedBeginToolCallsHint && this._hasEmittedAssistantText && toolCalls.length > 0) {
				progress.report(new vscode.LanguageModelTextPart(" "));
				this._emittedBeginToolCallsHint = true;
			}

			for (const tc of toolCalls) {
				const idx = (tc.index as number) ?? 0;
				// Ignore any further deltas for an index we've already completed
				if (this._completedToolCallIndices.has(idx)) {
					continue;
				}
				const buf = this._toolCallBuffers.get(idx) ?? { args: "" };
				if (tc.id && typeof tc.id === "string") {
					buf.id = tc.id as string;
				}
				const func = tc.function as Record<string, unknown> | undefined;
				if (func?.name && typeof func.name === "string") {
					buf.name = func.name as string;
				}
				if (typeof func?.arguments === "string") {
					buf.args += func.arguments as string;
				}
				this._toolCallBuffers.set(idx, buf);

				// Emit immediately once arguments become valid JSON to avoid perceived hanging
				await this.tryEmitBufferedToolCall(idx, progress);
			}
		}

		const finish = (choice.finish_reason as string | undefined) ?? undefined;
		if (finish === "tool_calls" || finish === "stop") {
			// On both 'tool_calls' and 'stop', emit any buffered calls and throw on invalid JSON
			await this.flushToolCallBuffers(progress, /*throwOnInvalid*/ true);
		}
		return emitted;
	}

	/**
	 * Process streamed text content for inline tool-call control tokens and emit text/tool calls.
	 * Returns which parts were emitted for logging/flow control.
	 */
	private processTextContent(
		input: string,
		progress: vscode.Progress<vscode.LanguageModelResponsePart>
	): { emittedText: boolean; emittedAny: boolean } {
		const BEGIN = "<|tool_call_begin|>";
		const ARG_BEGIN = "<|tool_call_argument_begin|>";
		const END = "<|tool_call_end|>";

		let data = this._textToolParserBuffer + input;
		let emittedText = false;
		let emittedAny = false;
		let visibleOut = "";

		while (data.length > 0) {
			if (!this._textToolActive) {
				const b = data.indexOf(BEGIN);
				if (b === -1) {
					// No tool-call start: emit visible portion, but keep any partial BEGIN prefix as buffer
					const longestPartialPrefix = ((): number => {
						for (let k = Math.min(BEGIN.length - 1, data.length - 1); k > 0; k--) {
							if (data.endsWith(BEGIN.slice(0, k))) {
								return k;
							}
						}
						return 0;
					})();
					if (longestPartialPrefix > 0) {
						const visible = data.slice(0, data.length - longestPartialPrefix);
						if (visible) {
							visibleOut += this.stripControlTokens(visible);
						}
						this._textToolParserBuffer = data.slice(data.length - longestPartialPrefix);
						data = "";
						break;
					} else {
						// All visible, clean other control tokens
						visibleOut += this.stripControlTokens(data);
						data = "";
						break;
					}
				}
				// Emit text before the token
				const pre = data.slice(0, b);
				if (pre) {
					visibleOut += this.stripControlTokens(pre);
				}
				// Advance past BEGIN
				data = data.slice(b + BEGIN.length);

				// Find the delimiter that ends the name/index segment
				const a = data.indexOf(ARG_BEGIN);
				const e = data.indexOf(END);
				let delimIdx = -1;
				let delimKind: "arg" | "end" | undefined = undefined;
				if (a !== -1 && (e === -1 || a < e)) {
					delimIdx = a;
					delimKind = "arg";
				} else if (e !== -1) {
					delimIdx = e;
					delimKind = "end";
				} else {
					// Incomplete header; keep for next chunk (re-add BEGIN so we don't lose it)
					this._textToolParserBuffer = BEGIN + data;
					data = "";
					break;
				}

				const header = data.slice(0, delimIdx).trim();
				const m = header.match(/^([A-Za-z0-9_\-.]+)(?::(\d+))?/);
				const name = m?.[1] ?? undefined;
				const index = m?.[2] ? Number(m?.[2]) : undefined;
				this._textToolActive = { name, index, argBuffer: "", emitted: false };
				// Advance past delimiter token
				if (delimKind === "arg") {
					data = data.slice(delimIdx + ARG_BEGIN.length);
				} else /* end */ {
					// No args, finalize immediately
					data = data.slice(delimIdx + END.length);
					const did = this.emitTextToolCallIfValid(progress, this._textToolActive, "{}");
					if (did) {
						this._textToolActive.emitted = true;
						emittedAny = true;
					}
					this._textToolActive = undefined;
				}
				continue;
			}

			// We are inside arguments, collect until END and emit as soon as JSON becomes valid
			const e2 = data.indexOf(END);
			if (e2 === -1) {
				// No end marker yet, accumulate and check for early valid JSON
				this._textToolActive.argBuffer += data;
				// Early emit when JSON becomes valid and we haven't emitted yet
				if (!this._textToolActive.emitted) {
					const did = this.emitTextToolCallIfValid(progress, this._textToolActive, this._textToolActive.argBuffer);
					if (did) {
						this._textToolActive.emitted = true;
						emittedAny = true;
					}
				}
				data = "";
				break;
			} else {
				this._textToolActive.argBuffer += data.slice(0, e2);
				// Consume END
				data = data.slice(e2 + END.length);
				// Final attempt to emit if not already
				if (!this._textToolActive.emitted) {
					const did = this.emitTextToolCallIfValid(progress, this._textToolActive, this._textToolActive.argBuffer);
					if (did) {
						emittedAny = true;
					}
				}
				this._textToolActive = undefined;
				continue;
			}
		}

		// Emit any visible text
		const textToEmit = visibleOut;
		if (textToEmit && textToEmit.length > 0) {
			progress.report(new vscode.LanguageModelTextPart(textToEmit));
			emittedText = true;
			emittedAny = true;
		}

		// Store leftover for next chunk
		this._textToolParserBuffer = data;

		return { emittedText, emittedAny };
	}

	private emitTextToolCallIfValid(
		progress: vscode.Progress<vscode.LanguageModelResponsePart>,
		call: { name?: string; index?: number; argBuffer: string; emitted?: boolean },
		argText: string
	): boolean {
		const name = call.name ?? "unknown_tool";
		const parsed = tryParseJSONObject(argText);
		if (!parsed.ok) {
			return false;
		}
		const canonical = JSON.stringify(parsed.value);
		const key = `${name}:${canonical}`;
		// identity-based dedupe when index is present
		if (typeof call.index === "number") {
			const idKey = `${name}:${call.index}`;
			if (this._emittedTextToolCallIds.has(idKey)) {
				return false;
			}
			// Mark identity as emitted
			this._emittedTextToolCallIds.add(idKey);
		} else if (this._emittedTextToolCallKeys.has(key)) {
			return false;
		}
		this._emittedTextToolCallKeys.add(key);
		const id = `tct_${Math.random().toString(36).slice(2, 10)}`;
		progress.report(new vscode.LanguageModelToolCallPart(id, name, parsed.value));
		return true;
	}

	private async flushActiveTextToolCall(progress: vscode.Progress<vscode.LanguageModelResponsePart>): Promise<void> {
		if (!this._textToolActive) {
			return;
		}
		const argText = this._textToolActive.argBuffer;
		const parsed = tryParseJSONObject(argText);
		if (!parsed.ok) {
			return;
		}
		// Emit (dedupe ensures we don't double-emit)
		this.emitTextToolCallIfValid(progress, this._textToolActive, argText);
		this._textToolActive = undefined;
	}

	/**
	 * Try to emit a buffered tool call when a valid name and JSON arguments are available.
	 * @param index The tool call index from the stream.
	 * @param progress Progress reporter for parts.
	 */
	private async tryEmitBufferedToolCall(
		index: number,
		progress: vscode.Progress<vscode.LanguageModelResponsePart>
	): Promise<void> {
		const buf = this._toolCallBuffers.get(index);
		if (!buf) {
			return;
		}
		if (!buf.name) {
			return;
		}
		const canParse = tryParseJSONObject(buf.args);
		if (!canParse.ok) {
			return;
		}
		const id = buf.id ?? `call_${Math.random().toString(36).slice(2, 10)}`;
		const parameters = canParse.value;
		try {
			const canonical = JSON.stringify(parameters);
			this._emittedTextToolCallKeys.add(`${buf.name}:${canonical}`);
		} catch {
			/* ignore */
		}
		progress.report(new vscode.LanguageModelToolCallPart(id, buf.name, parameters));
		this._toolCallBuffers.delete(index);
		this._completedToolCallIndices.add(index);
	}

	/**
	 * Flush all buffered tool calls, optionally throwing if arguments are not valid JSON.
	 * @param progress Progress reporter for parts.
	 * @param throwOnInvalid If true, throw when a tool call has invalid JSON args.
	 */
	private async flushToolCallBuffers(
		progress: vscode.Progress<vscode.LanguageModelResponsePart>,
		throwOnInvalid: boolean
	): Promise<void> {
		if (this._toolCallBuffers.size === 0) {
			return;
		}
		for (const [idx, buf] of Array.from(this._toolCallBuffers.entries())) {
			const parsed = tryParseJSONObject(buf.args);
			if (!parsed.ok) {
				if (throwOnInvalid) {
					console.error("[LiteLLM Model Provider] Invalid JSON for tool call", {
						idx,
						snippet: (buf.args || "").slice(0, 200),
					});
					throw new Error("Invalid JSON for tool call");
				}
				// When not throwing (e.g. on [DONE]), drop silently to reduce noise
				continue;
			}
			const id = buf.id ?? `call_${Math.random().toString(36).slice(2, 10)}`;
			const name = buf.name ?? "unknown_tool";
			try {
				const canonical = JSON.stringify(parsed.value);
				this._emittedTextToolCallKeys.add(`${name}:${canonical}`);
			} catch {
				/* ignore */
			}
			progress.report(new vscode.LanguageModelToolCallPart(id, name, parsed.value));
			this._toolCallBuffers.delete(idx);
			this._completedToolCallIndices.add(idx);
		}
	}

	/** Strip provider control tokens like <|tool_calls_section_begin|> and <|tool_call_begin|> from streamed text. */
	private stripControlTokens(text: string): string {
		try {
			// Remove section markers and explicit tool call begin/argument/end markers that some backends stream as text
			return text
				.replace(/<\|[a-zA-Z0-9_-]+_section_(?:begin|end)\|>/g, "")
				.replace(/<\|tool_call_(?:argument_)?(?:begin|end)\|>/g, "");
		} catch {
			return text;
		}
	}

	/**
	 * Transform a chat/completions request body to the responses API format.
	 * The responses API uses "input" (array format) instead of "messages".
	 * Tools use the SAME standard OpenAI format as chat/completions.
	 * @param requestBody The original chat/completions request body
	 * @returns Transformed request body for the responses endpoint
	 */
	private transformToResponsesFormat(requestBody: Record<string, unknown>): Record<string, unknown> {
		const messages = requestBody.messages as Record<string, unknown>[] | undefined;
		if (!messages || messages.length === 0) {
			throw new Error("Cannot transform empty messages to responses format");
		}

		// Log input message structure for diagnostics
		console.log("[LiteLLM Model Provider] transformToResponsesFormat input:", {
			messageCount: messages.length,
			messageRoles: messages.map((m) => m.role),
			messageContentLengths: messages.map((m) => {
				const content = m.content as unknown;
				if (typeof content === "string") {
					return `string:${content.length}`;
				}
				if (Array.isArray(content)) {
					return `array:${content.length}`;
				}
				return "other";
			}),
		});

		// Transform messages to the input array format for responses API
		// Extract system message separately as "instructions"
		const inputArray: Array<Record<string, unknown>> = [];
		let instructions: string | undefined;

		// Track which tool calls we've added so we can properly pair with outputs
		const addedToolCalls = new Set<string>();

		// First pass: collect all tool call IDs from assistant messages
		// so we know which ones we have before processing tool outputs
		const allToolCallIds = new Set<string>();
		const toolCallIdMap = new Map<string, string>(); // Map of original IDs to normalized IDs

		for (const msg of messages) {
			const role = msg.role as string;
			if (role === "assistant") {
				// Tool calls are in the tool_calls field, not in content
				const toolCalls = msg.tool_calls as Array<Record<string, unknown>> | undefined;
				if (toolCalls && Array.isArray(toolCalls)) {
					for (const tc of toolCalls) {
						const id = tc.id as string | undefined;
						if (id) {
							allToolCallIds.add(id);

							// Normalize the ID for matching purposes
							let normalizedId = id;
							if (!normalizedId.startsWith("fc_")) {
								normalizedId = `fc_${normalizedId}`;
							}
							toolCallIdMap.set(id, normalizedId);
							toolCallIdMap.set(normalizedId, normalizedId);

							console.log("[LiteLLM Model Provider] Found tool call ID from tool_calls field:", {
								original: id,
								normalized: normalizedId,
							});
						}
					}
				}

				// Also check content array for any tool_call parts (for compatibility)
				const content = msg.content as Array<Record<string, unknown>> | string | undefined;
				if (Array.isArray(content)) {
					for (const part of content) {
						if (part && typeof part === "object" && (part as Record<string, unknown>).type === "tool_call") {
							const callId = (part as Record<string, unknown>).callId as string;
							if (callId) {
								allToolCallIds.add(callId);

								// Normalize the ID
								let normalizedId = callId;
								if (!normalizedId.startsWith("fc_")) {
									normalizedId = `fc_${normalizedId}`;
								}
								toolCallIdMap.set(callId, normalizedId);
								toolCallIdMap.set(normalizedId, normalizedId);

								console.log("[LiteLLM Model Provider] Found tool call ID from content array:", {
									original: callId,
									normalized: normalizedId,
								});
							}
						}
					}
				}
			}
		}

		console.log("[LiteLLM Model Provider] Found tool call IDs in messages:", {
			count: allToolCallIds.size,
			ids: Array.from(allToolCallIds),
			normalizedMap: Array.from(toolCallIdMap.entries()),
		});

		// Second pass: transform messages to input format
		for (const msg of messages) {
			const role = msg.role as string;
			const content = msg.content as Array<Record<string, unknown>> | string | undefined;

			if (role === "system") {
				// System message becomes "instructions" parameter (NOT in input array)
				if (typeof content === "string") {
					instructions = content;
				}
			} else if (role === "user") {
				// User messages in input array
				if (typeof content === "string") {
					inputArray.push({
						type: "message",
						role: "user",
						content: [
							{
								type: "input_text",
								text: content,
							},
						],
					});
				}
			} else if (role === "assistant") {
				// Assistant messages can contain text and/or tool calls
				const assistantContent: Array<Record<string, unknown>> = [];
				let hasToolCalls = false;

				// Handle tool_calls field (from OpenAI format)
				const toolCallsField = msg.tool_calls as Array<Record<string, unknown>> | undefined;
				if (toolCallsField && Array.isArray(toolCallsField)) {
					for (const toolCall of toolCallsField) {
						hasToolCalls = true;
						const toolCallId = toolCall.id as string;
						const toolFunction = toolCall.function as Record<string, unknown> | undefined;
						const toolName = toolFunction?.name as string | undefined;
						const toolArgs = toolFunction?.arguments as unknown;

						// Use normalized ID from the map, or generate normalized version
						let normalizedId = toolCallIdMap.get(toolCallId);
						if (!normalizedId) {
							// If not in map, normalize it now
							if (!toolCallId.startsWith("fc_")) {
								normalizedId = `fc_${toolCallId}`;
							} else {
								normalizedId = toolCallId;
							}
							toolCallIdMap.set(toolCallId, normalizedId);
						}

						console.log("[LiteLLM Model Provider] Adding function_call from tool_calls field:", {
							originalId: toolCallId,
							normalizedId: normalizedId,
							name: toolName,
							argumentsPreview:
								typeof toolArgs === "string" ? toolArgs.slice(0, 50) : JSON.stringify(toolArgs || {}).slice(0, 50),
						});

						// Add tool call in responses API format with normalized ID
						const functionCall = {
							type: "function_call",
							id: normalizedId,
							call_id: normalizedId,
							name: toolName,
							arguments: typeof toolArgs === "string" ? toolArgs : JSON.stringify(toolArgs || {}),
						};
						inputArray.push(functionCall);
						addedToolCalls.add(normalizedId);
					}
				}

				if (Array.isArray(content)) {
					// Content is an array of parts (text, tool calls, etc)
					for (const part of content) {
						if (!part || typeof part !== "object") {
							continue;
						}

						const partType = (part as Record<string, unknown>).type as string;

						if (partType === "text") {
							// Text part
							assistantContent.push({
								type: "output_text",
								text: (part as Record<string, unknown>).value,
							});
						} else if (partType === "tool_call") {
							// Tool call part - need to convert to responses API format
							hasToolCalls = true;
							const toolCallId = (part as Record<string, unknown>).callId as string;
							const toolName = (part as Record<string, unknown>).name as string;
							const toolArgs = (part as Record<string, unknown>).arguments as unknown;

							// Use normalized ID from the map, or generate normalized version
							let normalizedId = toolCallIdMap.get(toolCallId);
							if (!normalizedId) {
								// If not in map, normalize it now
								if (!toolCallId.startsWith("fc_")) {
									normalizedId = `fc_${toolCallId}`;
								} else {
									normalizedId = toolCallId;
								}
								toolCallIdMap.set(toolCallId, normalizedId);
							}

							// Add tool call in responses API format with normalized ID
							const functionCall = {
								type: "function_call",
								id: normalizedId,
								call_id: normalizedId,
								name: toolName,
								arguments: typeof toolArgs === "string" ? toolArgs : JSON.stringify(toolArgs),
							};
							inputArray.push(functionCall);
							addedToolCalls.add(normalizedId);

							console.log("[LiteLLM Model Provider] Added function_call from content array:", {
								originalId: toolCallId,
								normalizedId: normalizedId,
								name: toolName,
								argumentsPreview:
									typeof toolArgs === "string" ? toolArgs.slice(0, 50) : JSON.stringify(toolArgs).slice(0, 50),
							});
						}
					}
				} else if (typeof content === "string") {
					// Simple string content
					assistantContent.push({
						type: "output_text",
						text: content,
					});
				}

				// Only add message if it has content (not just tool calls)
				if (assistantContent.length > 0) {
					inputArray.push({
						type: "message",
						role: "assistant",
						content: assistantContent,
					});
				} else if (hasToolCalls) {
					// If only tool calls and no text, still add a message for context
					inputArray.push({
						type: "message",
						role: "assistant",
						content: [
							{
								type: "output_text",
								text: "",
							},
						],
					});
				}
			} else if (role === "tool") {
				// Tool result messages
				// The Responses API REQUIRES that a function_call exists before its output
				const toolCallId = msg.tool_call_id as string | undefined;
				const toolContent = typeof content === "string" ? content : JSON.stringify(content);

				if (toolCallId) {
					// Look up the normalized ID from the map
					let matchingId = toolCallIdMap.get(toolCallId);

					// If not in map, try to match by normalizing
					if (!matchingId) {
						if (!toolCallId.startsWith("fc_")) {
							matchingId = `fc_${toolCallId}`;
						} else {
							matchingId = toolCallId;
						}
					}

					const matchFound = addedToolCalls.has(matchingId);

					console.log("[LiteLLM Model Provider] Processing tool result:", {
						originalCallId: toolCallId,
						normalizedMatchingId: matchingId,
						matchFound,
						inAddedToolCalls: matchFound,
						addedToolCallsSize: addedToolCalls.size,
						allToolCallIdsSize: allToolCallIds.size,
						outputPreview:
							typeof toolContent === "string" ? toolContent.slice(0, 100) : String(toolContent).slice(0, 100),
						addedToolCallsList: Array.from(addedToolCalls),
					});

					// CRITICAL: Only add function_call_output if we have the matching function_call
					// If the tool call is from a previous response, we skip it to avoid the API error
					if (addedToolCalls.has(matchingId)) {
						// We have the matching function_call in this request
						inputArray.push({
							type: "function_call_output",
							call_id: matchingId,
							output: toolContent,
						});

						console.log("[LiteLLM Model Provider] Added function_call_output to input:", {
							callId: toolCallId,
							matchingId: matchingId,
							outputPreview: typeof toolContent === "string" ? toolContent.slice(0, 50) : toolContent,
						});
					} else if (allToolCallIds.has(matchingId)) {
						// Tool call from a previous response - skip the output
						// The Responses API doesn't have the context of the previous call
						console.log(
							"[LiteLLM Model Provider] Skipping function_call_output for tool call from previous response:",
							matchingId
						);
					} else {
						// Tool call ID not found anywhere
						console.warn(
							"[LiteLLM Model Provider] Tool result for completely unknown tool call - skipping:",
							toolCallId
						);
					}
				}
			}
		}

		// Build the responses format request body
		const responsesBody: Record<string, unknown> = {
			model: requestBody.model,
			// Always send an array to keep the shape consistent for callers/tests.
			input: inputArray,
			stream: requestBody.stream,
		};

		// Add instructions if we extracted a system message
		if (instructions) {
			responsesBody.instructions = instructions;
		}

		// Map max_tokens to max_tokens in responses format (same parameter name)
		if (typeof requestBody.max_tokens === "number") {
			responsesBody.max_tokens = requestBody.max_tokens;
		}

		// Add temperature if present
		if (typeof requestBody.temperature === "number") {
			responsesBody.temperature = requestBody.temperature;
		}

		// Add top_p if present
		if (typeof requestBody.top_p === "number") {
			responsesBody.top_p = requestBody.top_p;
		}

		// Add frequency_penalty if present
		if (typeof requestBody.frequency_penalty === "number") {
			responsesBody.frequency_penalty = requestBody.frequency_penalty;
		}

		// Add presence_penalty if present
		if (typeof requestBody.presence_penalty === "number") {
			responsesBody.presence_penalty = requestBody.presence_penalty;
		}

		// Add stop sequences if present
		if (requestBody.stop) {
			responsesBody.stop = requestBody.stop;
		}

		// Add tools if present
		// LiteLLM's /responses endpoint expects tools in a flattened format with top-level name field
		// Original format (OpenAI): { type: "function", function: { name, description, parameters } }
		// LiteLLM responses format: { type: "function", name, description, parameters }
		if (requestBody.tools) {
			const tools = requestBody.tools as Array<Record<string, unknown>>;

			// Transform and validate tools for LiteLLM's responses endpoint
			const transformedTools = tools
				.map((tool) => {
					if (!tool || typeof tool !== "object") {
						console.warn("[LiteLLM Model Provider] Skipping non-object tool");
						return null;
					}

					// Extract function details from nested structure
					const func = tool.function as Record<string, unknown> | undefined;
					if (!func) {
						console.warn("[LiteLLM Model Provider] Tool missing function field:", tool);
						return null;
					}

					const name = func.name as string | undefined;
					const description = func.description as string | undefined;
					const parameters = func.parameters as object | undefined;

					// Validate required fields
					if (!name || typeof name !== "string" || name.length === 0) {
						console.warn("[LiteLLM Model Provider] Tool missing valid name:", { tool: JSON.stringify(tool) });
						return null;
					}

					if (!description || typeof description !== "string") {
						console.warn("[LiteLLM Model Provider] Tool missing valid description:", { name });
						return null;
					}

					if (!parameters || typeof parameters !== "object") {
						console.warn("[LiteLLM Model Provider] Tool missing valid parameters:", { name });
						return null;
					}

					// Return flattened tool structure for LiteLLM's responses endpoint
					return {
						type: "function",
						name: name,
						description: description,
						parameters: parameters,
					};
				})
				.filter(
					(tool): tool is { type: string; name: string; description: string; parameters: object } => tool !== null
				);

			if (transformedTools.length > 0) {
				responsesBody.tools = transformedTools;
				const firstTool = transformedTools[0] as Record<string, unknown> | undefined;
				console.log("[LiteLLM Model Provider] Transformed tools for responses endpoint:", {
					toolCount: transformedTools.length,
					toolNames: transformedTools.map((t) => (t as Record<string, unknown>).name),
					sampleTool: firstTool ? JSON.stringify(firstTool, null, 2) : "no tools",
				});
			} else if (tools.length > 0) {
				console.warn("[LiteLLM Model Provider] No valid tools to include in responses request", {
					totalTools: tools.length,
					invalidReasons: "Missing type, name, description, or parameters",
				});
			}
		}

		// Add tool_choice if present and if we have tools
		// The /responses endpoint uses the same tool_choice format as /chat/completions
		if (requestBody.tool_choice && responsesBody.tools) {
			responsesBody.tool_choice = requestBody.tool_choice;
			console.log("[LiteLLM Model Provider] Added tool_choice:", requestBody.tool_choice);
		}

		// Log the final input array structure for debugging tool call sequences
		if (Array.isArray(responsesBody.input)) {
			const inputArray = responsesBody.input as Array<Record<string, unknown>>;
			const inputSizeEstimate = JSON.stringify(responsesBody.input).length;

			console.log("[LiteLLM Model Provider] Input array structure:", {
				length: inputArray.length,
				estimatedSizeBytes: inputSizeEstimate,
				estimatedSizeMB: (inputSizeEstimate / 1024 / 1024).toFixed(2),
				types: inputArray.map((item) => item.type),
				callIds: inputArray
					.filter((item) => item.type === "function_call" || item.type === "function_call_output")
					.map((item) => item.call_id || item.id),
			});

			// Warn if input is getting very large (>5MB)
			if (inputSizeEstimate > 5 * 1024 * 1024) {
				console.warn("[LiteLLM Model Provider] WARNING: Input array is very large!", {
					sizeBytes: inputSizeEstimate,
					sizeMB: (inputSizeEstimate / 1024 / 1024).toFixed(2),
					itemCount: inputArray.length,
					recommendation: "Consider reducing message history or message content size",
				});
			}
		}

		console.log("[LiteLLM Model Provider] Transformed request to responses format", {
			originalMessageCount: (requestBody.messages as Array<unknown>)?.length ?? 0,
			transformedInputLength: Array.isArray(responsesBody.input)
				? (responsesBody.input as Array<unknown>).length
				: "string",
			hasTools: !!responsesBody.tools,
			inputType: typeof responsesBody.input,
			hasInstructions: !!responsesBody.instructions,
		});

		return responsesBody;
	}
}
