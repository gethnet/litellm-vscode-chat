## Problem Statement

LiteLLM connector is passing `temperature: 0.7` to an OpenAI model that only accepts the default temperature value of `1`. The OpenAI API rejects the request with a 400 Bad Request error, causing task failure.

This suggests:
- Model configuration does not validate OpenAI-specific parameter constraints before API call
- Model parameter constraints are not documented or enforced client-side
- User receives cryptic error instead of graceful fallback or validation

---

## Error Details

```
LiteLLM API error: 400 Bad Request
litellm.BadRequestError: OpenAIException -
Unsupported value: 'temperature' does not support 0.7 with this model.
Only the default (1) value is supported.
```

---

## Symptoms

- Task fails with 400 Bad Request when temperature parameter is set
- Error occurs when using specific OpenAI model(s)
- No client-side validation prevents invalid parameter combinations
- Wasted API quota on rejected requests

---

## Impact
- Tasks fail unexpectedly with cryptic errors
- Poor UX for users unfamiliar with OpenAI model constraints
- Difficult to debug (error doesn't specify model name clearly)
- No graceful fallback mechanism

---

## Acceptance Criteria

- [ ] Identify which OpenAI model(s) reject temperature parameter
- [ ] Document temperature constraints for affected OpenAI model(s)
- [ ] Add client-side parameter validation before OpenAI API call
- [ ] Implement graceful fallback (use default temperature or skip parameter)
- [ ] Improve error messaging to specify model name & supported parameter range
- [ ] Add configuration option to override parameter constraints (if intentional)
- [ ] Unit tests for OpenAI parameter validation
- [ ] Integration test with OpenAI API confirming valid requests succeed
- [ ] Update connector docs with OpenAI model-specific parameter constraints

---

## Investigation Steps

**For initial triage:**
1. [ ] Identify which OpenAI model was being used when error occurred
2. [ ] Check OpenAI API docs for temperature constraints on this model
3. [ ] Verify LiteLLM's OpenAI model configuration matches API reality
4. [ ] Determine if temperature should be removed or set to default (1)
5. [ ] Check if other OpenAI models have similar constraints
6. [ ] Review parameter validation logic for OpenAI provider in connector

---

## Suspected Root Causes

- [ ] OpenAI model configuration missing temperature constraints
- [ ] Parameter validation skipped or incomplete for OpenAI provider
- [ ] Hardcoded temperature value (0.7) incompatible with OpenAI model
- [ ] LiteLLM's OpenAI model definition out of sync with API behavior
- [ ] No parameter override mechanism for OpenAI model-specific quirks

---

## Reproduction Steps

```gherkin
Given a task configured with temperature: 0.7
And an OpenAI model that only supports temperature: 1
When the connector sends the request to OpenAI API
Then the request should succeed (use default or skip temperature)
But currently: 400 Bad Request error from OpenAI
```

---

## Solution Options (for discussion)

| Option | Tradeoff |
|--------|----------|
| **Skip unsupported params** | Graceful, but user loses control |
| **Use model default** | Safe, but may change model behavior |
| **Validate & fail early** | Clear error, but blocks user |
| **Config override** | Flexible, but complex configuration |
| **Warn & fallback** | Best UX, requires logging |

---

## Related Links
- [OpenAI API Parameter Docs](https://platform.openai.com/docs/api-reference)
- [LiteLLM OpenAI Provider](https://docs.litellm.ai/docs/providers/openai)
