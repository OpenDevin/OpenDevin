# 🤖 LLM Backends

OpenHands can connect to any LLM supported by LiteLLM. However, it requires a powerful model to work.

## Model Recommendations

Based on a recent evaluation of language models for coding tasks (using the SWE-bench dataset), we can provide some recommendations for model selection. The full analysis can be found in [this blog article](https://www.all-hands.dev/blog/evaluation-of-llms-as-coding-agents-on-swe-bench-at-30x-speed).

When choosing a model, consider both the quality of outputs and the associated costs. Here's a summary of the findings:

1. GPT-4 models generally provide the highest quality results but at a higher cost.
2. Claude models offer a good balance between quality and cost.
3. Gemini models show promising results and may be cost-effective for certain tasks.
4. Open-source models like Mixtral or CodeLlama can be viable options for those with budget constraints or privacy concerns, though they may have lower performance compared to proprietary models.

For the most up-to-date and detailed information on model performance and pricing, please refer to the [full article](https://www.all-hands.dev/blog/evaluation-of-llms-as-coding-agents-on-swe-bench-at-30x-speed).

Based on these findings and community feedback, the following models have been verified to work well with OpenHands:

* claude-3-5-sonnet (recommended)
* gemini-1.5-pro / gemini-1.5-flash
* gpt-4 / gpt-4o
* llama-3.1-405b / hermes-3-llama-3.1-405b

:::warning
OpenHands will issue many prompts to the LLM you configure. Most of these LLMs cost money, so be sure to set spending
limits and monitor usage.
:::

If you have successfully run OpenHands with specific LLMs not in the list, please add them to the verified list. We
also encourage you to open a PR to share your setup process to help others using the same provider and LLM!

For a full list of the providers and models available, please consult the
[litellm documentation](https://docs.litellm.ai/docs/providers).

:::note
Most current local and open source models are not as powerful. When using such models, you may see long
wait times between messages, poor responses, or errors about malformed JSON. OpenHands can only be as powerful as the
models driving it. However, if you do find ones that work, please add them to the verified list above.
:::

## LLM Configuration

The following can be set in the OpenHands UI through the Settings:
* `LLM Provider`
* `LLM Model`
* `API Key`
* `Base URL` (through `Advanced Settings`)

There are some settings that may be necessary for some LLMs/providers that cannot be set through the UI. Instead, these
can be set through environment variables passed to the [docker run command](/modules/usage/installation)
using `-e`:

* `LLM_API_VERSION`
* `LLM_EMBEDDING_MODEL`
* `LLM_EMBEDDING_DEPLOYMENT_NAME`
* `LLM_DROP_PARAMS`
* `LLM_DISABLE_VISION`
* `LLM_CACHING_PROMPT`

We have a few guides for running OpenHands with specific model providers:

* [Azure](llms/azure-llms)
* [Google](llms/google-llms)
* [Groq](llms/groq)
* [OpenAI](llms/openai-llms)
* [OpenRouter](llms/openrouter)

### API retries and rate limits

LLM providers typically have rate limits, sometimes very low, and may require retries. OpenHands will automatically retry requests if it receives a Rate Limit Error (429 error code), API connection error, or other transient errors.

You can customize these options as you need for the provider you're using. Check their documentation, and set the following environment variables to control the number of retries and the time between retries:

* `LLM_NUM_RETRIES` (Default of 8)
* `LLM_RETRY_MIN_WAIT` (Default of 15 seconds)
* `LLM_RETRY_MAX_WAIT` (Default of 120 seconds)
* `LLM_RETRY_MULTIPLIER` (Default of 2)

If you are running OpenHands in development mode, you can also set these options in the `config.toml` file:

```toml
[llm]
num_retries = 8
retry_min_wait = 15
retry_max_wait = 120
retry_multiplier = 2
```
