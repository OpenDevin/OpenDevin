from opendevin.core.logger import opendevin_logger as logger
from opendevin.llm.llm import LLM


class MemoryCondenser:
    async def condense(self, summarize_prompt: str, llm: LLM):
        """Attempts to condense the memory by using the llm

        Parameters:
        - llm (LLM): llm to be used for summarization

        Raises:
        - Exception: the same exception as it got from the llm or processing the response
        """
        try:
            messages = [{'content': summarize_prompt, 'role': 'user'}]
            resp = await llm.async_completion(messages=messages)
            summary_response = resp['choices'][0]['message']['content']
            return summary_response
        except Exception as e:
            logger.error('Error condensing thoughts: %s', str(e), exc_info=False)

            # TODO If the llm fails with ContextWindowExceededError, we can try to condense the memory chunk by chunk
            raise
