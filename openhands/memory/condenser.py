from openhands.core.logger import openhands_logger as logger
from openhands.llm.llm import LLM


class MemoryCondenser:
    def condense(self, summarize_prompt: str, llm: LLM):
        """Attempts to condense the memory by using the llm

        Parameters:
        - llm (LLM): llm to be used for summarization

        Raises:
        - Exception: the same exception as it got from the llm or processing the response
        """
        try:
            messages = [{'content': summarize_prompt, 'role': 'user'}]
            resp = llm.completion(messages=messages)
            summary_response = resp['choices'][0]['message']['content']
            return summary_response
        except Exception as e:
            logger.error(f'Error condensing thoughts: {e}', exc_info=False)

            # TODO If the llm fails with ContextWindowExceededError, we can try to condense the memory chunk by chunk
            raise
