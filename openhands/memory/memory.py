import threading

from openai._exceptions import APIConnectionError, InternalServerError, RateLimitError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

from openhands.core.config import LLMConfig
from openhands.core.logger import openhands_logger as logger
from openhands.core.utils import json

try:
    import chromadb
    import llama_index.embeddings.openai.base as llama_openai
    from llama_index.core import Document, VectorStoreIndex
    from llama_index.core.retrievers import VectorIndexRetriever
    from llama_index.vector_stores.chroma import ChromaVectorStore

    LLAMA_INDEX_AVAILABLE = True
except ImportError:
    LLAMA_INDEX_AVAILABLE = False

if LLAMA_INDEX_AVAILABLE:
    # TODO: this could be made configurable
    num_retries: int = 10
    retry_min_wait: int = 3
    retry_max_wait: int = 300

    # llama-index includes a retry decorator around openai.get_embeddings() function
    # it is initialized with hard-coded values and errors
    # this non-customizable behavior is creating issues when it's retrying faster than providers' rate limits
    # this block attempts to banish it and replace it with our decorator, to allow users to set their own limits

    if hasattr(llama_openai.get_embeddings, '__wrapped__'):
        original_get_embeddings = llama_openai.get_embeddings.__wrapped__
    else:
        logger.warning('Cannot set custom retry limits.')
        num_retries = 1
        original_get_embeddings = llama_openai.get_embeddings

    def attempt_on_error(retry_state):
        logger.error(
            f'{retry_state.outcome.exception()}. Attempt #{retry_state.attempt_number} | You can customize retry values in the configuration.',
            exc_info=False,
        )
        return None

    @retry(
        reraise=True,
        stop=stop_after_attempt(num_retries),
        wait=wait_random_exponential(min=retry_min_wait, max=retry_max_wait),
        retry=retry_if_exception_type(
            (RateLimitError, APIConnectionError, InternalServerError)
        ),
        after=attempt_on_error,
    )
    def wrapper_get_embeddings(*args, **kwargs):
        return original_get_embeddings(*args, **kwargs)

    llama_openai.get_embeddings = wrapper_get_embeddings

    class EmbeddingsLoader:
        """Loader for embedding model initialization."""

        @staticmethod
        def get_embedding_model(strategy: str, llm_config: LLMConfig):
            supported_ollama_embed_models = [
                'llama2',
                'mxbai-embed-large',
                'nomic-embed-text',
                'all-minilm',
                'stable-code',
                'bge-m3',
                'bge-large',
                'paraphrase-multilingual',
                'snowflake-arctic-embed',
            ]
            if strategy in supported_ollama_embed_models:
                from llama_index.embeddings.ollama import OllamaEmbedding

                return OllamaEmbedding(
                    model_name=strategy,
                    base_url=llm_config.embedding_base_url,
                    ollama_additional_kwargs={'mirostat': 0},
                )
            elif strategy == 'openai':
                from llama_index.embeddings.openai import OpenAIEmbedding

                return OpenAIEmbedding(
                    model='text-embedding-ada-002',
                    api_key=llm_config.api_key,
                )
            elif strategy == 'azureopenai':
                from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding

                return AzureOpenAIEmbedding(
                    model='text-embedding-ada-002',
                    deployment_name=llm_config.embedding_deployment_name,
                    api_key=llm_config.api_key,
                    azure_endpoint=llm_config.base_url,
                    api_version=llm_config.api_version,
                )
            elif (strategy is not None) and (strategy.lower() == 'none'):
                # TODO: this works but is not elegant enough. The incentive is when
                # an agent using embeddings is not used, there is no reason we need to
                # initialize an embedding model
                return None
            else:
                from llama_index.embeddings.huggingface import HuggingFaceEmbedding

                return HuggingFaceEmbedding(model_name='BAAI/bge-small-en-v1.5')


class LongTermMemory:
    """Handles storing information for the agent to access later, using chromadb."""

    def __init__(self, llm_config: LLMConfig, memory_max_threads: int = 1):
        """Initialize the chromadb and set up ChromaVectorStore for later use."""
        if not LLAMA_INDEX_AVAILABLE:
            raise ImportError(
                'llama_index and its dependencies are not installed. '
                'To use LongTermMemory, please run: poetry install --with llama-index'
            )

        db = chromadb.Client(chromadb.Settings(anonymized_telemetry=False))
        self.collection = db.get_or_create_collection(name='memories')
        vector_store = ChromaVectorStore(chroma_collection=self.collection)
        embedding_strategy = llm_config.embedding_model
        embed_model = EmbeddingsLoader.get_embedding_model(
            embedding_strategy, llm_config
        )
        self.index = VectorStoreIndex.from_vector_store(vector_store, embed_model)
        self.sema = threading.Semaphore(value=memory_max_threads)
        self.thought_idx = 0
        self._add_threads: list[threading.Thread] = []

    def add_event(self, event: dict):
        """Adds a new event to the long term memory with a unique id.

        Parameters:
        - event (dict): The new event to be added to memory
        """
        id = ''
        t = ''
        if 'action' in event:
            t = 'action'
            id = event['action']
        elif 'observation' in event:
            t = 'observation'
            id = event['observation']
        doc = Document(
            text=json.dumps(event),
            doc_id=str(self.thought_idx),
            extra_info={
                'type': t,
                'id': id,
                'idx': self.thought_idx,
            },
        )
        self.thought_idx += 1
        logger.debug(f'Adding {t} event to memory: {self.thought_idx}')
        thread = threading.Thread(target=self._add_doc, args=(doc,))
        self._add_threads.append(thread)
        thread.start()  # We add the doc concurrently so we don't have to wait ~500ms for the insert

    def _add_doc(self, doc):
        with self.sema:
            self.index.insert(doc)

    def search(self, query: str, k: int = 10):
        """Searches through the current memory using VectorIndexRetriever

        Parameters:
        - query (str): A query to match search results to
        - k (int): Number of top results to return

        Returns:
        - list[str]: list of top k results found in current memory
        """
        retriever = VectorIndexRetriever(
            index=self.index,
            similarity_top_k=k,
        )
        results = retriever.retrieve(query)
        return [r.get_text() for r in results]
