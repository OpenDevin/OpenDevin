import chromadb
from llama_index.core import Document, VectorStoreIndex
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.vector_stores.chroma import ChromaVectorStore

from . import json


class LongTermMemory:
    def __init__(self):
        db = chromadb.Client()
        self.collection = db.get_or_create_collection(name="memories")
        vector_store = ChromaVectorStore(chroma_collection=self.collection)
        self.index = VectorStoreIndex.from_vector_store(vector_store)
        self.thought_idx = 0

    def add_event(self, event):
        doc = Document(
            text=json.dumps(event),
            doc_id=str(self.thought_idx),
            extra_info={
                "type": event["action"],
                "idx": self.thought_idx,
            },
        )
        self.thought_idx += 1
        self.index.insert(doc)

    def search(self, query, k=10):
        retriever = VectorIndexRetriever(
            index=self.index,
            similarity_top_k=k,
        )
        results = retriever.retrieve(query)
        return [r.get_text() for r in results]
