from typing import List, Dict, Any
from rank_bm25 import BM25Okapi
from langchain_core.documents import Document

class HybridRetriever:
    """
    Combines dense vector retrieval and sparse BM25 retrieval for optimal recall.
    """
    def __init__(self, vector_store, bm25_corpus: List[Document]):
        self.vector_store = vector_store
        self.documents = bm25_corpus
        
        # Initialize BM25
        tokenized_corpus = [doc.page_content.split(" ") for doc in self.documents]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def retrieve(self, query: str, top_k: int = 20) -> List[Document]:
        """
        Retrieves top_k documents from both dense and sparse, combines and deduplicates them.
        """
        # 1. Sparse Retrieval (BM25)
        tokenized_query = query.split(" ")
        bm25_scores = self.bm25.get_scores(tokenized_query)
        bm25_top_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:top_k]
        bm25_results = [self.documents[i] for i in bm25_top_indices]
        
        # 2. Dense Retrieval (Vector DB)
        # Using similarity search (Assuming langchain vector_store interface)
        if self.vector_store:
            dense_results = self.vector_store.similarity_search(query, k=top_k)
        else:
            dense_results = []
            
        # 3. Combine & Deduplicate
        combined_results = []
        seen_ids = set()
        
        # Pool results together
        for doc in bm25_results + dense_results:
            doc_id = doc.metadata.get("chunk_id", str(id(doc)))
            if doc_id not in seen_ids:
                seen_ids.add(doc_id)
                combined_results.append(doc)
                
        return combined_results
