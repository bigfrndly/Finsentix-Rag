from sentence_transformers import CrossEncoder

class ReRanker:
    """
    Re-Ranks a pool of retrieved documents using a Cross-Encoder.
    A Cross-Encoder processes the query and the document simultaneously, yielding a highly accurate relevance score.
    """
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        print(f"Initializing CrossEncoder model: {model_name}...")
        self.encoder = CrossEncoder(model_name)
        
    def rank(self, query: str, documents: list, top_n: int = 5):
        """
        Takes a query and a list of candidate documents, scores them, and returns the top_n.
        """
        if not documents:
            return []
            
        # CrossEncoder expects pairs of [query, text]
        pairs = [[query, doc.page_content] for doc in documents]
        
        # Predict scores
        scores = self.encoder.predict(pairs)
        
        # Attach scores to documents for sorting
        scored_docs = list(zip(documents, scores))
        
        # Sort by score in descending order
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        # Return only the documents, up to top_n
        top_docs = [doc for doc, score in scored_docs[:top_n]]
        return top_docs
