import os
from dotenv import load_dotenv

# Load environment variables (like OPENAI_API_KEY)
load_dotenv()

from ingestion.document_processor import load_documents, chunk_documents
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.re_ranker import ReRanker
from generation.llm_generator import LLMGenerator
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

def main():
    print("Welcome to the Production RAG Application: 'Ask My Docs'")
    print("-------------------------------------------------------")
    
    docs_dir = "data/docs"
    db_dir = "data/db"
    
    # 1. Ingestion
    print("Initializing pipeline...")
    documents = load_documents(docs_dir)
    if not documents:
        print(f"No documents found in {docs_dir}. Please add some PDFs and restart.")
        return
        
    chunks = chunk_documents(documents)
    
    # 2. Vector DB Setup (using Chroma)
    print("Setting up Vector Database...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = Chroma.from_documents(chunks, embeddings, persist_directory=db_dir)
    
    # 3. Initialize Retriever and Re-ranker
    print("Setting up Hybrid Retriever and Re-Ranker...")
    retriever = HybridRetriever(vector_store, chunks)
    reranker = ReRanker()
    
    # 4. Initialize Generator
    print("Setting up LLM Generator...")
    generator = LLMGenerator()
    
    print("\nPipeline initialized. Ready to process and answer queries.")
    print("Type 'exit' or 'quit' to stop.\n")
    
    # Interactive Loop
    while True:
        try:
            query = input("\nAsk a question about your documents: ")
            if query.lower() in ['exit', 'quit']:
                break
                
            print("\nRetrieving documents...")
            retrieved_docs = retriever.retrieve(query, top_k=10)
            
            print("Re-ranking documents...")
            best_docs = reranker.rank(query, retrieved_docs, top_n=3)
            
            print("Generating answer...\n")
            answer = generator.generate(query, best_docs)
            
            print("="*50)
            print("ANSWER:")
            print(answer)
            print("="*50)
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
