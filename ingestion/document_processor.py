import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_documents(docs_dir: str):
    """Loads PDFs from the specified directory."""
    print(f"Loading documents from {docs_dir}...")
    loader = PyPDFDirectoryLoader(docs_dir)
    documents = loader.load()
    print(f"Loaded {len(documents)} document pagess.")
    return documents

def chunk_documents(documents, chunk_size=512, chunk_overlap=50):
    """Splits documents into coherent chunks with overlap."""
    print(f"Chunking documents (size={chunk_size}, overlap={chunk_overlap})...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)
    
    # Assign a unique ID to each chunk for metadata citing and deduplication
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = f"chunk_{i}"
        
    print(f"Created {len(chunks)} chunks.")
    return chunks

if __name__ == "__main__":
    # Test the processor
    docs_dir = "data/docs"
    # Ensure dir exists
    import os
    os.makedirs(docs_dir, exist_ok=True)
    docs = load_documents(docs_dir)
    chunks = chunk_documents(docs)
