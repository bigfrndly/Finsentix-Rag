import os
import streamlit as st
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from ingestion.document_processor import load_documents, chunk_documents
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.re_ranker import ReRanker
from generation.llm_generator import LLMGenerator
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

st.set_page_config(page_title="FinSentix RAG", page_icon="📈", layout="wide")

st.title("📈 FinSentix - RAG Dashboard")
st.write("Upload financial documents, process them, and ask questions about their content!")

DOCS_DIR = "data/docs"
DB_DIR = "data/db"
os.makedirs(DOCS_DIR, exist_ok=True)

@st.cache_resource
def get_pipeline_components():
    documents = load_documents(DOCS_DIR)
    if not documents:
        return None, None, None, None
    chunks = chunk_documents(documents)
    
    # Vector DB
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = Chroma.from_documents(chunks, embeddings, persist_directory=DB_DIR)
    
    # Pipeline components
    retriever = HybridRetriever(vector_store, chunks)
    reranker = ReRanker()
    generator = LLMGenerator()
    return retriever, reranker, generator, chunks

# --- Sidebar for uploading ---
with st.sidebar:
    st.header("Document Upload")
    uploaded_files = st.file_uploader("Upload PDF documents", type=["pdf"], accept_multiple_files=True)
    
    if st.button("Process Uploaded Documents"):
        if uploaded_files:
            for uploaded_file in uploaded_files:
                file_path = os.path.join(DOCS_DIR, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
            st.success(f"Saved {len(uploaded_files)} files!")
            
            # Clear cache so new documents are loaded into Chroma
            st.cache_resource.clear()
            st.rerun()
        else:
            st.warning("Please select at least one PDF file.")

# --- Main App Logic ---
with st.spinner("Loading AI Engine & Documents..."):
    retriever, reranker, generator, chunks = get_pipeline_components()

if retriever:
    st.header("Ask a Question")
    query = st.text_input("Enter your question based on the uploaded documents:")
    
    if st.button("Generate Answer"):
        if query:
            with st.spinner("Retrieving and Analyzing..."):
                retrieved_docs = retriever.retrieve(query, top_k=10)
                best_docs = reranker.rank(query, retrieved_docs, top_n=3)
                answer = generator.generate(query, best_docs)
                
                st.subheader("Answer")
                st.markdown(answer)
                
                with st.expander("View Retrieved Sources"):
                    for i, doc in enumerate(best_docs):
                        st.markdown(f"**[Doc {i+1}] Source:** `{doc.metadata.get('source', 'Unknown')}` | **Page:** `{doc.metadata.get('page', 'N/A')}`")
                        st.info(doc.page_content)
        else:
            st.warning("Please enter a question.")
else:
    st.info("No documents found in the database. Please upload a PDF from the sidebar to get started!")
