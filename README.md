# 📈 FinSentix: Production-Grade RAG Dashboard

![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![LangChain](https://img.shields.io/badge/LangChain-Enabled-green)
![Gemini](https://img.shields.io/badge/Google-Gemini%202.5-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![ChromaDB](https://img.shields.io/badge/ChromaDB-VectorStore-blueviolet)

**FinSentix** is a high-performance, Retrieval-Augmented Generation (RAG) system built to parse, index, and intuitively question financial documents and PDFs. Engineered for precision and minimal hallucination, this application uses a multi-stage retrieval architecture featuring semantic search, cross-encoder re-ranking, and strict source grounding.

---

## 🌟 Key Features

*   **Interactive Web UI:** Drag-and-drop document upload and conversational querying via **Streamlit**.
*   **Advanced Retrieval Pipeline:** Hybrid document retrieval that ensures high context relevance.
*   **Cross-Encoder Re-Ranking:** Uses `ms-marco-MiniLM-L-6-v2` to mathematically re-rank retrieved documents against the user's query, significantly reducing noise.
*   **Grounded Generation:** Enforces strict prompt-engineering constraints using **Google Gemini 2.5 Flash**, returning answers *only* with direct inline citations to source documents.
*   **Automated Evaluation:** Includes an evaluation suite using the `ragas` framework for CI/CD gating based on Answer Relevancy, Faithfulness, and Context Precision.

## 🏗️ Architecture Stack

1.  **Frontend:** Streamlit
2.  **Orchestration:** LangChain
3.  **Embeddings:** HuggingFace (`all-MiniLM-L6-v2`)
4.  **Vector Store:** ChromaDB (Local, Persistent)
5.  **Re-Ranker:** SentenceTransformers CrossEncoder
6.  **LLM Backend:** Google Generative AI (`gemini-2.5-flash`)

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/finsentix-rag.git
cd finsentix-rag
```

### 2. Set Up a Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Create a `.env` file in the root directory and add your API Key:
```env
GOOGLE_API_KEY="your_google_ai_studio_key_here"
```

### 5. Run the Application
Launch the Streamlit web dashboard:
```bash
streamlit run app.py
```
Open `http://localhost:8501` in your browser. Upload your PDFs in the sidebar, click "Process Documents", and start asking questions!

---

## 📂 Project Structure

```text
├── app.py                            # Streamlit Web Application entry point
├── main.py                           # CLI entry point (optional alternative)
├── requirements.txt                  # Python dependencies
├── .env                              # Environment variables (Git-ignored)
├── ingestion/
│   └── document_processor.py         # Handles PDF loading and text chunking
├── retrieval/
│   ├── hybrid_retriever.py           # Retrieves chunks from ChromaDB
│   └── re_ranker.py                  # Cross-Encoder logic for ranking documents
├── generation/
│   └── llm_generator.py              # LLM Prompting & Answer Generation (Gemini)
└── evaluation/
    └── test_rag_pipeline.py          # Automated Ragas evaluation tests
```

---

## 🧠 Future Roadmap

- [ ] Implement BM25 Keyword Search to combine with Vector Search for true Hybrid Retrieval.
- [ ] Add conversational memory for follow-up questions.
- [ ] Integrate PostgreSQL/pgvector for cloud-scale deployment.

---
*Built with ❤️ to demonstrate production-ready ML engineering and generative AI best practices.*
