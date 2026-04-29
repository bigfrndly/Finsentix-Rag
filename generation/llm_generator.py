from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.documents import Document

class LLMGenerator:
    """
    Handles prompt construction with strict citation enforcement and text generation.
    """
    def __init__(self, model_name: str = "gemini-2.5-flash", temperature: float = 0.0):
        # We use a low temperature to prioritize factuality over creativity
        self.llm = ChatGoogleGenerativeAI(model=model_name, temperature=temperature)
        
        # This prompt is crucial for enforcing citations. Notice the strict instructions.
        template = """
You are a reliable, factual assistant for answering questions based ONLY on the provided document context.

Context Documents:
{context}

Question: {question}

Instructions:
1. You must answer the Question using ONLY the information found in the Context Documents.
2. If the answer cannot be confidently formulated from the context, respond with "I cannot answer this based on the provided documents."
3. Every factual claim in your answer MUST be accompanied by an inline citation to the chunk that provided the information.
4. Use the exact identifier format provided in the context, e.g., [Doc 1], [Doc 2]. 
5. Do NOT include any external knowledge.

Answer:
"""
        self.prompt = PromptTemplate(template=template, input_variables=["context", "question"])

    def _format_context(self, docs: list[Document]) -> str:
        """
        Formats documents with clear identifiers so the LLM knows how to cite them.
        """
        formatted_str = ""
        for i, doc in enumerate(docs):
            # Source metadata or chunk_id can be used as the identifier
            source = doc.metadata.get("source", "Unknown")
            page = doc.metadata.get("page", "N/A")
            chunk_id = doc.metadata.get("chunk_id", f"unknown_chunk")
            
            identifier = f"[Doc {i+1}]"
            content = doc.page_content.replace("\n", " ")
            
            formatted_str += f"{identifier} (Source: {source}, Page: {page}, ID: {chunk_id}):\n{content}\n\n"
            
        return formatted_str

    def generate(self, query: str, retrieved_docs: list[Document]) -> str:
        """
        Generates an answer with citations based on the retrieved docs.
        """
        context_str = self._format_context(retrieved_docs)
        
        # Create full prompt string
        final_prompt = self.prompt.format(context=context_str, question=query)
        
        # Call LLM
        response = self.llm.invoke(final_prompt)
        
        return response.content
