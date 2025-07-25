# app.py
import streamlit as st
from agents.ingestion_agent import IngestionAgent
from agents.retrieval_agent import RetrievalAgent
from agents.response_agent import LLMResponseAgent

st.title(" Agentic RAG Chatbot")

uploaded_files = st.file_uploader("Upload Documents", type=["pdf", "docx", "csv", "pptx", "txt"], accept_multiple_files=True)
query = st.text_input("Ask a question based on the uploaded documents")

if st.button("Submit") and uploaded_files and query:
    ingestion_agent = IngestionAgent()
    documents = ingestion_agent.ingest(uploaded_files)

    retrieval_agent = RetrievalAgent()
    top_chunks = retrieval_agent.retrieve(documents, query)

    response_agent = LLMResponseAgent()
    answer, sources = response_agent.generate_answer(query, top_chunks)

    st.markdown(f"### Answer: {answer}")
    st.markdown("**Sources:**")
    for src in sources:
        st.markdown(f"- {src}")


# agents/ingestion_agent.py
class IngestionAgent:
    def ingest(self, files):
        chunks = []
        for file in files:
            content = file.read().decode("utf-8", errors="ignore")
            chunks.append({"source": file.name, "content": content})
        return chunks


# agents/retrieval_agent.py
from sentence_transformers import SentenceTransformer, util
import numpy as np

class RetrievalAgent:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def retrieve(self, documents, query, top_k=3):
        doc_texts = [doc['content'] for doc in documents]
        doc_embeddings = self.model.encode(doc_texts, convert_to_tensor=True)
        query_embedding = self.model.encode(query, convert_to_tensor=True)

        scores = util.pytorch_cos_sim(query_embedding, doc_embeddings)[0]
        top_indices = scores.argsort(descending=True)[:top_k]

        top_chunks = [documents[i] for i in top_indices]
        return top_chunks


# agents/response_agent.py
import openai

class LLMResponseAgent:
    def __init__(self):
        openai.api_key = "your-api-key"

    def generate_answer(self, query, top_chunks):
        context = "\n\n".join([chunk['content'] for chunk in top_chunks])
        sources = [chunk['source'] for chunk in top_chunks]

        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
            ]
        )
        answer = response.choices[0].message.content
        return answer, sources


# utils/mcp.py
def create_mcp_message(sender, receiver, msg_type, trace_id, payload):
    return {
        "sender": sender,
        "receiver": receiver,
        "type": msg_type,
        "trace_id": trace_id,
        "payload": payload
    }


# utils/parsers.py
# (Extend later to handle parsing for PDF, DOCX, PPTX, etc.)
def parse_file(file):
    return file.read().decode("utf-8", errors="ignore")
