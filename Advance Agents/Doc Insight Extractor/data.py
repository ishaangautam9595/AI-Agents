"""
Doc Insight Extractor Agent - RAG-based Document QA
Uploads docs/PDFs, retrieves relevant sections via vector search, answers with citations
"""

import streamlit as st
import anthropic
from PyPDF2 import PdfReader
import numpy as np
from typing import List, Dict
import hashlib

st.set_page_config(page_title="Doc Insight Extractor", page_icon="📄", layout="wide")

if 'documents' not in st.session_state:
    st.session_state.documents = []
if 'qa_history' not in st.session_state:
    st.session_state.qa_history = []

def extract_text_from_pdf(file) -> str:
    """Extract text from uploaded PDF"""
    try:
        pdf_reader = PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {str(e)}")
        return ""

def chunk_text(text: str, chunk_size: int = 1000) -> List[Dict]:
    """Split text into overlapping chunks for better retrieval"""
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size - 100):  # 100 word overlap
        chunk = ' '.join(words[i:i + chunk_size])
        chunks.append({
            "text": chunk,
            "start_idx": i,
            "id": hashlib.md5(chunk.encode()).hexdigest()[:8]
        })
    
    return chunks

def simple_embedding(text: str) -> np.ndarray:
    """Create simple embedding based on word frequencies (mock for demo)"""
    # In production, use: sentence-transformers or OpenAI embeddings
    words = text.lower().split()
    vocab = set(words)
    vector = np.zeros(100)
    
    for i, word in enumerate(vocab):
        if i < 100:
            vector[i] = words.count(word) / len(words)
    
    return vector / (np.linalg.norm(vector) + 1e-8)

def find_relevant_chunks(query: str, chunks: List[Dict], top_k: int = 3) -> List[Dict]:
    """Find most relevant chunks using vector similarity"""
    query_vec = simple_embedding(query)
    
    scored_chunks = []
    for chunk in chunks:
        chunk_vec = simple_embedding(chunk['text'])
        similarity = np.dot(query_vec, chunk_vec)
        scored_chunks.append((similarity, chunk))
    
    scored_chunks.sort(reverse=True, key=lambda x: x[0])
    return [chunk for _, chunk in scored_chunks[:top_k]]

def answer_with_rag(query: str, relevant_chunks: List[Dict], api_key: str) -> Dict:
    """Generate answer using Claude with RAG context"""
    try:
        client = anthropic.Anthropic(api_key=api_key)
        
        context = "\n\n".join([
            f"[Chunk {i+1} - ID: {chunk['id']}]\n{chunk['text']}"
            for i, chunk in enumerate(relevant_chunks)
        ])
        
        message = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1024,
            messages=[{
                "role": "user",
                "content": f"""Based on the following document excerpts, answer the question.
Cite the chunk IDs in your answer.

Context:
{context}

Question: {query}

Provide a clear answer with citations like [Chunk 1] or [ID: abc123]."""
            }]
        )
        
        return {
            "answer": message.content[0].text,
            "chunks_used": relevant_chunks
        }
    
    except Exception as e:
        return {"error": str(e)}

# UI Layout
st.title("📄 Doc Insight Extractor Agent")
st.markdown("*RAG-based document QA with vector retrieval and citations*")

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    api_key = st.text_input("Anthropic API Key", type="password",
                           help="Get it from: https://console.anthropic.com/")
    
    st.markdown("---")
    st.markdown("### 📚 Uploaded Documents")
    if st.session_state.documents:
        for doc in st.session_state.documents:
            st.markdown(f"- {doc['name']} ({doc['chunks']} chunks)")
    else:
        st.info("No documents uploaded yet")
    
    if st.button("Clear All Documents"):
        st.session_state.documents = []
        st.session_state.qa_history = []
        st.rerun()

# Main interface
tab1, tab2 = st.tabs(["Upload Documents", "Ask Questions"])

with tab1:
    st.markdown("### Upload your documents")
    uploaded_file = st.file_uploader(
        "Choose a PDF or TXT file",
        type=['pdf', 'txt'],
        help="Upload documents to extract insights from"
    )
    
    if uploaded_file:
        if st.button("Process Document", type="primary"):
            with st.spinner("Processing document..."):
                # Extract text
                if uploaded_file.type == "application/pdf":
                    text = extract_text_from_pdf(uploaded_file)
                else:
                    text = uploaded_file.read().decode('utf-8')
                
                if text:
                    # Chunk text
                    chunks = chunk_text(text)
                    
                    # Store document
                    st.session_state.documents.append({
                        "name": uploaded_file.name,
                        "text": text,
                        "chunks": len(chunks),
                        "chunk_data": chunks
                    })
                    
                    st.success(f"Processed {uploaded_file.name} into {len(chunks)} chunks!")
                    
                    # Show preview
                    with st.expander("Preview Document"):
                        st.text(text[:1000] + "..." if len(text) > 1000 else text)

with tab2:
    if not st.session_state.documents:
        st.warning("⚠️ Please upload documents first in the 'Upload Documents' tab")
    else:
        st.markdown("### Ask questions about your documents")
        
        query = st.text_input(
            "Your question:",
            placeholder="e.g., What are the main findings? Summarize the methodology."
        )
        
        if st.button("🔍 Get Answer", type="primary"):
            if not api_key:
                st.error("Please enter your Anthropic API key in the sidebar")
            elif not query:
                st.warning("Please enter a question")
            else:
                with st.spinner("Searching documents and generating answer..."):
                    # Combine all chunks from all documents
                    all_chunks = []
                    for doc in st.session_state.documents:
                        all_chunks.extend(doc['chunk_data'])
                    
                    # Find relevant chunks
                    relevant = find_relevant_chunks(query, all_chunks, top_k=3)
                    
                    # Generate answer
                    result = answer_with_rag(query, relevant, api_key)
                    
                    if "error" not in result:
                        st.success("Answer Generated!")
                        
                        # Display answer
                        st.markdown("### Answer")
                        st.markdown(result['answer'])
                        
                        # Display sources
                        st.markdown("---")
                        st.markdown("### Source Chunks")
                        for i, chunk in enumerate(result['chunks_used']):
                            with st.expander(f"Chunk {i+1} - ID: {chunk['id']}"):
                                st.text(chunk['text'][:500] + "..." if len(chunk['text']) > 500 else chunk['text'])
                        
                        # Save to history
                        st.session_state.qa_history.append({
                            "question": query,
                            "answer": result['answer'][:200] + "..."
                        })
                    else:
                        st.error(f"Error: {result['error']}")

# Display Q&A history
if st.session_state.qa_history:
    st.markdown("---")
    st.markdown("## Q&A History")
    
    for i, item in enumerate(reversed(st.session_state.qa_history)):
        with st.expander(f"Q{len(st.session_state.qa_history) - i}: {item['question']}"):
            st.markdown(f"**A**: {item['answer']}")