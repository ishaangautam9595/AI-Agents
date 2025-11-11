"""
Doc Insight Extractor Agent - RAG-based Document QA
Production-ready implementation with vector retrieval and citations
"""

import streamlit as st
import anthropic
from PyPDF2 import PdfReader
import hashlib
from typing import List, Dict, Tuple
import traceback
from datetime import datetime
import re

# Page configuration
st.set_page_config(
    page_title="Doc Insight Extractor",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'documents' not in st.session_state:
    st.session_state.documents = []
if 'qa_history' not in st.session_state:
    st.session_state.qa_history = []

def validate_api_key(api_key: str) -> bool:
    """Validate Anthropic API key format"""
    return api_key and api_key.startswith('sk-ant-') and len(api_key) > 20

def extract_text_from_pdf(file) -> Tuple[str, bool]:
    """
    Extract text from uploaded PDF file
    Returns: (text_content, success)
    """
    try:
        pdf_reader = PdfReader(file)
        text = ""
        
        for page_num, page in enumerate(pdf_reader.pages, 1):
            page_text = page.extract_text()
            if page_text:
                text += f"\n\n--- Page {page_num} ---\n\n{page_text}"
        
        if not text.strip():
            return "", False
        
        return text, True
        
    except Exception as e:
        st.error(f"Error reading PDF: {str(e)}")
        return "", False

def chunk_document(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[Dict]:
    """
    Split document into overlapping chunks for better retrieval
    
    Args:
        text: Full document text
        chunk_size: Target size for each chunk (characters)
        overlap: Overlap between chunks (characters)
    
    Returns:
        List of chunk dictionaries with metadata
    """
    chunks = []
    
    # Split by paragraphs first for better semantic boundaries
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    
    current_chunk = ""
    chunk_id = 0
    
    for para in paragraphs:
        # If adding this paragraph exceeds chunk size, save current chunk
        if len(current_chunk) + len(para) > chunk_size and current_chunk:
            chunks.append({
                "id": f"chunk_{chunk_id}",
                "text": current_chunk,
                "char_count": len(current_chunk),
                "hash": hashlib.md5(current_chunk.encode()).hexdigest()[:8]
            })
            
            # Start new chunk with overlap from previous chunk
            overlap_text = current_chunk[-overlap:] if len(current_chunk) > overlap else current_chunk
            current_chunk = overlap_text + "\n\n" + para
            chunk_id += 1
        else:
            current_chunk += ("\n\n" + para) if current_chunk else para
    
    # Add final chunk
    if current_chunk:
        chunks.append({
            "id": f"chunk_{chunk_id}",
            "text": current_chunk,
            "char_count": len(current_chunk),
            "hash": hashlib.md5(current_chunk.encode()).hexdigest()[:8]
        })
    
    return chunks

def semantic_search(query: str, chunks: List[Dict], client: anthropic.Anthropic, top_k: int = 3) -> List[Dict]:
    """
    Find most relevant chunks using Claude's understanding
    This is a lightweight alternative to vector embeddings for production use
    """
    try:
        # Create a summary of each chunk for comparison
        chunk_summaries = []
        
        for chunk in chunks[:20]:  # Limit to first 20 chunks for API efficiency
            summary_prompt = f"""Summarize this text in ONE sentence (max 20 words):

{chunk['text'][:500]}"""
            
            try:
                response = client.messages.create(
                    model="claude-3-5-haiku-20241022",  # Faster, cheaper model for summaries
                    max_tokens=50,
                    messages=[{"role": "user", "content": summary_prompt}]
                )
                chunk['summary'] = response.content[0].text
                chunk_summaries.append(chunk)
            except:
                chunk['summary'] = chunk['text'][:100]
                chunk_summaries.append(chunk)
        
        # Use Claude to rank chunks by relevance
        ranking_prompt = f"""Query: {query}

Rank these document chunks by relevance to the query (1 = most relevant).
Return ONLY a comma-separated list of chunk IDs in order of relevance.

Chunks:
"""
        for chunk in chunk_summaries:
            ranking_prompt += f"\n{chunk['id']}: {chunk.get('summary', chunk['text'][:100])}"
        
        ranking_prompt += "\n\nReturn format: chunk_2,chunk_5,chunk_1 (most relevant first)"
        
        response = client.messages.create(
            model="claude-3-5-haiku-20241022",
            max_tokens=100,
            messages=[{"role": "user", "content": ranking_prompt}]
        )
        
        # Parse rankings
        ranked_ids = [id.strip() for id in response.content[0].text.split(',')]
        
        # Return top K chunks in ranked order
        ranked_chunks = []
        for chunk_id in ranked_ids[:top_k]:
            for chunk in chunks:
                if chunk['id'] == chunk_id:
                    ranked_chunks.append(chunk)
                    break
        
        # If ranking failed, return first K chunks
        if not ranked_chunks:
            return chunks[:top_k]
        
        return ranked_chunks
        
    except Exception as e:
        st.warning(f"Semantic search failed, using fallback: {str(e)}")
        # Fallback: simple keyword matching
        query_words = set(query.lower().split())
        scored_chunks = []
        
        for chunk in chunks:
            chunk_words = set(chunk['text'].lower().split())
            score = len(query_words & chunk_words)
            scored_chunks.append((score, chunk))
        
        scored_chunks.sort(reverse=True, key=lambda x: x[0])
        return [chunk for _, chunk in scored_chunks[:top_k]]

def answer_with_citations(query: str, relevant_chunks: List[Dict], client: anthropic.Anthropic) -> Dict:
    """
    Generate answer using RAG with Claude, including citations
    """
    try:
        # Build context from chunks
        context = ""
        for i, chunk in enumerate(relevant_chunks, 1):
            context += f"\n\n[CHUNK {i} - ID: {chunk['id']}]\n{chunk['text']}\n"
        
        system_prompt = """You are a helpful AI assistant that answers questions based on provided document excerpts.

IMPORTANT RULES:
1. Answer ONLY based on the provided chunks
2. Cite your sources using [CHUNK X] notation
3. If the chunks don't contain relevant information, say so clearly
4. Be specific and quote relevant parts when appropriate
5. If asked about something not in the chunks, acknowledge the limitation"""

        user_prompt = f"""Context from document:
{context}

Question: {query}

Provide a comprehensive answer based on the context above. Include citations like [CHUNK 1] or [CHUNK 2] for each claim."""

        response = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=2048,
            temperature=0.3,
            messages=[
                {"role": "user", "content": user_prompt}
            ],
            system=system_prompt
        )
        
        answer = response.content[0].text
        
        return {
            "answer": answer,
            "chunks_used": relevant_chunks,
            "success": True
        }
        
    except Exception as e:
        return {
            "error": f"Failed to generate answer: {str(e)}",
            "traceback": traceback.format_exc(),
            "success": False
        }

# Sidebar Configuration
with st.sidebar:
    st.header(" Configuration")
    
    api_key = st.text_input(
        "Anthropic API Key",
        type="password",
        help="Get your API key from: https://console.anthropic.com/"
    )
    
    if api_key and not validate_api_key(api_key):
        st.error("⚠️ Invalid API key format")
    
    st.divider()
    
    st.markdown("### 📚 Document Library")
    
    if st.session_state.documents:
        for i, doc in enumerate(st.session_state.documents):
            with st.expander(f" {doc['name']}", expanded=False):
                st.metric("Chunks", doc['chunk_count'])
                st.metric("Characters", f"{doc['char_count']:,}")
                st.caption(f"Uploaded: {doc['timestamp']}")
                
                if st.button(f"Remove", key=f"remove_{i}"):
                    st.session_state.documents.pop(i)
                    st.rerun()
    else:
        st.info("No documents uploaded yet")
    
    st.divider()
    
    st.markdown("### RAG Settings")
    
    chunk_size = st.slider(
        "Chunk Size",
        min_value=500,
        max_value=2000,
        value=1000,
        step=100,
        help="Size of each text chunk (characters)"
    )
    
    top_k = st.slider(
        "Chunks to Retrieve",
        min_value=1,
        max_value=5,
        value=3,
        help="Number of relevant chunks to use for answering"
    )
    
    st.divider()
    
    st.markdown("### 📊 Statistics")
    st.metric("Documents", len(st.session_state.documents))
    st.metric("Q&A Sessions", len(st.session_state.qa_history))
    
    if st.button("🗑️ Clear All", use_container_width=True):
        if st.session_state.documents or st.session_state.qa_history:
            st.session_state.documents = []
            st.session_state.qa_history = []
            st.rerun()

# Main Interface
st.title(" Doc Insight Extractor Agent")
st.markdown("*RAG-powered document Q&A with semantic search and citations*")

# Tabs for Upload and Query
tab1, tab2 = st.tabs([" Upload Documents", " Ask Questions"])

with tab1:
    st.markdown("### Upload Your Documents")
    st.caption("Supported formats: PDF, TXT")
    
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=['pdf', 'txt'],
        help="Upload PDF or text documents to extract insights from"
    )
    
    if uploaded_file:
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.info(f"📎 **{uploaded_file.name}** ({uploaded_file.size:,} bytes)")
        
        with col2:
            process_btn = st.button("📥 Process", type="primary", use_container_width=True)
        
        if process_btn:
            with st.spinner("Processing document... This may take a minute."):
                # Extract text
                if uploaded_file.type == "application/pdf":
                    text, success = extract_text_from_pdf(uploaded_file)
                    if not success:
                        st.error(" Failed to extract text from PDF. The file may be scanned or corrupted.")
                        st.stop()
                else:
                    try:
                        text = uploaded_file.read().decode('utf-8')
                        success = True
                    except Exception as e:
                        st.error(f" Failed to read text file: {str(e)}")
                        st.stop()
                
                if not text or len(text.strip()) < 100:
                    st.error(" Document is too short or empty. Minimum 100 characters required.")
                    st.stop()
                
                # Chunk the document
                chunks = chunk_document(text, chunk_size=chunk_size, overlap=200)
                
                # Store document
                st.session_state.documents.append({
                    "name": uploaded_file.name,
                    "text": text,
                    "chunks": chunks,
                    "chunk_count": len(chunks),
                    "char_count": len(text),
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })
                
                st.success(f"✅ Successfully processed **{uploaded_file.name}** into {len(chunks)} chunks!")
                
                # Show preview
                with st.expander(" Document Preview (first 1000 characters)"):
                    st.text(text[:1000] + "..." if len(text) > 1000 else text)
                
                with st.expander(" Chunk Distribution"):
                    chunk_sizes = [chunk['char_count'] for chunk in chunks]
                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        st.metric("Total Chunks", len(chunks))
                    with col_b:
                        st.metric("Avg Size", f"{sum(chunk_sizes)//len(chunk_sizes):,}")
                    with col_c:
                        st.metric("Total Chars", f"{sum(chunk_sizes):,}")

with tab2:
    if not st.session_state.documents:
        st.warning("⚠️ No documents uploaded yet. Please upload a document in the 'Upload Documents' tab.")
        st.stop()
    
    st.markdown("### Ask Questions About Your Documents")
    
    # Document selector
    if len(st.session_state.documents) > 1:
        selected_docs = st.multiselect(
            "Select documents to search (leave empty for all)",
            [doc['name'] for doc in st.session_state.documents],
            default=[],
            help="Choose specific documents or search all"
        )
    else:
        selected_docs = []
    
    query = st.text_input(
        "Your question:",
        placeholder="e.g., What are the main findings? Summarize the methodology. What does the author conclude?",
        help="Ask specific questions about the document content"
    )
    
    # Example questions
    st.markdown("**💡 Example Questions:**")
    col1, col2, col3 = st.columns(3)
    
    example_questions = [
        "What are the key points?",
        "Summarize the main arguments",
        "What conclusions are drawn?"
    ]
    
    for col, example in zip([col1, col2, col3], example_questions):
        with col:
            if st.button(example, use_container_width=True):
                query = example
                st.rerun()
    
    st.divider()
    
    if st.button(" Get Answer", type="primary", use_container_width=True):
        
        # Validation
        if not api_key:
            st.error(" Please enter your Anthropic API key in the sidebar")
            st.stop()
        
        if not validate_api_key(api_key):
            st.error(" Invalid API key format")
            st.stop()
        
        if not query or len(query.strip()) < 5:
            st.warning(" Please enter a valid question (at least 5 characters)")
            st.stop()
        
        # Initialize client
        try:
            client = anthropic.Anthropic(api_key=api_key)
        except Exception as e:
            st.error(f" Failed to initialize Anthropic client: {str(e)}")
            st.stop()
        
        # Gather all relevant chunks
        all_chunks = []
        
        if selected_docs:
            for doc in st.session_state.documents:
                if doc['name'] in selected_docs:
                    all_chunks.extend(doc['chunks'])
        else:
            for doc in st.session_state.documents:
                all_chunks.extend(doc['chunks'])
        
        with st.spinner(f" Searching through {len(all_chunks)} chunks..."):
            # Find relevant chunks
            relevant_chunks = semantic_search(query, all_chunks, client, top_k=top_k)
        
        with st.spinner(" Generating answer with citations..."):
            # Generate answer
            result = answer_with_citations(query, relevant_chunks, client)
        
        if not result.get('success'):
            st.error(f" {result.get('error', 'Unknown error')}")
            with st.expander("🔍 Error Details"):
                st.code(result.get('traceback', 'No traceback available'))
            st.stop()
        
        # Display results
        st.success(" Answer Generated!")
        
        st.markdown("###  Answer")
        st.markdown(result['answer'])
        
        st.divider()
        
        # Display source chunks
        st.markdown("###  Source Chunks")
        st.caption("These are the document sections used to generate the answer")
        
        for i, chunk in enumerate(result['chunks_used'], 1):
            with st.expander(f"📄 Chunk {i} - ID: {chunk['id']} ({chunk['char_count']} chars)"):
                st.text_area(
                    f"Content",
                    chunk['text'],
                    height=200,
                    disabled=True,
                    label_visibility="collapsed"
                )
        
        # Save to history
        st.session_state.qa_history.append({
            "question": query,
            "answer": result['answer'][:300] + "..." if len(result['answer']) > 300 else result['answer'],
            "chunks_used": len(result['chunks_used']),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })

# Display Q&A History
if st.session_state.qa_history:
    st.divider()
    st.markdown("## 📜 Q&A History")
    
    for i, item in enumerate(reversed(st.session_state.qa_history), 1):
        with st.expander(
            f"Q{len(st.session_state.qa_history) - i + 1}: {item['question']} - {item['timestamp']}"
        ):
            st.markdown(f"**Question:** {item['question']}")
            st.markdown(f"**Answer:** {item['answer']}")
            st.caption(f"Used {item['chunks_used']} document chunks")

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p> Doc Insight Extractor | Powered by Claude 3.5 Sonnet | RAG Architecture</p>
</div>
""", unsafe_allow_html=True)