"""
Query Resolver Agent - Natural Language Query Handler
Handles queries by breaking them into steps: parse intent, fetch data, generate response
"""

import streamlit as st
import openai
from typing import Dict, List
import json

# Configure page
st.set_page_config(page_title="Query Resolver Agent", page_icon="🔍", layout="wide")

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Mock data store
MOCK_DATA = {
    "weather": {"city": "New York", "temp": 72, "condition": "Sunny"},
    "stock": {"AAPL": 180.5, "GOOGL": 140.2, "MSFT": 380.0},
    "user": {"name": "John Doe", "age": 30, "location": "NYC"}
}

def parse_intent(query: str, api_key: str) -> Dict:
    """Parse user intent using chain-of-thought prompting"""
    try:
        client = openai.OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{
                "role": "system",
                "content": """Analyze the query and identify:
1. Intent type (weather/stock/user info/general)
2. Key entities mentioned
3. Required data fields

Respond in JSON format: {"intent": "type", "entities": ["entity1"], "fields": ["field1"]}"""
            }, {
                "role": "user",
                "content": query
            }],
            temperature=0.3
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        return {"intent": "general", "entities": [], "fields": [], "error": str(e)}

def fetch_mock_data(intent_data: Dict) -> Dict:
    """Fetch relevant mock data based on intent"""
    intent = intent_data.get("intent", "general")
    if intent in MOCK_DATA:
        return MOCK_DATA[intent]
    return {}

def generate_response(query: str, context: Dict, api_key: str) -> str:
    """Generate final response with context"""
    try:
        client = openai.OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{
                "role": "system",
                "content": f"""You are a helpful assistant. Use this context to answer:
Context: {json.dumps(context)}

Provide a natural, conversational response."""
            }, {
                "role": "user",
                "content": query
            }],
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error generating response: {str(e)}"

# UI Layout
st.title("🔍 Query Resolver Agent")
st.markdown("*Natural language query handler with chain-of-thought reasoning*")

# Sidebar for API key
with st.sidebar:
    st.header("⚙️ Configuration")
    api_key = st.text_input("OpenAI API Key", type="password", help="Enter your OpenAI API key")
    st.markdown("---")
    st.markdown("### How it works")
    st.markdown("""
    1. **Parse Intent**: Analyzes your query
    2. **Fetch Data**: Retrieves relevant information
    3. **Generate Response**: Creates natural answer
    """)
    if st.button("Clear History"):
        st.session_state.chat_history = []
        st.rerun()

# Main interface
col1, col2 = st.columns([2, 1])

with col1:
    query = st.text_area("Enter your query:", height=100, 
                        placeholder="e.g., What's the weather like? Show me stock prices.")
    
    if st.button("🚀 Process Query", type="primary", use_container_width=True):
        if not api_key:
            st.error("Please enter your OpenAI API key in the sidebar")
        elif not query:
            st.warning("Please enter a query")
        else:
            with st.spinner("Processing..."):
                # Step 1: Parse intent
                with st.expander(" Step 1: Intent Parsing", expanded=True):
                    intent_data = parse_intent(query, api_key)
                    st.json(intent_data)
                
                # Step 2: Fetch data
                with st.expander(" Step 2: Data Retrieval", expanded=True):
                    data = fetch_mock_data(intent_data)
                    st.json(data if data else {"message": "No specific data needed"})
                
                # Step 3: Generate response
                with st.expander(" Step 3: Response Generation", expanded=True):
                    response = generate_response(query, data, api_key)
                    st.success(response)
                    
                # Add to history
                st.session_state.chat_history.append({
                    "query": query,
                    "intent": intent_data,
                    "response": response
                })

with col2:
    st.markdown("### 📚 Mock Data Store")
    st.json(MOCK_DATA)

# Display chat history
if st.session_state.chat_history:
    st.markdown("---")
    st.markdown("### Query History")
    for i, item in enumerate(reversed(st.session_state.chat_history)):
        with st.expander(f"Query {len(st.session_state.chat_history) - i}: {item['query'][:50]}..."):
            st.markdown(f"**Intent**: {item['intent'].get('intent', 'N/A')}")
            st.markdown(f"**Response**: {item['response']}")