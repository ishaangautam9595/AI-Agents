"""
Query Resolver Agent - Natural Language Query Handler
Production-ready implementation with chain-of-thought reasoning
"""

import streamlit as st
from openai import OpenAI
import json
from typing import Dict, Optional
import traceback

# Page configuration
st.set_page_config(
    page_title="Query Resolver Agent",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

def validate_api_key(api_key: str) -> bool:
    """Validate OpenAI API key format"""
    return api_key and api_key.startswith('sk-') and len(api_key) > 20

def parse_intent_with_cot(query: str, client: OpenAI) -> Dict:
    """
    Parse user intent using chain-of-thought prompting
    Returns structured intent data
    """
    try:
        system_prompt = """You are an expert query analyzer. Analyze queries using chain-of-thought reasoning.

Break down the query step by step:
1. Identify the main intent/goal
2. Extract key entities and parameters
3. Determine required information type
4. Identify any ambiguities

Return ONLY valid JSON (no markdown):
{
  "intent": "specific intent type",
  "entities": ["entity1", "entity2"],
  "query_type": "question/command/search/analysis",
  "reasoning": "your step-by-step thought process",
  "required_info": ["info1", "info2"],
  "ambiguities": ["any unclear aspects"]
}"""
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Analyze this query: {query}"}
            ],
            temperature=0.3,
            max_tokens=500
        )
        
        content = response.choices[0].message.content.strip()
        
        # Clean potential markdown
        if '```json' in content:
            content = content.split('```json')[1].split('```')[0].strip()
        elif '```' in content:
            content = content.split('```')[1].split('```')[0].strip()
        
        return json.loads(content)
        
    except json.JSONDecodeError as e:
        return {
            "error": "Failed to parse intent",
            "details": f"JSON parsing error: {str(e)}",
            "raw_response": content if 'content' in locals() else "No response"
        }
    except Exception as e:
        return {
            "error": "Intent parsing failed",
            "details": str(e),
            "traceback": traceback.format_exc()
        }

def generate_comprehensive_response(query: str, intent_data: Dict, client: OpenAI) -> str:
    """
    Generate comprehensive response using the parsed intent
    """
    try:
        system_prompt = """You are a helpful AI assistant that provides comprehensive, accurate answers.

Guidelines:
- Provide detailed, informative responses
- Use the intent analysis to structure your answer
- Be specific and actionable
- Cite reasoning when making claims
- If uncertain, acknowledge limitations
- Format with clear structure when appropriate"""

        user_prompt = f"""Query: {query}

Intent Analysis:
{json.dumps(intent_data, indent=2)}

Provide a comprehensive answer that addresses the user's intent."""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7,
            max_tokens=1000
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        return f"Error generating response: {str(e)}\n\nPlease try rephrasing your query."

# Sidebar Configuration
with st.sidebar:
    st.header(" Configuration")
    
    api_key = st.text_input(
        "OpenAI API Key",
        type="password",
        help="Enter your OpenAI API key (starts with sk-)"
    )
    
    if api_key and not validate_api_key(api_key):
        st.error(" Invalid API key format")
    
    st.divider()
    
    st.markdown("###  How It Works")
    st.markdown("""
    **Chain-of-Thought Process:**
    
    1. **Parse Intent** 
       - Analyze query structure
       - Extract entities
       - Identify goal
    
    2. **Reason** 
       - Apply logical steps
       - Consider context
       - Structure approach
    
    3. **Generate Response** 
       - Synthesize information
       - Provide clear answer
       - Include reasoning
    """)
    
    st.divider()
    
    st.markdown("###  Statistics")
    st.metric("Queries Processed", len(st.session_state.chat_history))
    
    if st.button("🗑️ Clear History", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

# Main Interface
st.title(" Query Resolver Agent")
st.markdown("*AI-powered query handler with transparent chain-of-thought reasoning*")

# Input Section
query = st.text_area(
    "Enter your query:",
    height=120,
    placeholder="Ask anything... Examples:\n• What are the benefits of renewable energy?\n• How does machine learning work?\n• Explain quantum computing in simple terms",
    help="Type any question or request"
)

# Example queries
st.markdown("**Quick Examples:**")
col1, col2, col3 = st.columns(3)

examples = [
    "Explain photosynthesis step by step",
    "What are the main causes of climate change?",
    "How do neural networks learn?"
]

for i, (col, example) in enumerate(zip([col1, col2, col3], examples)):
    with col:
        if st.button(f" Example {i+1}", use_container_width=True, key=f"ex_{i}"):
            query = example
            st.rerun()

st.divider()

# Process Button
if st.button(" Process Query", type="primary", use_container_width=True):
    
    # Validation
    if not api_key:
        st.error(" Please enter your OpenAI API key in the sidebar")
        st.stop()
    
    if not validate_api_key(api_key):
        st.error(" Invalid API key format")
        st.stop()
    
    if not query or len(query.strip()) < 3:
        st.warning(" Please enter a valid query (at least 3 characters)")
        st.stop()
    
    # Initialize client
    try:
        client = OpenAI(api_key=api_key)
    except Exception as e:
        st.error(f" Failed to initialize OpenAI client: {str(e)}")
        st.stop()
    
    # Processing
    with st.spinner(" Analyzing query..."):
        # Step 1: Parse Intent
        intent_data = parse_intent_with_cot(query, client)
        
        if "error" in intent_data:
            st.error(f" {intent_data['error']}")
            with st.expander(" Error Details"):
                st.json(intent_data)
            st.stop()
        
        # Display Intent Analysis
        with st.expander(" Step 1: Intent Analysis", expanded=True):
            col_a, col_b = st.columns(2)
            
            with col_a:
                st.markdown("**Intent Type:**")
                st.info(intent_data.get('intent', 'Unknown'))
                
                st.markdown("**Query Type:**")
                st.info(intent_data.get('query_type', 'Unknown'))
            
            with col_b:
                st.markdown("**Entities Identified:**")
                entities = intent_data.get('entities', [])
                if entities:
                    for entity in entities:
                        st.markdown(f"• {entity}")
                else:
                    st.markdown("_None identified_")
            
            st.markdown("**Reasoning Process:**")
            st.text_area(
                "Chain-of-Thought",
                intent_data.get('reasoning', 'No reasoning provided'),
                height=100,
                disabled=True,
                label_visibility="collapsed"
            )
            
            if intent_data.get('ambiguities'):
                st.warning("**Ambiguities Detected:**")
                for amb in intent_data['ambiguities']:
                    st.markdown(f"• {amb}")
    
    with st.spinner("💭 Generating comprehensive response..."):
        # Step 2: Generate Response
        response = generate_comprehensive_response(query, intent_data, client)
        
        # Display Response
        st.markdown("---")
        st.markdown("### Response")
        st.markdown(response)
        
        # Save to history
        st.session_state.chat_history.append({
            "query": query,
            "intent": intent_data.get('intent', 'Unknown'),
            "query_type": intent_data.get('query_type', 'Unknown'),
            "response": response,
            "timestamp": st.session_state.get('timestamp', 0)
        })
        
        st.success(" Query processed successfully!")

# Display Chat History
if st.session_state.chat_history:
    st.divider()
    st.markdown("##  Query History")
    
    for i, item in enumerate(reversed(st.session_state.chat_history), 1):
        with st.expander(
            f"Query {len(st.session_state.chat_history) - i + 1}: {item['query'][:60]}{'...' if len(item['query']) > 60 else ''}",
            expanded=False
        ):
            col1, col2 = st.columns([1, 3])
            
            with col1:
                st.markdown("**Intent:**")
                st.code(item['intent'])
                st.markdown("**Type:**")
                st.code(item['query_type'])
            
            with col2:
                st.markdown("**Query:**")
                st.info(item['query'])
                
                st.markdown("**Response:**")
                st.markdown(item['response'][:500] + "..." if len(item['response']) > 500 else item['response'])

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p> Query Resolver Agent | Powered by GPT-4o-mini | Chain-of-Thought Reasoning</p>
</div>
""", unsafe_allow_html=True)