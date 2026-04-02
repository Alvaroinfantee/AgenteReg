import streamlit as st
import os
from agents import AgentSystem
from logger import log_event

# Page config
st.set_page_config(page_title="Agente Regulación Bancaria RD", page_icon="🤖")

st.title("Agente Regulación Bancaria RD 20/12/25")

# Sidebar for API Key
with st.sidebar:
    st.header("Configuration")
    api_key = st.text_input("OpenAI API Key", type="password")
    if not api_key:
        # Check environment variable
        api_key = os.environ.get("OPENAI_API_KEY")
    
    if not api_key:
        st.warning("Please enter your OpenAI API Key to proceed.")
        st.stop()

# Initialize Agent System
if "agent_system" not in st.session_state:
    st.session_state.agent_system = AgentSystem(api_key=api_key)

# Session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Escribe tu pregunta aquí..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Log incoming query
    log_event("user_query", {"query": prompt})

    # Process likely involves multiple steps, so we use a spinner/status
    with st.status("Procesando consulta...", expanded=True) as status:
        
        # 1. Query Rewrite
        st.write("Rewriting query...")
        rewritten_query = st.session_state.agent_system.query_rewrite(prompt)
        st.write(f"Refined Query: *{rewritten_query}*")
        log_event("query_rewrite", {"original": prompt, "rewritten": rewritten_query})
        
        # 2. Classify
        st.write("Classifying intent...")
        operating_procedure = st.session_state.agent_system.classify(rewritten_query)
        st.write(f"Operating Procedure: **{operating_procedure}**")
        log_event("classification", {"query": rewritten_query, "procedure": operating_procedure})
        
        # 3. Route to specific agent
        response = ""
        try:
            if operating_procedure == "q-and-a":
                st.write("Routing to Internal Q&A...")
                response = st.session_state.agent_system.internal_qa(rewritten_query, st.session_state.messages)
            elif operating_procedure == "fact-finding":
                st.write("Routing to External Fact Finding...")
                response = st.session_state.agent_system.external_fact_finding(rewritten_query, st.session_state.messages)
            else:
                st.write("Routing to General Agent...")
                response = st.session_state.agent_system.general_agent(prompt, st.session_state.messages)
            
            status.update(label="Respuesta generada", state="complete", expanded=False)
            log_event("agent_response", {"procedure": operating_procedure, "response_length": len(response)})
            
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            response = "Lo siento, ocurrió un error al procesar tu solicitud."
            status.update(label="Error", state="error", expanded=True)
            log_event("error", {"error": str(e), "procedure": operating_procedure})

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(response)
