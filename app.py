import streamlit as st
from agent import PlannerAgent

st.title("Marketing Planner Agent")

# Initialize the agent
if 'agent' not in st.session_state:
    st.session_state.agent = PlannerAgent()
    st.session_state.agent._fetch_models()
    st.session_state.agent.chat_history.append({"role": "agent", "content": "Hello! I am your marketing planner assistant. How can I help you today?"})

# Display chat messages from history on app rerun
for message in st.session_state.agent.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Get user input
user_input = st.chat_input("You:")

if user_input:
    # Add user message to chat history and display it
    st.session_state.agent.chat_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # Let the agent process the input
    response = st.session_state.agent._send_llm_request(user_input)
    
    # Display the response
    with st.chat_message("agent"):
        st.markdown(response)
