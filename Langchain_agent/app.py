import streamlit as st
import os
from dotenv import load_dotenv

from langchain_core.tools import tool
from langchain.agents import create_tool_calling_agent
from langchain.agents.agent import AgentExecutor
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

# Load env
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

st.title(" AI Agent Assistant")

# -------- TOOLS --------

@tool
def calculator(expression: str) -> str:
    """Perform math calculations"""
    try:
        return str(eval(expression))
    except:
        return "Invalid expression"

@tool
def search(query: str) -> str:
    """Search for general information"""
    return f"Search result for: {query}"

tools = [calculator, search]

# -------- AGENT --------

if not groq_api_key:
    st.error("❌ GROQ_API_KEY missing in .env")

else:
    llm = ChatGroq(
        groq_api_key=groq_api_key,
        model_name="llama-3.1-8b-instant"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}")
    ])

    agent = create_tool_calling_agent(
        llm=llm,
        tools=tools,
        prompt=prompt
    )

    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True
    )

    user_input = st.text_input("Ask something")

    if st.button("Run 🚀") and user_input:
        with st.spinner("Thinking..."):
            response = agent_executor.invoke({
                "input": user_input
            })

        st.success("Done ✅")
        st.write(response["output"])