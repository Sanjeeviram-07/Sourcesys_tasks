import streamlit as st
from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

st.title("AI Chatbot Powered By Llama3")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_input = st.chat_input("Ask something...")

if user_input:
    if user_input.lower() == "exit":
        with st.chat_message("assistant"):
            st.markdown("Thank you!")
        st.stop()

    with st.chat_message("user"):
        st.markdown(user_input)

    st.session_state.messages.append(
        {"role": "user", "content": user_input}
    )

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=st.session_state.messages
    )

    reply = response.choices[0].message.content

    with st.chat_message("assistant"):
        st.markdown(reply)

    st.session_state.messages.append(
        {"role": "assistant", "content": reply}
    )
    