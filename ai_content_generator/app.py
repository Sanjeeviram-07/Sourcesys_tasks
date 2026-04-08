import streamlit as st
import os
from dotenv import load_dotenv

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

# Load env variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# UI
st.set_page_config(page_title="AI Content Generator", page_icon="🧩")
st.title(" AI Content Generator ")

topic = st.text_input("Enter Topic")

if not groq_api_key:
    st.error(" GROQ_API_KEY missing in .env")

elif topic:

    # LLM
    llm = ChatGroq(
        groq_api_key=groq_api_key,
        model_name="llama-3.1-8b-instant"
    )

    # Prompt
    prompt = PromptTemplate.from_template("""
    Topic: {topic}

    Generate:
    1. A short summary
    2. 5 key points
    3. 3 interview questions with answers
    """)

    # Output parser
    parser = StrOutputParser()

    # LCEL Chain 
    chain = prompt | llm | parser

    if st.button("Generate "):
        with st.spinner("Generating..."):
            result = chain.invoke({"topic": topic})

        st.success("Done ")
        st.markdown("###  Output")
        st.write(result)

else:
    st.info("Enter a topic to begin")
