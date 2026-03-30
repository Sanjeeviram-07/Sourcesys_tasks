import os
import logging
import streamlit as st
import time
from dotenv import load_dotenv
from graph import research_graph
from fpdf import FPDF

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

load_dotenv()

st.set_page_config(page_title="AI Research Assistant", layout="centered", initial_sidebar_state="expanded")

def inject_custom_css():
    st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
    color: #f8fafc;
}
h1, h2, h3, h4, h5, h6 {
    color: #ffffff !important;
    font-weight: 700 !important;
    font-family: 'Inter', sans-serif !important;
}
p, li, span {
    color: #cbd5e1;
    font-family: 'Inter', sans-serif !important;
}
div[data-baseweb="input"] > div {
    background-color: rgba(255, 255, 255, 0.05) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: 12px !important;
    backdrop-filter: blur(10px);
    color: white !important;
    padding: 8px;
}
div[data-baseweb="input"] > div:focus-within {
    border-color: #22c55e !important;
    box-shadow: 0 0 10px rgba(34, 197, 94, 0.3) !important;
}
div[data-baseweb="input"] input {
    color: white !important;
    font-size: 16px !important;
}
div[data-baseweb="input"] input::placeholder {
    color: #94a3b8 !important;
}
div.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #22c55e, #16a34a) !important;
    color: white !important;
    font-size: 16px !important;
    font-weight: 600 !important;
    padding: 12px 24px !important;
    border-radius: 14px !important;
    border: none !important;
    box-shadow: 0 0 15px rgba(34, 197, 94, 0.5) !important;
    transition: all 0.3s ease !important;
    height: 55px !important;
}
div.stButton > button[kind="primary"]:hover {
    transform: scale(1.02) !important;
    background: linear-gradient(135deg, #16a34a, #15803d) !important;
    box-shadow: 0 0 20px rgba(34, 197, 94, 0.7) !important;
}
div.stButton > button[kind="secondary"] {
    background: transparent !important;
    color: #cbd5e1 !important;
    font-size: 16px !important;
    font-weight: 600 !important;
    padding: 12px 24px !important;
    border-radius: 14px !important;
    border: 1px solid rgba(255, 255, 255, 0.2) !important;
    transition: all 0.3s ease !important;
    height: 55px !important;
}
div.stButton > button[kind="secondary"]:hover {
    background: rgba(255, 255, 255, 0.08) !important;
    border-color: rgba(255, 255, 255, 0.4) !important;
    color: white !important;
}
div[data-testid="stVerticalBlockBorderWrapper"] {
    background: rgba(15, 23, 42, 0.4) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: 20px !important;
    backdrop-filter: blur(16px) !important;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3) !important;
    padding: 2rem !important;
    transition: all 0.5s ease;
    animation: fadeIn 0.8s ease-in-out;
    margin-top: 1rem;
}
[data-testid="stSidebar"] {
    background: rgba(15, 23, 42, 0.8) !important;
    backdrop-filter: blur(15px) !important;
    border-right: 1px solid rgba(255, 255, 255, 0.05) !important;
}
[data-testid="stSidebar"] div.stButton > button {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    text-align: left !important;
    color: #cbd5e1 !important;
    padding: 10px 15px !important;
    height: auto !important;
    justify-content: flex-start !important;
    border-radius: 8px !important;
    width: 100% !important;
    transition: all 0.2s ease !important;
}
[data-testid="stSidebar"] div.stButton > button:hover {
    background: rgba(255, 255, 255, 0.1) !important;
    color: white !important;
    transform: translateX(4px) !important;
}
.block-container {
    max-width: 850px;
    padding-top: 3rem;
    padding-bottom: 3rem;
}
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(15px); }
    to { opacity: 1; transform: translateY(0); }
}
.shimmer {
    background: linear-gradient(90deg, rgba(255,255,255,0.03) 25%, rgba(255,255,255,0.08) 50%, rgba(255,255,255,0.03) 75%);
    background-size: 200% 100%;
    animation: shimmer 1.5s infinite;
    border-radius: 12px;
    height: 20px;
    margin-bottom: 15px;
    width: 100%;
}
.shimmer.short { width: 60%; }
.shimmer.medium { width: 80%; }
@keyframes shimmer {
    0% { background-position: 200% 0; }
    100% { background-position: -200% 0; }
}
hr {
    border-color: rgba(255, 255, 255, 0.08) !important;
    margin: 2.5rem 0 !important;
}
.stMarkdown a {
    color: #4ade80 !important;
    text-decoration: none !important;
    transition: color 0.2s;
}
.stMarkdown a:hover {
    color: #22c55e !important;
    text-decoration: underline !important;
}
#MainMenu {visibility: hidden;}
header {visibility: hidden;}
footer {visibility: hidden;}
</style>
    """, unsafe_allow_html=True)

def initialize_session_state():
    if "report" not in st.session_state:
        st.session_state.report = None
    if "processing" not in st.session_state:
        st.session_state.processing = False
    if "query_input" not in st.session_state:
        st.session_state.query_input = ""
    if "history" not in st.session_state:
        st.session_state.history = []

def generate_pdf(query, report_text):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Helvetica", 'B', size=16)
    safe_query = query.encode('latin-1', 'replace').decode('latin-1')
    pdf.cell(0, 10, txt=f"Research Report: {safe_query}", ln=True, align='C')
    pdf.ln(5)
    pdf.set_font("Helvetica", size=11)
    safe_report = report_text.encode('latin-1', 'replace').decode('latin-1')
    pdf.multi_cell(0, 7, txt=safe_report)
    try:
        return bytes(pdf.output())
    except TypeError:
        return pdf.output(dest='S').encode('latin-1')

def clear_results():
    st.session_state.report = None
    st.session_state.processing = False
    st.session_state.query_input = ""

def load_history_item(idx):
    item = st.session_state.history[idx]
    st.session_state.report = item["report"]
    st.session_state.query_input = item["query"]

def run_research(query):
    if not query.strip():
        st.warning("Please enter a research topic.")
        return
    if not os.getenv("GROQ_API_KEY") or not os.getenv("SERPER_API_KEY"):
        st.error("Missing API Keys in environment variables. Please check your .env file.")
        return
    st.session_state.processing = True
    st.session_state.report = None
    status_placeholder = st.empty()
    status_placeholder.markdown(
        "<div class='shimmer medium'></div>"
        "<div class='shimmer'></div>"
        "<div class='shimmer'></div>"
        "<div class='shimmer short'></div>", 
        unsafe_allow_html=True
    )
    logger.info(f"Starting research workflow for query: {query}")
    try:
        final_state = None
        for output in research_graph.stream({"query": query}):
            for node_name, state_update in output.items():
                logger.info(f"Node completed: {node_name}")
                final_state = state_update
        if final_state and "final_report" in final_state:
            st.session_state.report = final_state["final_report"]
            if not st.session_state.history or st.session_state.history[0]["query"] != query:
                st.session_state.history.insert(0, {
                    "query": query, 
                    "report": final_state["final_report"],
                    "timestamp": time.strftime("%I:%M %p")
                })
            logger.info("Research workflow completed successfully.")
        else:
            st.error("Workflow completed but no report was generated.")
    except Exception as e:
        error_msg = f"An error occurred during execution: {str(e)}"
        st.error(error_msg)
        logger.exception("Workflow execution failed.")
    finally:
        status_placeholder.empty()
        st.session_state.processing = False

def main():
    initialize_session_state()
    inject_custom_css()
    
    with st.sidebar:
        st.title("App Navigation")
        st.markdown("Easily manage your research tasks.")
        nav_mode = st.radio("Mode", ["Home", "History"], label_visibility="collapsed")
        st.divider()
        if st.session_state.history:
            st.markdown("### Recent History")
            for idx, item in enumerate(st.session_state.history):
                btn_label = f"{item['query'][:25]}..." if len(item['query']) > 25 else item['query']
                st.button(btn_label, key=f"hist_{idx}", on_click=load_history_item, args=(idx,))
        else:
            st.markdown("*(No history yet)*")

    st.title("AI Research Assistant")
    st.markdown("#### Autonomous research, synthesized insights, and comprehensive reports.")
    st.divider()

    with st.container():
        query = st.text_input(
            "Research Topic", 
            key="query_input",
            label_visibility="collapsed", 
            placeholder="Enter your research topic... (e.g. Latest solid-state batteries)"
        )
        st.markdown("<br>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            st.button("Clear", on_click=clear_results, use_container_width=True)
        with col3:
            generate_clicked = st.button("Generate Report", type="primary", use_container_width=True)

    if generate_clicked and query:
        with st.spinner("Analyzing sources and generating insights..."):
            run_research(query)

    if st.session_state.report:
        st.divider()
        st.markdown("### Research Report")
        with st.container(border=True):
            st.markdown(st.session_state.report)
        st.markdown("<br>", unsafe_allow_html=True)
        pdf_bytes = generate_pdf(query, st.session_state.report)
        st.download_button(
            label="Download as PDF",
            data=pdf_bytes,
            file_name=f"research_report_{int(time.time())}.pdf",
            mime="application/pdf",
            use_container_width=True
        )

if __name__ == "__main__":
    main()
