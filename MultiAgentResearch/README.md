# AI Multi-Agent Research Assistant

A production-ready GenAI application built with Streamlit, LangChain, and LangGraph. This application uses an orchestra of AI agents to search the web, scrape relevant pages, summarize them, and finally generate a comprehensive research report based on your query.

## Architecture

The project is structured modularly:
- **`app.py`**: The Streamlit frontend for user interaction.
- **`graph.py`**: The LangGraph workflow orchestrating the agents.
- **`agents.py`**: The LLM logic defining our Search, Scraper, Summarizer, and Report agents.
- **`tools.py`**: Web search (Serper API) and scraping (BeautifulSoup) tools.

## Prerequisites

- Python 3.9+
- [Groq API Key](https://console.groq.com/keys) (Free fast inference)
- [Serper API Key](https://serper.dev/) (For Google Search results)

## Setup Instructions

1. **Navigate to the project directory**
   ```bash
   cd d:\Sourcesys\MultiAgentResearch
   ```

2. **(Optional but recommended) Create a virtual environment**
   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # macOS/Linux
   source venv/bin/activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure Environment Variables**
   Create a `.env` file in the root directory (you can copy `.env.example`):
   ```env
   GROQ_API_KEY=your_groq_api_key_here
   SERPER_API_KEY=your_serper_api_key_here
   ```
   *Note: You can also input these keys directly in the Streamlit UI sidebar.*

## Running the Application

1. Start the Streamlit app:
   ```bash
   streamlit run app.py
   ```
2. Open your browser to the URL provided (usually `http://localhost:8501`).
3. Enter your research query and click **Start Research**.
