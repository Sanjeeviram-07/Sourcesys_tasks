import os
import json
import logging
from typing import TypedDict, List
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage
from tools import search_web_tool, scrape_website_tool

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class ResearchState(TypedDict):
    query: str
    search_raw_output: str
    urls_to_scrape: List[str]
    scraped_content: List[str]
    summaries: List[str]
    final_report: str
    error: str

def get_llm():
    return ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0)

def search_agent(state: ResearchState) -> ResearchState:
    query = state.get("query", "")
    logger.info(f"SEARCH AGENT: Searching for '{query}'")
    
    llm = get_llm()
    search_results = search_web_tool.invoke(query)
    
    if search_results.startswith("Error"):
        logger.error(f"Search tool returned error: {search_results}")
        return {"search_raw_output": search_results, "urls_to_scrape": [], "error": search_results}
    
    prompt = f"""You are a research assistant. Analyze these search results and extract up to 3 most relevant URLs to scrape.
Return ONLY a JSON array of string URLs. Do not include any other text or markdown formatting.
Search Results:
{search_results}
"""
    try:
        response = llm.invoke([SystemMessage(content=prompt)])
        content = response.content.strip()
        if content.startswith("```"):
            first_newline = content.find('\\n')
            if first_newline != -1:
                content = content[first_newline+1:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()
        
        urls = json.loads(content)
        if not isinstance(urls, list):
            urls = []
        logger.info(f"SEARCH AGENT: Extracted {len(urls)} URLs")
    except Exception as e:
        error_info = response.content if 'response' in locals() else str(e)
        logger.error(f"Error parsing JSON from search tool. Details: {error_info}")
        urls = []
        
    return {"search_raw_output": search_results, "urls_to_scrape": urls}

def scraper_agent(state: ResearchState) -> ResearchState:
    urls = state.get("urls_to_scrape", [])
    logger.info(f"SCRAPER AGENT: Scraping {len(urls)} URLs")
    
    scraped_content = []
    for url in urls:
        logger.info(f"Scraping URL: {url}")
        content = scrape_website_tool.invoke(url)
        if not "Error scraping" in content:
            scraped_content.append({"url": url, "content": content})
        else:
            logger.warning(f"Failed to scrape {url}: {content}")
            
    return {"scraped_content": scraped_content}

def summarizer_agent(state: ResearchState) -> ResearchState:
    scraped_content = state.get("scraped_content", [])
    logger.info(f"SUMMARIZER AGENT: Processing {len(scraped_content)} pages")
    
    llm = get_llm()
    summaries = []
    
    for item in scraped_content:
        prompt = f"""Summarize the following web page content. Extract the key facts, findings, and important details relevant to a research report. Keep it concise.
URL: {item['url']}
Content:
{item['content'][:8000]}
"""
        response = llm.invoke([SystemMessage(content=prompt)])
        summaries.append(f"Source: {item['url']}\\nSummary:\\n{response.content}\\n")
        
    return {"summaries": summaries}

def report_generator_agent(state: ResearchState) -> ResearchState:
    query = state.get("query", "")
    summaries = state.get("summaries", [])
    logger.info("REPORT GENERATOR AGENT: Synthesizing final report")
    
    error = state.get("error", "")
    if error:
        logger.error(f"Report generation aborted due to state error: {error}")
        return {"final_report": f"**Research failed due to an error during the process:**\\n\\n{error}"}
        
    llm = get_llm()
    
    if not summaries:
        search_raw = state.get("search_raw_output", "")
        logger.warning("No summaries available. Generating report from raw search output.")
        
        prompt = f"""You are an expert AI research assistant and technical writer.

Your task is to transform raw, unstructured search output into a clean, professional research report.

### Instructions:
1. Remove any irrelevant or unnecessary text.
2. Convert the content into a well-structured report with the following sections:
   * Title (based on the overall topic)
   * Introduction (2–3 lines summarizing the topic)
   * Key Trends / Findings (bullet points or numbered list)
   * Detailed Insights (expand each trend in 2–4 lines)
   * Conclusion (short summary of future outlook)
3. Use formal, professional language suitable for academic or industry reports.
4. Organize all links properly:
   * Present each source as:
     **Title**
     Short explanation
     (Optional: include link inline or as reference)
5. Merge similar ideas into cohesive points instead of repeating information.
6. Ensure clarity, readability, and logical flow.
7. Do NOT include raw snippets directly — rewrite them into meaningful insights.

### Input:
{search_raw}
"""
        response = llm.invoke([SystemMessage(content=prompt)])
        logger.info("Final report successfully generated from raw search.")
        return {"final_report": response.content}
        
    combined_summaries = "\\n---\\n".join(summaries)
    
    prompt = f"""You are an expert AI Research Assistant. Create a comprehensive, well-structured, and highly readable Markdown report based on the provided summaries.
Research Query: {query}

Constraints:
1. Use clear headings (##) and bullet points.
2. Synthesize the information logically into narrative sections.
3. INLINE CITATIONS: Use markdown links or explicit references to cite facts using the provided Source URLs.
4. Keep the tone professional and objective.
5. APPEND SOURCES: Always append a bold list of "### Sources" at the very end of the report containing all the URLs used.

Summaries:
{combined_summaries}
"""
    response = llm.invoke([SystemMessage(content=prompt)])
    logger.info("Final report successfully generated.")
    
    return {"final_report": response.content}
