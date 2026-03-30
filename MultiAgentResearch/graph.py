from langgraph.graph import StateGraph, END
from agents import (
    ResearchState,
    search_agent,
    scraper_agent,
    summarizer_agent,
    report_generator_agent
)

def build_graph():
    # Initialize the state graph
    workflow = StateGraph(ResearchState)
    
    # Add nodes representing the agents
    workflow.add_node("search", search_agent)
    workflow.add_node("scrape", scraper_agent)
    workflow.add_node("summarize", summarizer_agent)
    workflow.add_node("report", report_generator_agent)
    
    # Define the execution flow (edges)
    workflow.set_entry_point("search")
    workflow.add_edge("search", "scrape")
    workflow.add_edge("scrape", "summarize")
    workflow.add_edge("summarize", "report")
    workflow.add_edge("report", END)
    
    # Compile the graph
    app = workflow.compile()
    return app

# Expose a pre-built instance for easy importing
research_graph = build_graph()
