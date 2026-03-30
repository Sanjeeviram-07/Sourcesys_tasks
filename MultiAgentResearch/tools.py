import os
import json
import logging
import requests
from requests.exceptions import RequestException, Timeout
from bs4 import BeautifulSoup
from langchain_core.tools import tool

# Configure basic logging for tools
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

@tool
def search_web_tool(query: str) -> str:
    """
    Search the web using the Google Serper API.
    Provides snippets and URLs from the top search results.
    """
    api_key = os.getenv("SERPER_API_KEY")
    if not api_key:
        logger.error("SERPER_API_KEY is not set.")
        return "Error: SERPER_API_KEY is not set. Please configure your environment variables."

    url = "https://google.serper.dev/search"
    # Limit to top 5 results to keep the prompt context concise
    payload = json.dumps({"q": query, "num": 5})
    headers = {
        'X-API-KEY': api_key,
        'Content-Type': 'application/json'
    }

    try:
        response = requests.post(url, headers=headers, data=payload, timeout=10)
        response.raise_for_status()
        results = response.json()
        
        snippets = []
        if "organic" in results:
            for item in results["organic"]:
                snippets.append(f"Title: {item.get('title')}\\nLink: {item.get('link')}\\nSnippet: {item.get('snippet')}")
        
        if not snippets:
            logger.info(f"No organic search results found for query: {query}")
            return "No relevant search results found."
        
        return "\\n\\n".join(snippets)
    except Timeout:
        logger.error(f"Search API timeout for query: {query}")
        return "Error performing web search: The request to the Serper API timed out."
    except RequestException as e:
        logger.error(f"Search API request failed: {e}")
        return f"Error performing web search: {str(e)}"
    except Exception as e:
        logger.exception("Unexpected error in search_web_tool")
        return f"Unexpected error during search: {str(e)}"

@tool
def scrape_website_tool(url: str) -> str:
    """
    Scrapes the text content of a given webpage URL using BeautifulSoup.
    Handles invalid links, redirects, and timeouts gracefully.
    """
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5"
        }
        
        response = requests.get(url, headers=headers, timeout=15, allow_redirects=True)
        response.raise_for_status()
        
        # Determine the encoding to avoid text corruption
        response.encoding = response.apparent_encoding
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Remove noisy elements that don't add value to the content analysis
        noisy_tags = ["script", "style", "nav", "footer", "header", "aside", "form", "noscript"]
        for element in soup(noisy_tags):
            element.decompose()
            
        # Extract meaningful text, separating semantic blocks with spaces
        text = soup.get_text(separator=' ', strip=True)
        
        # Remove excessive whitespace
        text = ' '.join(text.split())
        
        # Limit the text length to avoid token limit issues (8000 chars roughly equals 2000 tokens)
        max_length = 8000
        if len(text) > max_length:
            logger.info(f"Truncated scraped content for {url} (original size: {len(text)})")
            text = text[:max_length] + "... [Content Truncated]"
            
        return text
    except Timeout:
        logger.error(f"Scraping timeout for URL: {url}")
        return f"Error scraping {url}: The request timed out."
    except RequestException as e:
        logger.error(f"Scraping request failed for URL {url}: {e}")
        return f"Error scraping {url}: Unable to access the page ({str(e)})."
    except Exception as e:
        logger.exception(f"Unexpected error scraping {url}")
        return f"Unexpected error scraping {url}: {str(e)}"
