from duckduckgo_search import DDGS
from typing import List, Dict

def web_search(query: str, num_results: int = 5) -> List[Dict[str, str]]:
    """
    Performs a web search using DuckDuckGo.
    
    Args:
        query (str): The search query.
        num_results (int): The number of results to return.
        
    Returns:
        List[Dict[str, str]]: A list of dictionaries containing the title, link, and snippet of each result.
    """
    results = []
    try:
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=num_results):
                results.append({
                    "title": r.get("title", ""),
                    "link": r.get("href", ""),
                    "snippet": r.get("body", "")
                })
    except Exception as e:
        print(f"Error during web search: {e}")
        return []
    
    return results
