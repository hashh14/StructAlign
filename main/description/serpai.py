import requests
import json
import os
from typing import Any, Dict


def fetch_serperdev_results(query: str, api_key: str = None, progress=None) -> Dict[str, Any]:
    """
    Fetch Google search results via the Serper.dev API.

    Args:
        query (str): The search query string.
        api_key (str, optional): Serper.dev API key. Falls back to the
            SERPERDEV_KEY environment variable if not provided.
        progress: Optional progress-bar object with an ``update(n)`` method.

    Returns:
        dict: Formatted search results with 'Input', 'Sources', and
              optionally 'knowledge_graph' keys.

    Raises:
        ValueError: If no API key is available.
        Exception: If the HTTP request fails.
    """
    if not api_key:
        api_key = os.environ.get("SERPERDEV_KEY")
        if not api_key:
            raise ValueError(
                "Serper.dev API key is required. "
                "Provide it as a parameter or set the SERPERDEV_KEY environment variable."
            )

    url = "https://google.serper.dev/search"
    payload = json.dumps({"q": query, "num": 10})
    headers = {
        "X-API-KEY": api_key,
        "Content-Type": "application/json",
    }

    response = requests.post(url, headers=headers, data=payload)

    if response.status_code != 200:
        raise Exception(f"Error fetching results: {response.status_code} - {response.text}")

    search_results = response.json()

    formatted_results: Dict[str, Any] = {"Input": query, "Sources": []}

    if "organic" in search_results:
        for result in search_results["organic"]:
            formatted_results["Sources"].append({
                "url": result.get("link", ""),
                "title": result.get("title", ""),
                "content": result.get("snippet", ""),
                "attr": result.get("attributes", ""),
            })

    if "knowledgeGraph" in search_results:
        formatted_results["knowledge_graph"] = search_results["knowledgeGraph"]

    if progress:
        progress.update(1)

    return formatted_results


if __name__ == "__main__":
    pass
