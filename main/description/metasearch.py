import requests
import json
import time
from functools import partial
from concurrent.futures import ThreadPoolExecutor

from bs4 import BeautifulSoup
from tqdm import tqdm


def search_companies(companies_list, output_file=None, port=3242, max_workers=3, progress_bar=True):
    """
    Fetch web search results for a list of company names.

    Uses the Serper.dev API (via ``serpai.fetch_serperdev_results``) by default.
    A local SearXNG instance is also defined and can be swapped in if needed.

    Args:
        companies_list (list[str]): Company names to search for.
        output_file (str, optional): JSON file path to save results.
        port (int): Local SearXNG port (default 3242, unused with Serper.dev).
        max_workers (int): Thread-pool concurrency (default 3).
        progress_bar (bool): Show a tqdm progress bar (default True).

    Returns:
        dict: Mapping of company name -> search result dict.
    """

    def searxng(query, progress=None):
        """Search using a local SearXNG instance."""
        websearch_results = {}

        url = f"http://localhost:{port}/search?categories=general&language=en"
        payload = {"q": {query}, "engines": ["google", "bing"]}

        response = requests.post(url, params=payload)
        response_data = response.content.decode("utf-8")
        soup = BeautifulSoup(response_data, "html.parser")
        results = soup.find("div", id="results")

        if results:
            answers = results.find_all("div", id="answers")
            url_div = results.find("div", id="urls")

            if answers:
                answer = answers[0].find_all("div", class_="answer")
                text = answer[0].find("span").get_text()
                websearch_results["answer"] = [text]
            else:
                websearch_results["answer"] = []

            if url_div:
                websearch_results["sources"] = []
                articles = url_div.find_all("article", class_="result")
                articles = [
                    a for a in articles
                    if "qwant" not in a.find("div", class_="engines").get_text()
                ]
                urls = [a.find("a", class_="url_header", href=True)["href"] for a in articles]
                contents = [a.find("p", class_="content").get_text() for a in articles]
                titles = [a.find("h3").get_text() for a in articles]

                for link, content, title in zip(urls, contents, titles):
                    websearch_results["sources"].append({
                        "url": link,
                        "title": title,
                        "content": content.strip(),
                    })
                if not websearch_results["sources"]:
                    print(response_data, type(response_data))
            else:
                websearch_results["sources"] = [response_data]
        else:
            websearch_results["answer"] = []
            websearch_results["sources"] = [response_data]

        if progress:
            progress.update(1)
        return websearch_results

    def fetch_serperdev_results(query, progress=None):
        """Wrapper around the Serper.dev search function."""
        from .serpai import fetch_serperdev_results as _fetch
        result = _fetch(query)
        if progress:
            progress.update(1)
        return result

    progress = tqdm(total=len(companies_list), desc="Processing") if progress_bar else None

    serp_function = partial(fetch_serperdev_results, progress=progress)

    results = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            company: executor.submit(serp_function, f"{company} information and current owner")
            for company in companies_list
        }

    for company, future in futures.items():
        results[company] = {"desc": future.result()}

    if output_file:
        with open(output_file, "w") as json_file:
            json.dump(results, json_file, indent=2)

    if progress:
        progress.close()

    return results


def get_embedding(text, model="llama2"):
    """
    Get text embeddings from a local Ollama instance.

    Args:
        text (str): Text to embed.
        model (str): Ollama model name (default 'llama2').

    Returns:
        list: Embedding vector.
    """
    url = "http://localhost:11434/api/embeddings"
    data = {"model": model, "prompt": text}

    response = requests.post(url, json=data)
    if response.status_code == 200:
        return response.json()["embedding"]
    raise Exception(f"Error getting embedding: {response.status_code}")
