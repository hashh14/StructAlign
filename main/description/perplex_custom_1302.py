import json5
import json
import os
import logging
import traceback
import re
import time
from typing import Dict

import requests
from bs4 import BeautifulSoup
from langchain_ollama import OllamaLLM


def clean_json_response(response: str) -> str:
    """Strip markdown code fences from an LLM response to get raw JSON."""
    if "```json" in response:
        response = response.split("```json")[1]
    if "```" in response:
        response = response.split("```")[0]
    return response.strip()


def save_intermediate_results(data, dirname, source_idx):
    """Persist in-progress results to disk after each company is processed."""
    os.makedirs(dirname, exist_ok=True)
    output_file = os.path.join(dirname, "intermediate_results.json")
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logging.error(f"Error saving intermediate results: {e}")
        traceback.print_exc()


def save_final_results(data: Dict[str, str], filename: str):
    """Save the final results dictionary to a JSON file."""
    os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)
    try:
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logging.error(f"Error saving final results: {e}")
        traceback.print_exc()


def clean_text(text):
    """Collapse whitespace and strip leading/trailing spaces from extracted text."""
    if text:
        text = re.sub(r"\s+", " ", text)
        return text.strip()
    return ""


def scrape_and_structure_url(url):
    """
    Scrape paragraph text from a URL.

    Args:
        url (str): The URL to scrape.

    Returns:
        str | None: Extracted text joined by double newlines, or None on failure.
    """
    try:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/91.0.4472.124 Safari/537.36"
            )
        }
        response = requests.get(url, headers=headers, timeout=10)

        if response.status_code == 200:
            soup = BeautifulSoup(response.content, "html.parser")
            text_elements = []
            for element in soup.find_all("p"):
                if element.parent.name == "p":
                    continue
                text = clean_text(element.get_text())
                if text:
                    text_elements.append(text)
            return "\n\n".join(text_elements)

        print(f"Failed to retrieve the URL: {url}. Status code: {response.status_code}")
        return None

    except Exception as e:
        print(f"An error occurred with URL {url}: {e}")
        return None


def get_llm_summary(texts, answers, company_name):
    """
    Use a local Ollama LLM to produce a one-sentence company description.

    Args:
        texts (list[str]): Snippets / content from search result sources.
        answers (dict): Any direct answers returned by the search engine.
        company_name (str): The company being described.

    Returns:
        dict | None: Parsed JSON dict with 'description' and 'reason', or None on error.
    """
    llm = OllamaLLM(model="qwen2.5:7b-instruct-q8_0")

    descriptions_text = "\n\n".join(
        f"Company Description {i}:\n{text}" for i, text in enumerate(texts, 1)
    )

    prompt = f"""\
You are a helpful assistant who is tasked to retrieve a short and concise description about the company \
{company_name} from the following content extracted from a search engine:
It is EXTREMELY IMPORTANT to directly address only the description and ownership information about the organization:

{descriptions_text}

You must return the description along with the reasoning as a JSON object:

    ```json
    {{
        "description": "<retrieved description>",
        "reason": "<reason>"
    }}```

Guidelines:
    1. The summary must always be 1 sentence long.
    2. If the given descriptions are ambiguous or correspond to various organizations, return 'Not sure'.
    3. Always return output in English.
"""

    try:
        response = llm.invoke(prompt)
        print(response)
        return json5.loads(clean_json_response(response))
    except Exception as e:
        print(f"Error getting LLM summary: {e}")
        return None


def process_companies(input_file, output_file):
    """
    Process multiple companies from a JSON search-results file and generate
    one-sentence descriptions via a local LLM.

    Args:
        input_file (str): Path to JSON file produced by ``search_companies``.
        output_file (str): Path to write the final descriptions JSON.

    Returns:
        dict | None: Mapping of company name -> LLM summary dict, or None on error.
    """
    try:
        with open(input_file, "r", encoding="utf-8") as f:
            companies_data = json.load(f)

        results = {}
        intermediate_results = {}
        total_time = 0
        company_count = 0

        for company_name, company_info in list(companies_data.items()):
            start_time = time.time()
            print(f"\nProcessing company: {company_name}")

            answers = company_info.get("answers", {})

            sources = company_info.get("sources", [])
            if not sources and "desc" in company_info:
                sources = company_info["desc"].get("Sources", [])

            contents = []
            for source in sources:
                if source["url"].startswith("http"):
                    contents.append(source["title"] + source["content"])

            if not contents:
                print(f"No valid content found for {company_name}")
                continue

            summary = get_llm_summary(contents, answers, company_name)
            results[company_name] = summary
            intermediate_results[company_name] = summary

            save_intermediate_results(
                intermediate_results,
                os.path.dirname(output_file),
                len(intermediate_results),
            )

            if summary and "Not sure" in str(summary):
                print(f"Could not generate reliable summary for {company_name}")

            duration = time.time() - start_time
            total_time += duration
            company_count += 1
            print(f"Time taken to process {company_name}: {duration:.2f}s")
            print(f"Average time taken: {total_time / company_count:.2f}s")

        save_final_results(results, os.path.splitext(output_file)[0])
        print(f"\nResults have been saved to {output_file}")
        return results

    except Exception as e:
        print(f"An error occurred while processing companies: {e}")
        return None


if __name__ == "__main__":
    input_file = "company_links.json"
    output_file = "results/companies_summaries.json"
    process_companies(input_file, output_file)
