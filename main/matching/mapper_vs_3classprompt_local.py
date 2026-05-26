import json
import json5
import os
import logging
import re
import time
import traceback
from typing import Dict, List
from langchain_ollama import OllamaLLM
from tqdm import tqdm


def brand_matcher(input_file: str, output_file: str) -> Dict[str, str]:
    """
    Match source brands to target brands using a local Ollama LLM.

    For each (source, candidate-targets) pair in the input file the LLM is asked
    to decide whether each candidate is a match, not a match, or uncertain.
    Results are saved incrementally to guard against crashes on long runs.

    Args:
        input_file (str): Path to the model-input JSON produced by ``process_json_files``.
        output_file (str): Path to write the final matching results JSON.

    Returns:
        dict: Mapping of source brand name -> LLM classification result.
    """

    def save_intermediate_results(data, dirname, source_idx):
        """Persist in-progress results after each brand is processed."""
        os.makedirs(dirname, exist_ok=True)
        intermediate_path = os.path.join(dirname, "intermediate_results.json")
        try:
            with open(intermediate_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logging.error(f"Error saving intermediate results: {e}")
            traceback.print_exc()

    def save_final_results(data: Dict[str, str], filename: str):
        """Write the final results to disk."""
        os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)
        try:
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logging.error(f"Error saving final results: {e}")
            traceback.print_exc()

    def clean_json_response(response: str) -> str:
        """Strip markdown code fences from an LLM response."""
        if "```json" in response:
            response = response.split("```json")[1]
        if "```" in response:
            response = response.split("```")[0]
        return response.strip()

    def process_brand_matching(input_data: List[Dict]) -> Dict[str, str]:
        """Run the LLM over each (source, targets) entry and collect results."""
        llm = OllamaLLM(model="qwen2.5:32b-instruct-q8_0", temperature=0.65)
        results = {}

        for idx, entry in tqdm(enumerate(input_data), total=len(input_data)):
            source_brand = list(entry["source"].keys())[0]
            source_description = entry["source"][source_brand]
            target_brands_with_descriptions = entry["target"]

            prompt = f"""\
You are a brand matching expert. You will be given:
    1. A single brand name to match with its description
    2. A list of potential matching brands with their descriptions

Your task is to check if there are any appropriate matching brand names in the given list.
The brand names are a match if they are the same company.
The brand names are not a match if they are not the same company.
Assign "Not sure" if there is not enough certainty for a match or no match.

brand: {source_brand.lower()}
Description: {source_description}

List of brands to match against with their descriptions:
{json.dumps(target_brands_with_descriptions, indent=2)}

Always return every response in this json format:

```json
{{
"{source_brand.lower()}": ["brand 1", "brand 2", ...] or "No match" or "Not sure"
}}
```

Guidelines:
    1. While mapping, note that brand names can have variations, abbreviations, and different nomenclatures.
    2. **ONLY** use the brand names provided in the given list. Do not modify the existing names.
"""

            try:
                response = llm.invoke(prompt)
                with open("complete_response.txt", "a+") as f:
                    f.write(response + "\n\n")

                cleaned_response = clean_json_response(response)
                try:
                    result = json5.loads(cleaned_response)
                    results[source_brand] = list(result.values())[0]
                except json.JSONDecodeError as e:
                    logging.error(f"JSON parsing error for {source_brand}: {e}")
                    traceback.print_exc()
                    logging.error(f"Raw response: {response}")

            except Exception as e:
                logging.error(f"Error processing {source_brand}: {e}")
                traceback.print_exc()

            save_intermediate_results(results, "matching/results", idx)

        save_final_results(results, output_file)
        return results

    def preprocess_citations(data: List[Dict]) -> List[Dict]:
        """
        Remove inline citation markers (e.g. [1], [4][5]) from description strings.

        Args:
            data: List of source/target dicts from the model-input file.

        Returns:
            Cleaned copy of the input list.
        """
        def clean_text(text):
            return re.sub(r"(\[\d+\])+\.$", "", text).strip()

        return [
            {
                "source": {k: clean_text(v) for k, v in item["source"].items()},
                "target": {k: clean_text(v) for k, v in item["target"].items()},
            }
            for item in data
        ]

    start_time = time.time()

    try:
        with open(input_file, "r", encoding="utf-8") as f:
            input_data = json.load(f)
    except Exception as e:
        print(f"Error reading input file: {e}")
        traceback.print_exc()
        return {}

    input_data = preprocess_citations(input_data)
    results = process_brand_matching(input_data)

    print(f"Execution time: {time.time() - start_time:.4f} seconds")
    return results
