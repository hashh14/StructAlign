import json
from collections import defaultdict

import numpy as np
import faiss
from tqdm import tqdm
from langchain_ollama import OllamaEmbeddings
from fuzzywuzzy import fuzz


def load_json_file(file_path):
    """
    Load and parse a JSON file.

    Args:
        file_path (str): Path to the JSON file.

    Returns:
        dict | None: Parsed data, or None on error.
    """
    try:
        with open(file_path, "r") as file:
            return json.load(file)
    except FileNotFoundError:
        print(f"Error: File {file_path} not found")
        return None
    except json.JSONDecodeError:
        print(f"Error: File {file_path} contains invalid JSON")
        return None


def string_match(string1, string2):
    """Basic weighted token-overlap similarity between two strings."""
    string1 = string1.lower()
    string2 = string2.lower()

    split_1 = string1.split()
    split_2 = string2.split()

    weights = 1 / (2 ** (np.arange(0, len(split_1))))
    occurrence = np.array(list(map(lambda x: x in split_2, split_1)))
    matchless = set(split_2) - (set(split_1) & set(split_2))

    matchless_score = (len(split_1) + len(split_2) - len(matchless)) / (len(split_1) + len(split_2))
    match_score = np.sum(weights * occurrence) / np.sum(weights)
    match_score *= matchless_score

    return match_score


def string_match_improved(string1, string2):
    """
    Weighted token-overlap similarity with substring awareness.

    Tokens score positively if either string contains the other as a substring,
    rather than requiring exact equality.
    """
    def check_substring(x, l):
        return [x in elem or elem in x for elem in l]

    string1 = string1.lower()
    string2 = string2.lower()

    split_1 = string1.split()
    split_2 = string2.split()

    weights = 1 / (2 ** (np.arange(0, len(split_1))))
    occurrence_array = np.array(list(map(lambda x: check_substring(x, split_2), split_1)))
    occurrence = np.array(list(map(max, occurrence_array)))

    matchless = np.sum(occurrence_array, axis=0)
    matchless = matchless[matchless == 0]

    matchless_score = (len(split_1) + len(split_2) - (len(matchless) * 0.5)) / (len(split_1) + len(split_2))
    match_score = np.sum(weights * occurrence) / np.sum(weights)
    match_score *= matchless_score

    return match_score


def fuzzy_compute(string, list_of_strings):
    """Compute fuzzy (Levenshtein) ratio scores between a string and a list."""
    return np.array(list(map(lambda x: fuzz.ratio(string, x), list_of_strings)))


def fuzzy_string_match(string1, string2):
    """Weighted fuzzy token-level similarity between two strings."""
    string1 = string1.lower()
    string2 = string2.lower()

    split_1 = string1.split()
    split_2 = string2.split()

    weights = 1 / (2 ** (np.arange(0, len(split_1))))
    occurrence_array = np.array(list(map(lambda x: fuzzy_compute(x, split_2), split_1)))
    occurrence = np.array(list(map(max, occurrence_array)))
    print(occurrence)
    match_score = np.sum(weights * occurrence) / np.sum(weights)

    return match_score


def hybrid_mapping_ollama(
    map_from_path,
    map_to_path,
    similarity_threshold=0.6,
    top_k=5,
    model_name="nomic-embed-text:latest",
    save_results=True,
):
    """
    Perform hybrid semantic mapping between two company-description dictionaries
    using Ollama embeddings and FAISS similarity search.

    Args:
        map_from_path (str | dict): Path to source JSON file, or the dict itself.
        map_to_path (str | dict): Path to target JSON file, or the dict itself.
        similarity_threshold (float): Minimum similarity to count as a match (default 0.6).
        top_k (int): Number of top candidates to return per source entry (default 5).
        model_name (str): Ollama embedding model name (default 'nomic-embed-text:latest').
        save_results (bool): Whether to write results to disk (default True).

    Returns:
        dict | None: Mapping from source keys to lists of target keys, or None on error.
    """
    if isinstance(map_to_path, str) and isinstance(map_from_path, str):
        map_from_dict = load_json_file(map_from_path)
        map_to_dict = load_json_file(map_to_path)
    else:
        map_from_dict = map_from_path
        map_to_dict = map_to_path

    if map_from_dict is None or map_to_dict is None:
        return None

    ollama_embedder = OllamaEmbeddings(model=model_name)

    def get_ollama_embedding(text):
        if isinstance(text, dict) and "description" in text:
            text = text["description"]
        return np.array(ollama_embedder.embed_query(text))

    map_to_desc = list(map_to_dict.values())
    map_to_keys = list(map_to_dict.keys())

    to_embeddings = np.array(
        [get_ollama_embedding(desc) for desc in tqdm(map_to_desc, desc="Encoding 'To' Candidates")]
    ).astype("float32")

    dimension = to_embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(to_embeddings)

    mapping = defaultdict(list)
    unmatched_leaves = []

    for from_key, from_value in tqdm(map_from_dict.items(), desc="Computing similarity"):
        query_vector = get_ollama_embedding(from_value).reshape(1, -1).astype("float32")
        distances, indices = index.search(query_vector, len(map_to_dict))

        matches = [
            (map_to_keys[idx], 1 / (1 + np.sqrt(dist)))
            for dist, idx in zip(distances[0], indices[0])
        ]
        matches = sorted(matches, key=lambda x: x[1], reverse=True)

        final_matches = matches[:top_k]

        if final_matches:
            mapping[from_key] = [m[0] for m in final_matches]
        else:
            unmatched_leaves.append(from_key)

    if save_results:
        with open("temp/similarity_matches.json", "w") as file:
            json.dump(mapping, file, indent=4)
        print("Description-based matches stored to temp/similarity_matches.json")

        with open("unmatched_list.txt", "w") as file:
            file.write(json.dumps(unmatched_leaves))

    return mapping


if __name__ == "__main__":
    print(string_match_improved("SanTan Local Dealer", "SAN TAN FORD"))
