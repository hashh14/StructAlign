import json

import numpy as np
from fuzzywuzzy import fuzz
from tqdm import tqdm


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


def fuzzy_compute(string, list_of_strings):
    """
    Compute clamped fuzzy (Levenshtein) ratio scores.

    Scores below 50 are set to 0 to suppress low-quality partial matches.
    """
    scores = np.array(list(map(lambda x: fuzz.ratio(string, x), list_of_strings)))
    scores[scores < 50] = 0
    return scores


def fuzzy_string_match(string1, string2):
    """Weighted fuzzy token-level similarity between two strings."""
    string1 = string1.lower()
    string2 = string2.lower()

    split_1 = string1.split()
    split_2 = string2.split()

    weights = 1 / (2 ** (np.arange(0, len(split_1))))
    occurrence_array = np.array(list(map(lambda x: fuzzy_compute(x, split_2), split_1)))
    occurrence = np.array(list(map(max, occurrence_array)))
    match_score = np.sum(weights * occurrence) / np.sum(weights)

    return match_score


def string_match_improved(string1, string2):
    """
    Weighted token-overlap similarity with substring awareness.

    Tokens score positively if either string contains the other as a substring.
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


def string_match_weighted(string1, string2):
    """
    Weighted token similarity using proportional substring length scoring.

    Unlike ``string_match_improved``, partial substring overlaps are scored
    proportionally by the ratio of matched-to-total characters.
    """
    def check_substring(x, l):
        return [
            len(elem) / len(x) if elem in x else (len(x) / len(elem) if x in elem else 0)
            for elem in l
        ]

    string1 = string1.lower()
    string2 = string2.lower()

    split_1 = string1.split()
    split_2 = string2.split()

    weights = 1 / (2 ** (np.arange(0, len(split_1))))
    occurrence_array = np.array(list(map(lambda x: check_substring(x, split_2), split_1)))
    occurrence = np.array(list(map(sum, occurrence_array)))

    matchless = np.sum(occurrence_array, axis=0)
    matchless = matchless[matchless == 0]

    matchless_score = (len(split_1) + len(split_2) - (len(matchless) * 0.5)) / (len(split_1) + len(split_2))
    match_score = np.sum(weights * occurrence) / np.sum(weights)
    match_score *= matchless_score

    return match_score


def find_string_matches(file1_path, file2_path, output_path):
    """
    Find the top-5 string matches for each key in file1 against keys in file2,
    and write the results to a JSON file.

    Args:
        file1_path (str | dict): Source JSON file path or dict.
        file2_path (str | dict): Target JSON file path or dict.
        output_path (str): Path to write the match results JSON.

    Returns:
        dict | None: Mapping of source key -> list of top-5 target keys.
    """
    ignore_words = ["&", "Co", "Inc", "LLC", "of", "corp", "group"]

    def load_json_file(file_path):
        try:
            with open(file_path, "r") as file:
                return json.load(file)
        except FileNotFoundError:
            print(f"Error: File {file_path} not found")
            return None
        except json.JSONDecodeError:
            print(f"Error: File {file_path} contains invalid JSON")
            return None

    def find_matching_keys(file1_dict, file2_dict, ignore_words=None):
        result = {}
        for key1 in tqdm(file1_dict):
            matches_scores = [
                (key2, max(string_match_weighted(key1, key2), string_match_weighted(key2, key1)))
                for key2 in file2_dict
            ]
            matches_scores = sorted(matches_scores, key=lambda x: x[1], reverse=True)
            matches = [elem[0] for elem in matches_scores[:5]]
            if matches:
                result[key1] = matches
        return result

    def write_output_to_json(output_dict, output_file):
        try:
            with open(output_file, "w") as file:
                json.dump(output_dict, file, indent=4)
            return True
        except Exception as e:
            print(f"Error writing to output file: {e}")
            return False

    if isinstance(file1_path, str) and isinstance(file2_path, str):
        file1 = load_json_file(file1_path)
        file2 = load_json_file(file2_path)
    else:
        file1 = file1_path
        file2 = file2_path

    if not file1 or not file2:
        return None

    result = find_matching_keys(file1, file2, ignore_words)
    if write_output_to_json(result, output_path):
        print(f"Results successfully written to {output_path}")
    return result


def find_string_matches_altered(file1_dict, file2_dict, ignore_words=None):
    """
    Variant of ``find_string_matches`` that operates directly on dicts
    and also returns the raw scores for the last processed key.

    Args:
        file1_dict (dict): Source company descriptions.
        file2_dict (dict): Target company descriptions.
        ignore_words (list, optional): Words to ignore (currently unused).

    Returns:
        tuple[dict, list]: (matches dict, scores for the last key processed)
    """
    result = {}
    matches_scores = []
    for key1 in tqdm(file1_dict):
        matches_scores = [
            (key2, string_match_weighted(key1, key2)) for key2 in file2_dict
        ]
        matches_scores = sorted(matches_scores, key=lambda x: x[1], reverse=True)[:5]
        matches = [elem[0] for elem in matches_scores]
        if matches:
            result[key1] = matches

    return result, matches_scores


if __name__ == "__main__":
    print(string_match_weighted("SanTan Local Dealer", "SAN TAN FORD"))
