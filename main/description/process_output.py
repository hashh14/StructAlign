import json
from collections import defaultdict


def load_json_file(filename):
    """Load and return JSON data from a file."""
    with open(filename, "r") as f:
        return json.load(f)


def join_mappings(string_matches, desc_matches):
    """
    Merge string-match results into the description-match dictionary.

    For keys present in both, the description-match values take precedence
    (the union is computed but stored under the desc_matches key).

    Args:
        string_matches (dict): Output of string-based matching.
        desc_matches (dict): Output of semantic/description-based matching.

    Returns:
        dict: Updated desc_matches with string-match candidates folded in.
    """
    for key, value in string_matches.items():
        if key in desc_matches:
            value = set(value)
            value_to_join = set(desc_matches[key])
            value.union(value_to_join)
            desc_matches[key] = list(value_to_join)
    return desc_matches


def process_mappings(mapping_dict, source_dict, target_dict):
    """
    Build the structured model-input list from merged mapping results.

    Args:
        mapping_dict (dict): Merged mapping of source key -> list of target keys.
        source_dict (dict): Source company description data.
        target_dict (dict): Target company description data.

    Returns:
        list[dict]: Each element has 'source' and 'target' sub-dicts.
    """
    results = defaultdict(dict)

    for key, value_list in mapping_dict.items():
        if not isinstance(value_list, list):
            value_list = [value_list]

        for value in value_list:
            if key in source_dict and value in target_dict:
                if key not in results:
                    results[key] = {
                        "source": {key: source_dict[key]},
                        "target": {},
                    }
                results[key]["target"][value] = target_dict[value]
            else:
                print(f"Warning: Key '{key}' or value '{value}' not found in respective files")

    return list(results.values())


def process_json_files(string_matches_file, source_file, target_file, desc_matches_file, output_file):
    """
    Merge string-match and semantic-match results, then write the combined
    model-input JSON file.

    Args:
        string_matches_file (str): Path to string-match JSON output.
        source_file (str | dict): Path to source descriptions JSON, or the dict itself.
        target_file (str | dict): Path to target descriptions JSON, or the dict itself.
        desc_matches_file (str): Path to semantic-match JSON output.
        output_file (str): Path to write the combined model-input JSON.
    """
    try:
        if isinstance(string_matches_file, str) and isinstance(source_file, str):
            file2_data = load_json_file(source_file)
            file3_data = load_json_file(target_file)
        else:
            file2_data = source_file
            file3_data = target_file

        file1_data = load_json_file(string_matches_file)
        file4_data = load_json_file(desc_matches_file)

        merged_dict = join_mappings(file1_data, file4_data)
        results = process_mappings(merged_dict, file2_data, file3_data)

        with open(output_file, "w") as f:
            json.dump(results, f, indent=4)

        print(f"Processed data saved to {output_file}")

    except FileNotFoundError as e:
        print(f"Error: Could not find one of the input files - {e}")
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in one of the input files - {e}")
    except Exception as e:
        print(f"Unexpected error occurred: {e}")
