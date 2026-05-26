from description import search_companies, process_companies, hybrid_mapping_ollama, find_string_matches, process_json_files
from matching import brand_matcher
import json
import numpy as np
import os
from tree.tree import search_and_print_company, extract_names_from_tree

# DESCRIPTION FUNCTIONS

def d1(input1, output1):
    """Fetch web links and contents for a list of companies."""
    with open(input1, "r", encoding="utf-8") as file:
        companies = [comp.strip() for comp in file.readlines()]

    print("Length of companies list:", len(companies))
    search_companies(companies, output1)


def d2(input2, output2):
    """Generate descriptions using the local LLM model."""
    process_companies(input2, output2)


def d3(input3_1, input3_2):
    """Run semantic (embedding-based) matching between two company sets."""
    hybrid_mapping_ollama(
        map_from_path=input3_1,
        map_to_path=input3_2,
        similarity_threshold=0.6,
        top_k=10,
        model_name="nomic-embed-text:latest",
        save_results=True,
    )


def d4(input4_1, input4_2, output4_2):
    """Run string matching then merge with semantic matches for model input."""
    output4_1 = "temp/string_matches.json"

    find_string_matches(input4_1, input4_2, output4_1)

    process_json_files(
        string_matches_file=output4_1,
        source_file=input4_1,
        target_file=input4_2,
        desc_matches_file="temp/similarity_matches.json",
        output_file=output4_2,
    )


# MAPPER FUNCTIONS

def m1(input_file, output):
    """Transform model input format and run brand matching."""
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Flatten nested description dicts to plain description strings
    transformed_data = []
    for entry in data:
        transformed_entry = {"source": {}, "target": {}}
        for source_key, source_value in entry["source"].items():
            transformed_entry["source"][source_key] = source_value["description"]
        for target_key, target_value in entry["target"].items():
            transformed_entry["target"][target_key] = target_value["description"]
        transformed_data.append(transformed_entry)

    with open(input_file, "w", encoding="utf-8") as f:
        json.dump(transformed_data, f, indent=4, ensure_ascii=False)

    print("Transformation complete.")
    brand_matcher(input_file, output)


# LEAF PROCESSING UTILITIES

def transform_data(input_json, df1, df2):
    """Map advertisers to parent companies and their brand leaves."""
    with open(input_json, "r", encoding="utf-8") as f:
        advertiser_to_parents = json.load(f)

    result = []
    for advertiser, parents in advertiser_to_parents.items():
        if isinstance(parents, str):
            parents = [parents]

        advertiser_brands = df1[df1["Advertiser"] == advertiser]["Brand (Leaf)"].unique().tolist()

        for parent in parents:
            entry = {
                advertiser: parent,
                "details": {advertiser: advertiser_brands},
            }
            if parent not in ("Not sure", "No match"):
                entry["details"][parent] = df2[df2["PARENT"] == parent]["BRAND"].unique().tolist()
            else:
                entry["details"][parent] = []
            result.append(entry)

    with open("results/leaves_of_matched_parents_v2.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    return result


def extract_unique_names(json_file, output_file):
    """Extract and sort all unique brand names across all matched entries."""
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    all_names = set()
    for entry in data:
        for name_list in entry.get("details", {}).values():
            all_names.update(name_list)

    sorted_names = sorted(all_names)
    with open(output_file, "w", encoding="utf-8") as f:
        for name in sorted_names:
            f.write(f"{name}\n")

    print(f"Extracted {len(sorted_names)} unique names to {output_file}")
    return sorted_names


def extract_no_match_companies(input_file, output_file=None):
    """
    Extract company names mapped to 'No match' from a JSON results file.

    Args:
        input_file (str): Path to the JSON input file.
        output_file (str, optional): Path to write unmatched company names.

    Returns:
        list: Company names with no match.
    """
    try:
        with open(input_file, "r") as f:
            data = json.load(f)

        unmatched_companies = [company for company, value in data.items() if value == "No match"]

        if output_file is not None:
            with open(output_file, "w") as f:
                for company in unmatched_companies:
                    f.write(company + "\n")
            print(f"Successfully extracted {len(unmatched_companies)} unmatched companies to {output_file}")

    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found")
        return []
    except json.JSONDecodeError:
        print(f"Error: Unable to parse JSON from '{input_file}'")
        return []
    except Exception as e:
        print(f"Error: {str(e)}")
        return []

    return unmatched_companies


def extract_match_companies(input_file, output_file):
    """
    Extract company names that have a confirmed match from a JSON results file.

    Args:
        input_file (str): Path to the JSON input file.
        output_file (str): Path to write matched company names.
    """
    try:
        with open(input_file, "r") as f:
            data = json.load(f)

        matched_companies = [
            company for company, value in data.items()
            if value not in ("No match", "Not sure")
        ]

        with open(output_file, "w") as f:
            for company in matched_companies:
                f.write(company + "\n")

        print(f"Successfully extracted {len(matched_companies)} matched companies to {output_file}")

    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found")
    except json.JSONDecodeError:
        print(f"Error: Unable to parse JSON from '{input_file}'")
    except Exception as e:
        print(f"Error: {str(e)}")


def create_source_1_level_files(json_file_path, output_folder="source_1_levels"):
    """
    Split a JSON file of level-based company names into per-level text files.

    Args:
        json_file_path (str): Path to the JSON input file.
        output_folder (str): Folder to write the level files into.
    """
    try:
        with open(json_file_path, "r") as f:
            data = json.load(f)

        os.makedirs(output_folder, exist_ok=True)
        print(f"Created folder: {output_folder}")

        for level_key in data:
            level_num = level_key.split()[-1]
            output_file_path = os.path.join(output_folder, f"source_1_level_{level_num}")

            with open(output_file_path, "w") as f:
                for item in data[level_key]:
                    f.write(item + "\n")

            print(f"Created file: {output_file_path}")

        print(f"Successfully processed {len(data)} levels from {json_file_path}")

    except FileNotFoundError:
        print(f"Error: Input file '{json_file_path}' not found")
    except json.JSONDecodeError:
        print(f"Error: Unable to parse JSON from '{json_file_path}'")
    except Exception as e:
        print(f"Error: {str(e)}")


def get_company_description(name, file_path=None):
    """
    Look up a company's description from the description database.

    Args:
        name (str): Company name to look up.
        file_path (str, optional): Path to the description database JSON.

    Returns:
        dict: {name: {description: ...}} or an error dict.
    """
    if file_path is None:
        file_path = DESCRIPTION_DATABASE_FILE

    try:
        with open(file_path, "r", encoding="utf-8") as file:
            data = json.load(file)

        for entry in data:
            if entry["company"] == name:
                return {name: {"description": entry["description"]}}

        return {name: {"description": "Company not found."}}

    except FileNotFoundError:
        return {"error": "File not found."}
    except json.JSONDecodeError:
        return {"error": "Invalid JSON format in file."}


def filter_json(input_json_file):
    """Return only entries that have a confirmed match (not 'Not sure' or 'No match')."""
    with open(input_json_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {k: v for k, v in data.items() if v not in ("Not sure", "No match")}


def string_match_weighted(string1, string2):
    """Compute a weighted string similarity score between two strings."""

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


def find_shared_values(dictionary, key_list):
    """Find values that are shared across multiple keys in the dictionary."""
    value_to_keys = {}

    for key in key_list:
        if key in dictionary:
            value = dictionary[key]
            if isinstance(value, list):
                for item in value:
                    value_to_keys.setdefault(item, []).append(key)

    return {value: keys for value, keys in value_to_keys.items() if len(keys) > 1}


def safe_remove(lst, value, default="No match"):
    """Remove a value from a list and return the list, or default if it becomes empty."""
    if value in lst:
        lst.remove(value)
    return lst if lst else default


def get_max_scored_elements(D):
    """
    For each key, identify which candidate values score below the best match
    (i.e. the ones to remove in favour of the top scorer).
    """
    result = {}

    for key, values in D.items():
        if not values:
            continue

        scores = [
            (value, max(string_match_weighted(key, value), string_match_weighted(value, key)))
            for value in values
        ]
        max_score = max(scores, key=lambda x: x[1])[1]
        max_elements = [value for value, score in scores if score < max_score]

        best = [val for val in values if val not in max_elements]
        if len(best) > 1:
            print(best.pop(0))
            max_elements.extend(best)

        result[key] = max_elements

    return result


def postprocessing(data_copy):
    """
    Post-process matching output: normalise singleton lists, then resolve
    conflicts where multiple source companies map to the same target.
    """
    for key, value in data_copy.items():
        if value == ["No match"]:
            data_copy[key] = "No match"
        elif value == ["Not sure"]:
            data_copy[key] = "Not sure"

    mapping = find_shared_values(data_copy, list(data_copy.keys()))
    print(mapping)
    result = get_max_scored_elements(mapping)
    print(result)

    for key, values in result.items():
        for value in values:
            data_copy[value] = safe_remove(data_copy[value], key)

    print(data_copy)
    return data_copy


def source_2_source_1_matching():
    """
    Full end-to-end pipeline: match source_2 advertisers to source_1 parents
    by traversing source_1 tree levels, then recursively match their subtrees.
    """
    final_result = {}
    source_2_dict = {}

    print("Fetching source_2 data...")
    with open(source_2_COMPANIES_FILE, "r", encoding="utf-8") as file:
        for line in file:
            company_name = line.strip()
            print(company_name)
            result = get_company_description(company_name)
            source_2_dict.update(result)

    source_2_dict_filtered = source_2_dict
    all_output_no_match = {}

    level_files = [
        f for f in os.listdir(source_1_LEVELS_DIR)
        if os.path.isfile(os.path.join(source_1_LEVELS_DIR, f))
    ]

    for level in range(len(level_files)):
        filename = f"source_1_level_{level}"
        source_1_dict = {}

        print("Fetching source_1 data...")
        with open(os.path.join(source_1_LEVELS_DIR, filename), "r") as file:
            print(f"Processing {filename}")
            lines = file.readlines()

            if not lines:
                print(f"{filename} is empty. Skipping...")
                continue

            for line in lines:
                print(line)
                result = get_company_description(line.strip())
                source_1_dict.update(result)

        d3(source_2_dict_filtered, source_1_dict)
        d4(source_2_dict_filtered, source_1_dict, f"matching/levels/intermediate/model_input_l{level}.json")
        m1(f"matching/levels/intermediate/model_input_l{level}.json",
           f"matching/levels/intermediate/model_output_l{level}.json")

        with open(f"matching/levels/intermediate/model_output_l{level}.json", "r") as file:
            matched_data = json.load(file)

        all_output_no_match.update(matched_data)
        no_match_companies = extract_no_match_companies(
            f"matching/levels/intermediate/model_output_l{level}.json"
        )
        source_2_dict_filtered = {
            k: v for k, v in source_2_dict.items() if k in no_match_companies
        }

    final_result.update(all_output_no_match)

    with open("matching/levels/all_matches_no_match_traversal.json", "w", encoding="utf-8") as file:
        json.dump(all_output_no_match, file, indent=2, ensure_ascii=False)

    # --- Subtree matching for confirmed parent-level matches ---
    dict_with_matches = filter_json("matching/levels/all_matches_no_match_traversal.json")
    print("Extracted the dictionary with only matches...")

    all_output_match = {}

    for key, values in dict_with_matches.items():
        for value in values:
            print(f"Matching: {key} - {value} subtrees")

            source_2_subtree = search_and_print_company(
                source_2_NESTED_TREE, key, f"tree/{key}_source_2_tree.json", save_subtree=True
            )
            print("source_2 subtree extracted...")

            source_1_subtree = search_and_print_company(
                source_1_NESTED_TREE, value, f"tree/{value}_source_1_tree.json", save_subtree=True
            )
            print("source_1 subtree extracted...")

            source_2_matched_name_list = extract_names_from_tree(f"tree/{key}_source_2_tree.json")
            print("source_2 names extracted from subtree...")

            source_1_matched_name_list = extract_names_from_tree(f"tree/{value}_source_1_tree.json")
            print("source_1 names extracted from subtree...")

            source_2_matched_dict = {}
            source_1_matched_dict = {}

            for name in source_2_matched_name_list:
                source_2_matched_dict.update(get_company_description(name))
            print("source_2 description dictionary generated...")

            for name in source_1_matched_name_list:
                source_1_matched_dict.update(get_company_description(name))
            print("source_1 description dictionary generated...")

            print("Matching started...")
            intermediate_input = f"matching/levels/intermediate/model_input_{key}-{value}_matching.json"
            intermediate_output = f"matching/levels/intermediate/model_output_{key}-{value}_matching.json"

            d3(source_2_matched_dict, source_1_matched_dict)
            d4(source_2_matched_dict, source_1_matched_dict, intermediate_input)
            m1(intermediate_input, intermediate_output)
            print("Matching completed...")

            print("Starting postprocessing...")
            with open(intermediate_output, "r") as file:
                raw_data = json.load(file)

            output = postprocessing(raw_data)
            print("Postprocessing complete...")

            all_output_match.update(output)

    with open("matching/levels/all_matches_match_traversal.json", "w", encoding="utf-8") as file:
        json.dump(all_output_match, file, indent=2, ensure_ascii=False)
    print("matching/levels/all_matches_match_traversal.json saved...")

    final_result.update(all_output_match)

    with open("matching/levels/final_result.json", "w", encoding="utf-8") as file:
        json.dump(final_result, file, indent=2, ensure_ascii=False)
    print("Final result saved...")


if __name__ == "__main__":
    source_2_source_1_matching()
