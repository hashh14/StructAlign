import json
import os

from description import search_companies, process_companies


def d1(input1, output1):
    """Fetch web links and contents for a list of companies from a text file."""
    with open(input1, "r", encoding="utf-8") as file:
        companies = [comp.strip() for comp in file.readlines()]

    print("Length of companies list:", len(companies))
    search_companies(companies, output1)


def d2(input2, output2):
    """Generate descriptions using the local LLM model."""
    process_companies(input2, output2)


def generate_descriptions(input_txt, database_file="description/results/description_database.json"):
    """
    Check which companies in a text file are missing from the description database,
    fetch their web data, and generate LLM descriptions for them.

    Args:
        input_txt (str): Path to a text file with one company name per line.
        database_file (str): Path to the existing description database JSON.

    Returns:
        bool: True if new descriptions were generated, False if all were already present.
    """
    output_txt = "temp/no_desc_companies.txt"
    os.makedirs("temp", exist_ok=True)

    with open(database_file, "r", encoding="utf-8") as file:
        description_data = json.load(file)

    existing_companies = {entry["company"] for entry in description_data}

    with open(input_txt, "r", encoding="utf-8") as file:
        company_names = {line.strip() for line in file}

    missing_companies = company_names - existing_companies
    print(f"Number of companies in the input:              {len(company_names)}")
    print(f"Number of companies in description_database:   {len(existing_companies)}")
    print(f"Number of companies needing descriptions:       {len(missing_companies)}")

    if not missing_companies:
        return False

    with open(output_txt, "w", encoding="utf-8") as file:
        for company in sorted(missing_companies):
            file.write(company + "\n")

    print(f"Missing company names saved to '{output_txt}'.")

    d1("temp/no_desc_companies.txt", "temp/no_desc_companies.json")
    d2("temp/no_desc_companies.json", "temp/no_desc_companies_output.json")

    return True


def modify_and_append_json(
    input_file,
    output_file="description/results/description_database.json",
    source_label="unknown",
    column_label="unknown",
):
    """
    Parse an LLM-output JSON file and append the entries to the description database.

    Args:
        input_file (str): Path to the LLM output JSON (company -> {description, reason}).
        output_file (str): Path to the description database to update.
        source_label (str): Value to store in the 'file' field of each entry.
        column_label (str): Value to store in the 'column' field of each entry.
    """
    existing_data = []
    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as f:
            try:
                existing_data = json.load(f)
            except json.JSONDecodeError:
                existing_data = []

    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not data:
        print("No data to append.")
        return

    modified_entries = []
    print(f"Number of companies being added: {len(data)}")
    for company, details in data.items():
        modified_entries.append({
            "company": company,
            "description": details["description"],
            "reason": details["reason"],
            "file": source_label,
            "column": column_label,
        })

    existing_data.extend(modified_entries)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(existing_data, f, indent=4, ensure_ascii=False)

    print(f"Modified data successfully appended to {output_file}")


if __name__ == "__main__":
    # Example: update the database for a new input file
    # Set DATABASE_FILE and SOURCE_LABEL/COLUMN_LABEL as appropriate for your run.
    DATABASE_FILE = "description/results/description_database.json"
    INPUT_TXT = "path/to/your/companies.txt"

    output = generate_descriptions(INPUT_TXT, database_file=DATABASE_FILE)

    if output:
        modify_and_append_json(
            "temp/no_desc_companies_output",
            output_file=DATABASE_FILE,
            source_label="source_1",
            column_label="BRAND",
        )
    else:
        print("No data to append.")
