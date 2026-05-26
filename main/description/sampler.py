import pandas as pd
import random


def sample_accounts(input_csv: str, output_txt: str, sample_size: int):
    """
    Randomly sample company names from a CSV, excluding those starting with 'M',
    and append them to a text file.

    Args:
        input_csv (str): Path to input CSV file (must contain a
            'salesforce_account_name' column).
        output_txt (str): Path to the output text file (appended to, not overwritten).
        sample_size (int): Number of companies to sample.
    """
    try:
        df = pd.read_csv(input_csv, encoding="utf-8")

        filtered_companies = (
            df[~df["salesforce_account_name"].str.lower().str.startswith("m", na=False)][
                "salesforce_account_name"
            ].unique()
        )

        actual_sample_size = min(sample_size, len(filtered_companies))
        sampled_companies = random.sample(list(filtered_companies), actual_sample_size)

        with open(output_txt, "a", encoding="utf-8") as f:
            for company in sampled_companies:
                f.write(f"{company}\n")

        print(f"Successfully sampled and appended {actual_sample_size} companies to {output_txt}")

    except FileNotFoundError:
        print(f"Error: Could not find the input file {input_csv}")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    csv_file = "data/advertisers.csv"
    output_file = "data/unique_parents_sample.txt"
    n_samples = 5703

    sample_accounts(csv_file, output_file, n_samples)
