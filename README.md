# source_2–source_1 Brand Matching Pipeline

A multi-stage NLP pipeline that maps **source_2** advertiser/brand names to their
counterparts in the **source_1** advertising database.  
Matching is performed by combining web-scraped company descriptions, semantic
embeddings, weighted string similarity, and a local LLM classifier.

---

## How It Works

```
source_2 companies        source_1 tree levels
        │                           │
   ┌────▼────┐                 ┌────▼────┐
   │  d1/d2  │  web search +   │  d1/d2  │  web search +
   │         │  LLM summaries  │         │  LLM summaries
   └────┬────┘                 └────┬────┘
        │                           │
        └──────────┬────────────────┘
                   │
            ┌──────▼──────┐
            │  d3 (FAISS) │  semantic matching 
            └──────┬──────┘
                   │
            ┌──────▼──────┐
            │  d4 (string)│  weighted string matching 
            └──────┬──────┘
                   │
            ┌──────▼──────┐
            │  m1 (LLM)   │  LLM classifier
            └──────┬──────┘
                   │
            ┌──────▼──────┐
            │postprocessing│  conflict resolution for shared targets
            └──────┬──────┘
                   │
            ┌──────▼──────┐
            │ subtree     │  recursively match brand leaves within
            │ matching    │  confirmed parent pairs
            └─────────────┘
```

---



## Prerequisites

| Requirement | Notes |
|---|---|
| Python 3.10+ | Tested on 3.11 |
| [Ollama](https://ollama.com) | Must be running locally (`ollama serve`) |
| `nomic-embed-text` model | `ollama pull nomic-embed-text` |
| `qwen2.5:7b-instruct-q8_0` | `ollama pull qwen2.5:7b-instruct-q8_0` (descriptions) |
| `qwen2.5:32b-instruct-q8_0` | `ollama pull qwen2.5:32b-instruct-q8_0` (matching) |

---

## Setup

```bash
# 1. Clone the repository
git clone https://github.com/hashh14/StructAlign.git

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt


## Running the Pipeline

```bash
cd main
python main.py
```

The pipeline will:
1. Fetch descriptions for all source_2 companies (web search + LLM).
2. Iterate through source_1 tree levels, fetching descriptions and running matching.
3. For each confirmed parent match, recursively match the brand leaves.

### Keeping the description database current

```bash
python description_database.py
```

This checks which companies in your input file are missing from the database,
fetches their descriptions, and appends them.

---

