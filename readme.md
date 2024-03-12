# Categorizer Python Project

## Overview

The Categorizer Python project is a tool designed to categorize text based on predefined categories. It leverages the LangChain platform, OpenAI's GPT-3.5-turbo model, and an external API (ElmiLab) to achieve accurate and efficient categorization. The project includes a main script, `Categorizer.py`, and a driver script, `main.py`.

## Categorizer.py

### Dependencies

- Python 3.x
- Requests library
- LangChain Core
- LangChain OpenAI
- LangSmith
- OpenAI GPT-3.5-turbo
- Python-dotenv

### Environment Variables

Before running the script, ensure the following environment variables are set:

```python
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Enrich My Data"
# Add OpenAI and LangSmith API keys to .env
```

## Usage

The `Categorizer` class within `Categorizer.py` provides functionalities to categorize text using both Expert.AI Elmi LLM and OpenAI's GPT-3.5-turbo model. The script includes:

- Building prompts for categorization.
- Interaction with the ElmiLab API for additional categorization options.
- Integration with OpenAI's GPT-3.5-turbo model for categorization.

The `run_on_dataset` method initiates the categorization process on a specified dataset, allowing users to choose between different categorization methods and evaluates the outputs using the eval_configuration of the categorizer object.

## main.py

### Dependencies

- Python 3.x
- Python-dotenv
- Categorizer.py

### Usage

The `main.py` script serves as the driver for the Categorizer project. It includes:

- Loading environment variables using dotenv.
- Instantiating the `Categorizer` class.
- Running the categorization process on specific datasets using different methods (`ask_chatgpt` and `ask_elmib`).

## Running the Project

1. Set the required environment variables in `Categorizer.py`.
2. Ensure dependencies are installed (`pip install -r requirements.txt`).
3. Run `main.py` to initiate the categorization process.


