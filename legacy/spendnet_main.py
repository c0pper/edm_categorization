import os
from langsmith import Client
from tqdm import tqdm
from Categorizer import Categorizer, RelevanceEvaluator
from dotenv import load_dotenv
import pandas as pd

load_dotenv()

###### SPEND NETWORK CATEGORIZATION
# categorizer = Categorizer(mode="cat")

# categorizer.run_on_dataset("SN_data_subset", categorizer.ask_chatgpt)
# categorizer.run_on_dataset("SN_data_subset", categorizer.ask_elmib)

# categorizer.run_on_dataset(dataset_name="SN_data", eval_prompt_template=categorizer.EVAL_PROMPT_TEMPLATE_CATEGORIZATION, llm_or_chain_factory=categorizer.ask_chatgpt)
# categorizer.run_on_dataset(dataset_name="SN_data", eval_prompt_template=categorizer.EVAL_PROMPT_TEMPLATE_CATEGORIZATION, llm_or_chain_factory=categorizer.ask_elmib)



from spendnet_platform_library_creator import df, unique_cats
from langfuse import Langfuse

df["title_desc"] = "TITLE: " + df["title"] + " ----- " + "DESCRIPTION: " + df["description"]
print(df["title_desc"][0])
print(len(df["category"].unique()))
print(len(unique_cats))
# df.to_csv("data/SN_20k_balanced_forgenai.csv", index=False, encoding="utf8")

langfuse = Langfuse(
    secret_key=os.getenv("LANGFUSE_PRIVATE"),
    public_key=os.getenv("LANGFUSE_PUBLIC"),
    host="https://cloud.langfuse.com"
)

MODE = "cat"

# Create dataset
dataset_name = "SN20k"
# langfuse.create_dataset(name=dataset_name)

# df = df.head(10)

# # Add elements to dataset
# rows = df.iterrows()
# for index, row in tqdm(rows):
#     # Create a dataset item using langfuse.create_dataset_item()
#     langfuse.create_dataset_item(
#         dataset_name=dataset_name,
#         input={"text": row["title_desc"]}, 
#         expected_output={"text": row["category"]}  
#     )


dataset = langfuse.get_dataset(dataset_name)
categorizer = Categorizer(mode=MODE)
evaluator = RelevanceEvaluator(mode=MODE)
for item in tqdm(dataset.items):
    # execute application function and get Langfuse parent observation (span/generation/event)
    # output also returned as it is used to evaluate the run
    output = categorizer.ask_chatgpt(item.input, unique_cats)["output"]
    generation = langfuse.generation(
        name="testgeneration",
        input=item.input,
        output=output
    )
 
    # link the execution trace to the dataset item and give it a run_name
    item.link(generation, "<run_name>")
 
    # optionally, evaluate the output to compare different runs more easily
    evaluation = evaluator._evaluate_strings(
            input=item.input,
            prediction=output,
            reference=item.expected_output
        )
    generation.score(
        name="<example_eval>",
        # any float value
        value=evaluation["score"],
        comment=evaluation["explaination"]
    )

# client = Client()

# csv_file = 'data/SN_20k_balanced_forgenai.csv'
# input_keys = ['title_desc'] # replace with your input column names
# output_keys = ['category'] # replace with your output column names

# dataset = client.upload_csv(
#     csv_file=csv_file,
#     input_keys=input_keys,
#     output_keys=output_keys,
#     name="My CSV Dataset",
#     description="Dataset created from a CSV file",
#     data_type="kv"
# )
# categorizer = Categorizer(mode="cat")
