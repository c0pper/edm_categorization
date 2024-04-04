import csv
from datetime import datetime
import functools
import hashlib
import json
from uuid import uuid4
from langfuse import Langfuse
from openai import OpenAI
import pandas as pd
from tqdm import tqdm
from Categorizer import Categorizer, RelevanceEvaluator
from ragas.metrics import (
    answer_relevancy,
    faithfulness,
    context_recall,
    context_precision,
)
from datasets import Dataset, Features, Value, Sequence
from langchain.output_parsers import PydanticOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field, validator
from typing import List
from ragas import evaluate
from dotenv import load_dotenv
import os

load_dotenv()

df = pd.read_csv("data/SN_20k.csv", encoding="UTF8")
categories = df["category"].unique()
cache_path = "cache/sn_llm_cat_cache"

class Category(BaseModel):
    output_categories: List[str]  = Field(description="list of categories in which the offer has been categorized")
    explanation: str  = Field(description="explanation of the reason why the text was categorized this way")

    # You can add custom validation logic easily with Pydantic.
    @validator("output_categories")
    def is_in_categories(cls, field:list):
        for cat in field:
            if cat not in categories:
                raise ValueError(f"Category '{cat}' not present in original categories!")
        return field


def create_trace(response_dict, run_id):
    trace = langfuse.trace(
        session_id=run_id,
        input=response_dict["question"],
        output=response_dict["output"]
    )

    return trace


def create_generation(response_dict, trace, db_item, parent_observation_id=None, run_id=None):
    # Create generation in Langfuse
    generation_kwargs = {
        "name": f'{response_dict["llm"]}_generation',
        "model": response_dict["llm"],
        "input": response_dict["prompt_str"],
        "metadata": {"llm": response_dict["llm"], "llm_config": response_dict["llm_config"]},
        "start_time": response_dict["start_time"],
        "end_time": response_dict["end_time"],
        "prompt": response_dict["prompt_obj"],
        "trace_id": trace.id
    }
    
    # Conditionally include parent_generation if not None
    if parent_observation_id is not None:
        generation_kwargs["parent_observation_id"] = parent_observation_id
        

    generation = langfuse.generation(**generation_kwargs)
    generation.end(output=response_dict["output"])

    db_item.link(generation, run_id)
    langfuse.flush()
    
    return generation


def retry_until_output_categories_present(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        response = {"output": "{}"}
        valid_json = False
        while not valid_json:
            try:
                json_data = json.loads(response["output"])
                if "output_categories" in json_data.keys():
                    valid_json = True
            except json.JSONDecodeError:
                print(f'invalid json\n{response["output"]}')  # Ignore JSONDecodeError and continue retrying
            
            response = func(*args, **kwargs)
            response["output"] = response["output"].replace("`", "").replace("json","")
        return response
    return wrapper

@retry_until_output_categories_present
def invoke_llm(client, prompt_obj, prompt_dictionary, model="gpt-3.5-turbo", llm_config=None):
    generation_start_time = datetime.now()
    if llm_config is None:
        llm_config = {}

    # Get prompt object
    prompt_str = prompt_obj.compile(**prompt_dictionary)

    # Check cache
    md5_hash = hashlib.md5(prompt_dictionary["offer"].encode()).hexdigest()
    cache_filename = f"cache/sn_llm_cat_cache/{md5_hash}.cache"

    # Check if file exists in cache
    if os.path.exists(cache_filename):
        with open(cache_filename, "r") as cache_file:
            cached_json = json.load(cache_file)
            cached_response = {
                "input": cached_json["offer"],
                "output": f"{{\"output_categories\": {json.dumps(cached_json['output'])}, \"explanation\": {json.dumps(cached_json['output_explanation'])}}}",
                "prompt_obj": prompt_obj,
                "prompt_str": prompt_str,
                "llm": model,
                "llm_config": llm_config,
                "start_time": generation_start_time,
                "end_time": datetime.now(),
            }

        return cached_response



    # Create chat completion request
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt_str}],
        **llm_config
    )

    content = response.choices[0].message.content

    response_data = {
        "input": prompt_str,
        "output": content,
        "prompt_obj": prompt_obj,
        "prompt_str": prompt_str,
        "llm": model,
        "llm_config": llm_config,
        "start_time": generation_start_time,
        "end_time": datetime.now(),
    }

    return response_data


def evaluate_generation(generation, response_dict, db_item, trace):
    print(f"\n[+] Evaluating {response_dict['llm']} response...")
    
    # # evaluation ragas  ---- require whole dataset, not single item
    # # result = evaluate(
    # #     dataset,
    # #     metrics=[
    # #         context_precision,
    # #         faithfulness,
    # #         answer_relevancy,
    # #         context_recall,
    # #     ],
    # # )

    evaluator = RelevanceEvaluator(mode=MODE)
    ground_truth = db_item["ground_truth"] if isinstance(db_item, dict) else db_item.expected_output

    # evalautor_prompt_obj = langfuse.get_prompt("qa_evaluator")
    evalautor_prompt_obj = langfuse.get_prompt("qa_evaluator_bullets")
    evaluator_prompt_str = evalautor_prompt_obj.compile(
            query=response_dict["question"],
            true_answer=ground_truth,
            model_answer=response_dict["output"]
        )

    evaluation = evaluator._evaluate_strings(
        # prediction=response_dict["output"],
        # input=response_dict["question"],
        # reference=ground_truth,
        evaluation_prompt=evaluator_prompt_str
    )

    evaluation_generation = langfuse.generation(
        name="final_evaluation",
        model=response_dict["llm"],
        input=evaluation["prompt"],
        trace_id=trace.id,
        parent_observation_id=generation.id,
        prompt=evalautor_prompt_obj,
        metadata={"question": response_dict["question"], "score": evaluation["score"], "grade": evaluation["value"], "expected_output": ground_truth, "model_output": response_dict["output"], "juror_scores": evaluation["juror_scores"]}
    )
    evaluation_generation.end(output=evaluation["value"])


    generation.score(
        name="CORRECTNESS",
        # any float value
        value=float(evaluation["score"])
    )

    # Store the evaluation score for each juror
    for juror, score_info in evaluation["juror_scores"].items():
        juror_generation = langfuse.generation(
            name=juror,
            model=response_dict["llm"],
            input=evaluation["prompt"],
            trace_id=trace.id,
            prompt=evalautor_prompt_obj,
            parent_observation_id=evaluation_generation.id,
            metadata={"grade": score_info["score"], "explanation": score_info["explanation"]}
        )
        juror_generation.end(output=score_info["explanation"])

    
def llm_generate_and_evaluate(response_dict, db_item, run_id, using_LF_dataset):
    trace = create_trace(response_dict, run_id)
    generation = create_generation(response_dict, trace)
    evaluate_generation(generation, response_dict, db_item, trace)

    if using_LF_dataset:
        db_item.link(generation, run_id)

    langfuse.flush()


def run_experiment(dataset, gpt35turboinstruct_config=None, gpt35turbo_config=None, elmib_config=None):
    # df = pd.read_csv( "data/bosch-samples_3_with_answers.csv", encoding='cp1252')
    if not gpt35turboinstruct_config:
        gpt35turboinstruct_config = {}
    if not gpt35turbo_config:
        gpt35turbo_config = {}
    if not elmib_config:
        elmib_config = {}

    openai_client = OpenAI()

    # Split the categories array into chunks of 100 elements each
    categories_subsets = [categories[i:i+100] for i in range(0, len(categories), 100)]

    categorizer = Categorizer(mode=MODE)

    parser = PydanticOutputParser(pydantic_object=Category)

    expertiment_uuid = str(uuid4())[:8]
    llms = ["gpt-3.5-turbo"]#, "elmib"]
    for l in llms:
        run_id = f"{l}_{expertiment_uuid}"
        csv_results = []
        for item in tqdm(dataset.items[:200]):
            offer = item.input.strip()

            
            gpt35turbo_shortlisted_categories = []
            if l == "gpt-3.5-turbo":
                for idx, subset in enumerate(categories_subsets):
                    # print(f"\nCategorizing '{offer}'\nModel: {l}")
                    prompt_obj = langfuse.get_prompt("categorizator_multi")
                    response = invoke_llm(
                        client=openai_client,
                        prompt_obj=prompt_obj,
                        prompt_dictionary={
                            "offer": offer,
                            "categories": ", ".join(subset),
                            "n_cat": "3",
                            "output_instructions": parser.get_format_instructions()
                        }
                    )
                    gpt35turbo_shortlisted_categories += json.loads(response["output"])["output_categories"]

                    # shortlist_generation = create_generation(response_dict=response, trace=trace, db_item=item, parent_observation_id=high_level_cat_span.id, run_id=run_id)


                # Categorize based on shortlisted categories
                final_response = invoke_llm(
                    client=openai_client,
                    prompt_obj = prompt_obj,
                    prompt_dictionary={
                        "offer": offer, 
                        "categories": ", ".join(gpt35turbo_shortlisted_categories), 
                        "n_cat": "1", 
                        "output_instructions": parser.get_format_instructions()
                    }
                )
                final_category = json.loads(final_response["output"])["output_categories"]

                trace = langfuse.trace(input=offer, output=final_category, session_id=run_id, metadata={
                    "gpt35turbo_final_output": json.loads(final_response["output"])["output_categories"],
                    "gpt35turbo_firstlevel_outputs": gpt35turbo_shortlisted_categories,
                    "expected_category": item.expected_output,
                    "output_explanation": json.loads(final_response["output"])["explanation"]
                })
                main_span = langfuse.span(trace_id=trace.id, name="main_span", input=offer)
                high_level_cat_span = langfuse.span(trace_id=trace.id, input=response["prompt_str"], output=gpt35turbo_shortlisted_categories, name="high_level_categorization", parent_observation_id=main_span.id)
                final_cat_span = langfuse.span(trace_id=trace.id, name="final_categorization", input=offer, output=final_category, parent_observation_id=main_span.id)
                final_generation = create_generation(response_dict=final_response, trace=trace, db_item=item, parent_observation_id=final_cat_span.id, run_id=run_id)
                
                final_data = {
                    "offer": offer,
                    "output": final_category,
                    "expected_output": item.expected_output,
                    "output_explanation": json.loads(final_response["output"])["explanation"],
                    "shortlisted_categories": gpt35turbo_shortlisted_categories
                }
                csv_results.append(final_data)
                md5_hash = hashlib.md5(offer.encode()).hexdigest()
        
                # Write cache file
                cache_filename = f"cache/sn_llm_cat_cache/{md5_hash}.cache"
                if not os.path.exists(cache_filename):
                    with open(cache_filename, "w") as cache_file:
                        json.dump(final_data, cache_file, indent=4)


            
        with open(f"experiment_{run_id}.csv", "w", newline="") as csvfile:
            fieldnames = ["offer", "output", "expected_output","output_explanation"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for result in csv_results:
                writer.writerow(result)





if __name__ == "__main__":
    langfuse = Langfuse(
        secret_key=os.getenv("LOCAL_LANGFUSE_PRIVATE"),
        public_key=os.getenv("LOCAL_LANGFUSE_PUBLIC"),
        host=os.getenv("LOCAL_LANGFUSE_HOST")
    )


    MODE = "cat"
    file_name = "SN_20k"
    dataset_name = "spendnet_500"
    # langfuse.create_dataset(name=dataset_name)
    # with open(f'data/{file_name}.csv', 'r', encoding="UTF8") as csv_file:
    #     csv_reader = csv.reader(csv_file)
    #     next(csv_reader)
        
    #     for idx, row in enumerate(csv_reader):
    #         if idx >= 500:
    #             break
    #         langfuse.create_dataset_item(
    #             dataset_name=dataset_name,
    #             input=f"### TITLE:\n {row[2].strip()}\n\n ### DESCRIPTION:\n {row[3]}",
    #             expected_output=row[4].strip()
    #         )
    
    dataset = langfuse.get_dataset(dataset_name)

    run_experiment(dataset, gpt35turboinstruct_config={
        "frequency_penalty": 0,
        "presence_penalty": 0,
        "top_p": 0.75,
        "temperature": 0.1,
        "max_tokens": 2000
    })