from uuid import uuid4
from langfuse import Langfuse
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
from ragas import evaluate
from dotenv import load_dotenv
import os

load_dotenv()


def create_trace(response_dict, run_id):
    trace = langfuse.trace(
        session_id=run_id,
        input=response_dict["question"],
        output=response_dict["output"]
    )

    return trace

def create_generation(response_dict, trace):
    # Create generation in Langfuse
    generation = langfuse.generation(
        name=f'{response_dict["llm"]}_generation',
        model=response_dict["llm"],
        input=response_dict["prompt"],
        metadata={"llm": response_dict["llm"]},
        trace_id=trace.id
    )
    generation.end(output=response_dict["output"])

    return generation

def evaluate_generation(generation, response_dict, db_item, trace):
    
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
    evaluation = evaluator._evaluate_strings(
        prediction=response_dict["output"],
        reference=db_item["ground_truth"] if isinstance(db_item, dict) else db_item.expected_output,
        input=response_dict["question"]
    )

    evaluation_generation = langfuse.generation(
        name="evaluation",
        model=response_dict["llm"],
        input=evaluation["prompt"],
        trace_id=trace.id,
        parent_observation_id=generation.id,
        metadata={"score": evaluation["score"], "grade": evaluation["value"]}
    )
    evaluation_generation.end(output=evaluation["explanation"])


    generation.score(
        name="CORRECTNESS",
        # any float value
        value=float(evaluation["score"]),
        comment=evaluation["explanation"]
    )


def llm_generate_and_evaluate(response_dict, db_item, run_id):
    trace = create_trace(response_dict, run_id)
    generation = create_generation(response_dict, trace)
    evaluate_generation(generation, response_dict, db_item, trace)

    if using_LF_dataset:
        db_item.link(generation, run_id)

    langfuse.flush()


if __name__ == "__main__":
    langfuse = Langfuse(
        secret_key=os.getenv("LANGFUSE_PRIVATE"),
        public_key=os.getenv("LANGFUSE_PUBLIC"),
        host="https://cloud.langfuse.com"
    )

    MODE = "nestor_cgpt"

    df = pd.read_csv( "data/bosch-samples_3_with_answers.csv", encoding='cp1252')

    categorizer = Categorizer(mode="nestor_cgpt")

    dataset = langfuse.get_dataset("bosch-samples")
    using_LF_dataset = not isinstance(dataset, Dataset)


    llms = ["elmib", "chatgpt-3.5-turbo"]
    for l in llms:
        run_id = f"{l}_{str(uuid4())[:8]}"
        for item in tqdm(dataset if not using_LF_dataset else dataset.items):
            if using_LF_dataset:
                question = item.input
            else:
                question = str(item['question'])

            if using_LF_dataset:
                contexts = question.split("CONTEXT")[1]
                contexts = contexts.replace("\\\\n\\\\n", "\n\n")
                contexts = contexts.replace("\\n\\n", "\n\n")
                contexts = contexts.split("\n\n")
            else:
                contexts = item["contexts"].split("\\\\n\\\\n")  #the df has the contexts column
            
            # Combine the "question" and "context" with "CONTEXT" in between
            nl = "\n\n"
            input_text = f"{question} CONTEXT {nl.join(contexts)}"

            if l == "chatgpt-3.5-turbo":
                response = categorizer.ask_chatgpt({"question": input_text})
                response = {"question": question, "output": response["output"], "prompt": response["prompt"], "llm": l}

            elif l == "elmib":
                response = categorizer.ask_elmib({"question": input_text})
                response = {"question": question, "output": response["output"], "prompt": response["prompt"], "llm": l}


            llm_generate_and_evaluate(response, item, run_id)