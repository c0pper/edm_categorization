import csv
from datetime import datetime
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
        input=response_dict["prompt_str"],
        metadata={"llm": response_dict["llm"]},
        start_time=response_dict["start_time"],
        end_time=response_dict["end_time"],
        prompt=response_dict["prompt_obj"],
        trace_id=trace.id
    )
    generation.end(output=response_dict["output"])

    return generation


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

    evalautor_prompt_obj = langfuse.get_prompt("qa_evaluator")
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


def run_bosch_qa_experiment():
    # df = pd.read_csv( "data/bosch-samples_3_with_answers.csv", encoding='cp1252')

    categorizer = Categorizer(mode=MODE)

    dataset = langfuse.get_dataset("bosch-samples")
    using_LF_dataset = not isinstance(dataset, Dataset)

    expertiment_uuid = str(uuid4())[:8]
    llms = ["elmib", "chatgpt-3.5-turbo"]
    for l in llms:
        run_id = f"{l}_{expertiment_uuid}"
        for item in tqdm(dataset if not using_LF_dataset else dataset.items):
            if using_LF_dataset:
                question = item.input.split("CONTEXT")[0].strip()
            else:
                question = str(item['question'])

            if using_LF_dataset:
                contexts = item.input.split("CONTEXT")[1]
                contexts = contexts.replace("\\\\n\\\\n", "\n\n")
                contexts = contexts.replace("\\n\\n", "\n\n")
                contexts = contexts.split("\n\n")
            else:
                contexts = item["contexts"].split("\\\\n\\\\n")  #the df has the contexts column
            
            # Combine the "question" and "context" with "CONTEXT" in between
            nl = "\n\n"
            # input_text = f"{question} CONTEXT {nl.join(contexts)}"
            
            generationStartTime = datetime.now()
            if l == "chatgpt-3.5-turbo":
                prompt_obj = langfuse.get_prompt("nestor_cgpt_original_prompt")
                prompt_str = prompt_obj.compile(context=nl.join(contexts), query=question)
                response = categorizer.ask_chatgpt({"question": prompt_str})
                response = {"question": question, "output": response["output"], "prompt_obj": prompt_obj, "prompt_str": prompt_str, "llm": l, "start_time": generationStartTime, "end_time": datetime.now()}

            elif l == "elmib":
                prompt_obj = langfuse.get_prompt("nestor_elmib_original_prompt")
                prompt_str = prompt_obj.compile(context=nl.join(contexts), query=question)
                response = categorizer.ask_elmib({"question": prompt_str})
                response = {"question": question, "output": response["output"], "prompt_obj": prompt_obj, "prompt_str": prompt_str, "llm": l, "start_time": generationStartTime, "end_time": datetime.now()}
                
            llm_generate_and_evaluate(response, item, run_id, using_LF_dataset)



if __name__ == "__main__":
    # langfuse = Langfuse(
    #     secret_key=os.getenv("LANGFUSE_PRIVATE"),
    #     public_key=os.getenv("LANGFUSE_PUBLIC"),
    #     host="https://cloud.langfuse.com"
    # )
    langfuse = Langfuse(
        secret_key=os.getenv("LOCAL_LANGFUSE_PRIVATE"),
        public_key=os.getenv("LOCAL_LANGFUSE_PUBLIC"),
        host=os.getenv("LOCAL_LANGFUSE_HOST")
    )

    MODE = "nestor_cgpt"
    # langfuse.create_dataset(name="bosch-samples")
    # with open('data/bosch-samples_for_ragas.csv', 'r') as csv_file:
    #     csv_reader = csv.reader(csv_file)
    #     next(csv_reader)
        
    #     for row in csv_reader:
    #         langfuse.create_dataset_item(
    #             dataset_name="bosch-samples",
    #             input=f"{row[0].strip()} CONTEXT {row[2]}",
    #             expected_output=row[1].strip()
    #         )

    run_bosch_qa_experiment()



    # MODE = "nestor_cgpt"

    # df = pd.read_csv( "data/bosch-samples_3_with_answers.csv", encoding='cp1252')

    # categorizer = Categorizer(mode="nestor_cgpt")

    # dataset = langfuse.get_dataset("bosch-samples")
    # using_LF_dataset = not isinstance(dataset, Dataset)


    # llms = ["elmib", "chatgpt-3.5-turbo"]
    # for l in llms:
    #     run_id = f"{l}_{str(uuid4())[:8]}"
    #     for item in tqdm(dataset if not using_LF_dataset else dataset.items):
    #         if using_LF_dataset:
    #             question = item.input
    #         else:
    #             question = str(item['question'])

    #         if using_LF_dataset:
    #             contexts = question.split("CONTEXT")[1]
    #             contexts = contexts.replace("\\\\n\\\\n", "\n\n")
    #             contexts = contexts.replace("\\n\\n", "\n\n")
    #             contexts = contexts.split("\n\n")
    #         else:
    #             contexts = item["contexts"].split("\\\\n\\\\n")  #the df has the contexts column
            
    #         # Combine the "question" and "context" with "CONTEXT" in between
    #         nl = "\n\n"
    #         input_text = f"{question} CONTEXT {nl.join(contexts)}"

    #         if l == "chatgpt-3.5-turbo":
    #             response = categorizer.ask_chatgpt({"question": input_text})
    #             response = {"question": question, "output": response["output"], "prompt": response["prompt"], "llm": l}

    #         elif l == "elmib":
    #             response = categorizer.ask_elmib({"question": input_text})
    #             response = {"question": question, "output": response["output"], "prompt": response["prompt"], "llm": l}


    #         llm_generate_and_evaluate(response, item, run_id)