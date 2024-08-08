import csv
from datetime import datetime
import json
from uuid import uuid4
from langfuse import Langfuse
import pandas as pd
from tqdm import tqdm
from Categorizer import Categorizer, RelevanceEvaluator
import pandas as pd
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
        input=response_dict["question"],
        metadata={"llm": response_dict["llm"], "llm_config": response_dict["llm_config"]},
        start_time=response_dict["start_time"],
        end_time=response_dict["end_time"],
        prompt=response_dict["prompt_obj"],
        trace_id=trace.id
    )
    generation.end(output=response_dict["output"])

    return generation


def evaluate_generation(generation, response_dict, db_item, trace):
    """
    Evaluates the response generated by an LLM (Large Language Model) by comparing it with the ground truth.

    Args:
        generation (Generation): The generation object representing the LLM's response.
        response_dict (dict): A dictionary containing information about the LLM's response, including the question and the output generated.
        db_item (dict or DBItem): The database item containing the ground truth against which the LLM's response will be evaluated. It can be a dictionary or a DBItem object.
        trace (Trace): The trace object used to track the evaluation process.

    Returns:
        None: This function does not return any value. It performs evaluation and scoring operations in place.

    Description:
        The function evaluates the correctness and relevance of an LLM's response by comparing it with the ground truth. 
        It uses a relevance evaluator to generate an evaluation prompt, which is then used to obtain an evaluation of the LLM's response.
        The evaluation results, including scores and explanations from multiple jurors, are stored and associated with the relevant generation and trace objects.
        Finally, the function updates the generation object with the overall correctness score and stores individual juror scores.
    """
    print(f"\n[+] Evaluating {response_dict['llm']} response...")

    evaluator = RelevanceEvaluator(mode=MODE)
    ground_truth = db_item["ground_truth"] if isinstance(db_item, dict) else db_item.expected_output
    ground_thruth_bullets = [b.strip() for b in ground_truth.split("- ") if len(b) > 2]

    # evalautor_prompt_obj = langfuse.get_prompt("qa_evaluator")
    # evalautor_prompt_obj = langfuse.get_prompt("qa_evaluator_bullets")
    # evaluator_prompt_str = evalautor_prompt_obj.compile(
    #         query=response_dict["question"],
    #         true_answer=ground_truth,
    #         model_answer=response_dict["output"]
    #     )
    
    evalautor_prompt_obj = langfuse.get_prompt("info_coverage_checker")
    
    context_span = langfuse.span(name="context", input=response_dict["prompt_str"], trace_id=trace.id, parent_observation_id=generation.id)
    bullets_span = langfuse.span(name="Bullets evaluation", trace_id=trace.id, parent_observation_id=generation.id)
    
    bullet_scores = []
    for idx, bullet in enumerate(ground_thruth_bullets):
        evaluator_prompt_str = evalautor_prompt_obj.compile(
                fact=bullet,
                larger_text=response_dict["output"]
        )
        evalautor_prompt_dict = {"fact": bullet, "larger_text": response_dict["output"]}

        evaluation = evaluator._evaluate_strings(
            evaluation_prompt=evaluator_prompt_str,
            evalautor_prompt_dict=evalautor_prompt_dict
        )
        bullet_scores.append(evaluation["score"])

        evaluation_generation = langfuse.generation(
            name=f"Bullet {idx+1}",
            model=response_dict["llm"],
            input=evaluation["prompt"],
            trace_id=trace.id,
            parent_observation_id=bullets_span.id,
            prompt=evalautor_prompt_obj,
            metadata={"question": response_dict["question"], "score": evaluation["score"], "grade": evaluation["value"], "expected_output": ground_truth, "model_output": response_dict["output"], "juror_scores": evaluation["juror_scores"]}
        )
        evaluation_generation.end(output=evaluation["value"])


        # generation.score(
        #     name="CORRECTNESS",
        #     # any float value
        #     value=float(evaluation["score"])
        # )

        # Store the evaluation score for each juror
        all_explanations = "\n".join([f'Juror {idx+1}: {v["score"]} - {v["explanation"]}' for idx, v in enumerate(evaluation["juror_scores"].values()) if v["explanation"] != "Not interrogated"])
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
    
    overall_answer_score = sum(bullet_scores) / len(bullet_scores)
    # generation.score(name="CORRECTNESS", value=float(overall_answer_score))
    ground_truths_formatted = "\n".join(ground_thruth_bullets)
    bullets_span.update(input=f"Information the answer should have covered:\n\n {ground_truths_formatted}", output=f"Overall answer score based on information coverage: {overall_answer_score}")
    trace.update(name=response_dict["question"], output=response_dict["output"])
    generation.score(name="CORRECTNESS", value=float(overall_answer_score), comment=all_explanations)

    
def llm_generate_and_evaluate(response_dict, db_item, run_id, using_LF_dataset):
    trace = create_trace(response_dict, run_id)
    generation = create_generation(response_dict, trace)
    evaluate_generation(generation, response_dict, db_item, trace)

    if using_LF_dataset:
        db_item.link(generation, run_id)

    langfuse.flush()


def run_bosch_qa_experiment(dataset, gpt35turboinstruct_config=None, gpt35turbo_config=None, elmib_config=None):
    # df = pd.read_csv( "data/bosch-samples_3_with_answers.csv", encoding='cp1252')
    if not gpt35turboinstruct_config:
        gpt35turboinstruct_config = {}
    if not gpt35turbo_config:
        gpt35turbo_config = {}
    if not elmib_config:
        elmib_config = {}

    categorizer = Categorizer(mode=MODE)

    using_LF_dataset = not isinstance(dataset, Dataset)

    expertiment_uuid = str(uuid4())[:8]
    
    # #all llms
    # llms = ["gpt-3.5-turbo-instruct", "gpt-3.5-turbo", "elmib"]
    # llms = ["notebookLM"]
    # llms = ["notebookLM", "elmib"]
    llms = ["gpt4o_rag"]
    
    for l in llms:
        run_id = f"{l}_{expertiment_uuid}"
        for item in tqdm(dataset if not using_LF_dataset else dataset.items):
            if using_LF_dataset:
                question = item.input.split("CONTEXT")[0].strip()
                contexts = item.input.split("CONTEXT")[1]
                contexts = contexts.replace("\\\\n\\\\n", "\n\n")
                contexts = contexts.replace("\\n\\n", "\n\n")
                contexts = contexts.split("\n\n")
            else:
                question = str(item['question'])
                contexts = item["contexts"].split("\\\\n\\\\n")  #the df has the contexts column
            
            # Combine the "question" and "context" with "CONTEXT" in between
            nl = "\n\n"
            # input_text = f"{question} CONTEXT {nl.join(contexts)}"
            
            generationStartTime = datetime.now()
            if l == "gpt-3.5-turbo":
                print(f"\nAsking '{question}'\nModel: {l}")
                prompt_obj = langfuse.get_prompt("nestor_cgpt_original_prompt")
                prompt_str = prompt_obj.compile(context=nl.join(contexts), query=question)
                response = categorizer.ask_chatgpt({"question": prompt_str, "model": l})
                response = {"question": question, "output": response["output"], "prompt_obj": prompt_obj, "prompt_str": prompt_str, "llm": l, "llm_config": response.get("llm_config", {}), "start_time": generationStartTime, "end_time": datetime.now()}

            elif l == "gpt-3.5-turbo-instruct":
                print(f"\nAsking '{question}'\nModel: {l}")
                prompt_obj = langfuse.get_prompt("nestor_cgpt_original_prompt")
                prompt_str = prompt_obj.compile(context=nl.join(contexts), query=question)
                response = categorizer.ask_chatgpt({"question": prompt_str, "model": l}, llm_config=gpt35turboinstruct_config)
                response = {"question": question, "output": response["output"], "prompt_obj": prompt_obj, "prompt_str": prompt_str, "llm": l, "llm_config": response.get("llm_config", {}), "start_time": generationStartTime, "end_time": datetime.now()}

            elif l == "elmib":
                print(f"\nAsking '{question}'\nModel: {l}")
                prompt_obj = langfuse.get_prompt("nestor_elmib_original_prompt")
                prompt_str = prompt_obj.compile(context=nl.join(contexts), query=question)
                response = categorizer.ask_elmib({"question": prompt_str})
                response = {"question": question, "output": response["output"], "prompt_obj": prompt_obj, "prompt_str": prompt_str, "llm": l, "llm_config": response.get("llm_config", {}), "start_time": generationStartTime, "end_time": datetime.now()}
                
            elif l == "notebookLM":
                print(f"\nAsking '{question}'\nModel: {l}")
                with open("bosch_notebookLM_answers.json", "r", encoding="utf8") as f:
                    notebookLM_answers = json.load(f)
                prompt_obj = langfuse.get_prompt("nestor_elmib_original_prompt")
                prompt_str = f"DUMMY - {prompt_obj.compile(context=nl.join(contexts), query=question)}"
                response = [a for a in notebookLM_answers if a["q"] == question][0].get("a", "no answer found in notebookLM_answers json")
                response = {"question": question, "output": response, "prompt_obj": prompt_obj, "prompt_str": prompt_str, "llm": l, "llm_config": {}, "start_time": generationStartTime, "end_time": datetime.now()}                
                
            elif l == "gpt4o_rag": # caricamento pdf su interfaccia chatgpt e rag esclusivamente da li
                print(f"\nAsking '{question}'\nModel: {l}")
                with open("bosch_chatgpt4orag_answers.json", "r", encoding="utf8") as f:
                    gpt4orag_answers = json.load(f)
                prompt_obj = langfuse.get_prompt("nestor_elmib_original_prompt")
                prompt_str = f"DUMMY - {prompt_obj.compile(context=nl.join(contexts), query=question)}"
                response = [a for a in gpt4orag_answers if a["q"] == question][0].get("a", "no answer found in bosch_chatgpt4orag_answers json")
                response = {"question": question, "output": response, "prompt_obj": prompt_obj, "prompt_str": prompt_str, "llm": l, "llm_config": {}, "start_time": generationStartTime, "end_time": datetime.now()}
                
            llm_generate_and_evaluate(response, item, run_id, using_LF_dataset)



if __name__ == "__main__":
    langfuse = Langfuse(
        secret_key=os.getenv("LOCAL_LANGFUSE_PRIVATE"),
        public_key=os.getenv("LOCAL_LANGFUSE_PUBLIC"),
        host=os.getenv("LOCAL_LANGFUSE_HOST")
    )

    MODE = "nestor_cgpt"
    dataset_name = "bosch-samples-bullets"
    # langfuse.create_dataset(name=dataset_name)
    # df = pd.read_excel(f"data/{dataset_name}.xlsx")
        
    # for index, row in list(df.iterrows()):
    #     langfuse.create_dataset_item(
    #         dataset_name=dataset_name,
    #         input=f"{row[0].strip()} CONTEXT {row[2]}",
    #         expected_output=row[1].strip()
    #     )
    
    dataset = langfuse.get_dataset(dataset_name)

    run_bosch_qa_experiment(dataset, gpt35turboinstruct_config={
        "frequency_penalty": 0,
        "presence_penalty": 0,
        "top_p": 0.75,
        "temperature": 0.1,
        "max_tokens": 2000
    })