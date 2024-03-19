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


langfuse = Langfuse(
    secret_key=os.getenv("LANGFUSE_PRIVATE"),
    public_key=os.getenv("LANGFUSE_PUBLIC"),
    host="https://cloud.langfuse.com"
)

MODE = "nestor_cgpt"

df = pd.read_csv( "data/bosch-samples_3_with_answers.csv", encoding='cp1252')
contexts_list = df['contexts'].str.split('\n\n').tolist()

categorizer = Categorizer(mode="nestor_cgpt")

dataset = Dataset.from_pandas(df)
print(dataset[0])


for item in tqdm(dataset):
    question = str(item['question'])
    contexts = item["contexts"].split("\\\\n\\\\n")
    
    # Combine the "question" and "context" with "CONTEXT" in between
    nl = "\n\n"
    input_text = f"{question} CONTEXT {nl.join(contexts)}"

    responses = []
    response_cgpt = categorizer.ask_chatgpt({"question": input_text})
    response_cgpt = {"output": response_cgpt["output"], "prompt": response_cgpt["prompt"], "llm": "chatgpt-3.5-turbo"}
    responses.append(response_cgpt)

    response_elmi = categorizer.ask_elmib({"question": input_text})
    response_elmi = {"output": response_elmi["output"], "prompt": response_elmi["prompt"], "llm": "elmib"}
    responses.append(response_elmi)

    for r in responses:
        trace = langfuse.trace(
            session_id=r["llm"],
            input=item["question"],
            output=r["output"]
        )

        # Create generation in Langfuse
        generation = langfuse.generation(
            name=f'{r["llm"]}_generation',
            model="gpt-3.5-turbo",
            model_parameters={"maxTokens": "1000", "temperature": "0.9"},
            input=r["prompt"],
            metadata={"llm": r["llm"]},
            trace_id=trace.id
        )
        generation.end(output=r["output"])


        # evaluation ragas  ---- require whole dataset, not single item
        # result = evaluate(
        #     dataset,
        #     metrics=[
        #         context_precision,
        #         faithfulness,
        #         answer_relevancy,
        #         context_recall,
        #     ],
        # )

        # evalution custom evaluator
        evaluator = RelevanceEvaluator(mode=MODE)
        evaluation = evaluator._evaluate_strings(
            prediction=r["output"],
            reference=item["ground_truth"],
            input=item["question"]
        )

        evaluation_observation = langfuse.generation(
            name="evaluation",
            model=r["llm"],
            input=r["prompt"],
            trace_id=trace.id,
            parent_observation_id=generation.id,
            metadata={"score": evaluation["score"], "grade": evaluation["value"]}
        )
        evaluation_observation.end(output=evaluation["explanation"])

        generation.score(
            name="real eval",
            # any float value
            value=float(evaluation["score"]),
            comment=evaluation["explanation"]
        )
        
        # The SDK executes network requests in the background.
        # To ensure that all requests are sent before the process exits, call flush()
        # Not necessary in long-running production code
        langfuse.flush()