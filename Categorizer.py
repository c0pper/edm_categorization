import os
import requests
from langchain_core.prompts.prompt import PromptTemplate
from langchain.smith import RunEvalConfig, run_on_dataset
from langchain.evaluation import StringEvaluator
from typing import Any, Optional
import re
import langchain_openai
from langsmith import Client
from openai import OpenAI
from uuid import uuid4
import pandas as pd
from pathlib import Path
import prompts
import json


os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Enrich My Data"
# add openai and langsmith api key to .env






class RelevanceEvaluator(StringEvaluator):
    """An LLM-based relevance evaluator."""

    def __init__(self, mode):
        self.mode = mode
        self.llm = Categorizer.eval_llm
        self.template = """You are now a evaluator for a question answering system.

# Task

Your task is to give a grade CORRECT or INCORRECT based on how fitting modelOutput was given the true answer.

# Input Data Format

You will receive a query, a true_answer and a model_answer. The query is the input that was given to the model. The model_answer is the output that the model generated for the given query.

# Score Format Instructions

The score format is a value of CORRECT or INCORRECT. Evaluate the model_answer based on the following criteria.

# Score Criteria

You will be given criteria by which the grade is influenced. Always follow those instructions to determine the grade.

In your step by step explanation you will explain how you evaluated the model_answer for each criteria.

- Make sure that any numeric information mentioned in the model_answer related to voltages and such are accurate according exclusively to the true_answer. 
- Additional information in the model_answer does NOT affect the grade.
- Disregard differences in phrasing, and consider the model_answer correct as long as it covers the information in the provided true_answer.

# Process Instructions

Give a grade that is explained by walking through all the different criteria by which you evaluated the model_answer.

Think step by step to make sure we get an accurate grade!

### input:

[QUERY]: {query}

[TRUE_ANSWER]: {true_answer}

[MODEL_ANSWER]: {model_answer}

First give the grade <GRADE: (CORRECT or INCORRECT here)>.
Then explain all the criteria you went through step by step <EXPLANATION: (EXPLANATION here)>.
"""

        self.eval_chain = PromptTemplate.from_template(self.template) | self.llm

    @staticmethod
    def extract_final_score(text):
        # Find the line containing "Final Score" (case-insensitive)
        final_score_line = re.search(r'Final Score:[\s\S]*', text, re.IGNORECASE)
        
        if final_score_line:
            # Extract the number between 0-100 in the final score line
            score_match = re.search(r'\b\d{1,3}\b', final_score_line.group())
            
            if score_match:
                # Convert the matched score to an integer
                final_score = int(score_match.group())
                return final_score
            else:
                print("No score found")
                return None  # No score found
        else:
            print(f"'Final Score' line not found in the text \n {text}")
            return None  # "Final Score" line not found in the text

    @property
    def requires_input(self) -> bool:
        return True

    @property
    def requires_reference(self) -> bool:
        return True

    @property
    def evaluation_name(self) -> str:
        return "scored_relevance"

    def _evaluate_strings(
        self,
        prediction: str,
        input: Optional[str] = None,
        reference: Optional[str] = None,
        **kwargs: Any
    ) -> dict:
        if self.mode != "cat":
            query_query = input.split("CONTEXT")[0]
        elif self.mode == "cat":
            query_query = list(input.values())[0]
        evaluation_data = {"input": query_query, "prediction": prediction, "query": query_query, "true_answer": reference, "model_answer": prediction}
        evaluator_result = self.eval_chain.invoke(
            evaluation_data, kwargs
        )

        # SCORE BASED
        # final_score = self.extract_final_score(evaluator_result)
        # if final_score:
        #     return {"score": float(final_score)}
        # else:
        #     return {"score": float(0)}

        # CORRECT/INCORRECT
        matches = list(re.finditer(r'INCORRECT|CORRECT', evaluator_result))

        if matches:
            score = matches[0].group(0).strip()
            if score == "INCORRECT":
                return {"score": float(0), "value": "INCORRECT", "explanation": evaluator_result, "prompt": self.template.format(**evaluation_data)}
            elif score == "CORRECT":
                return {"score": float(1), "value": "CORRECT", "explanation": evaluator_result, "prompt": self.template.format(**evaluation_data)}
        else:
            print("evalutar result\n\n" + evaluator_result)
            print("matches\n\n")
            for match in matches:
                print(match)
            print("No grade text found.")
            return {"score": 0, "value": "INCORRECT", "explanation": evaluator_result, "prompt": self.template.format(**evaluation_data)}




class Categorizer:
    def __init__(self, mode):
        self.client = Client()
        self.mode = mode

        Categorizer.eval_llm = langchain_openai.OpenAI(model="gpt-3.5-turbo-instruct")


        if self.mode == "cat":
            self.BASE_PROMPT = prompts.BASE_PROMPT_EDM_CAT

            self.EVAL_PROMPT = PromptTemplate(
                input_variables=["query", "answer", "result"], template=prompts.EVAL_PROMPT_TEMPLATE_EDM_CAT
            )

            self.eval_config = RunEvalConfig(
                evaluators=[
                    RunEvalConfig.QA(llm=self.eval_llm, prompt=self.EVAL_PROMPT),
                ]
            )

        elif self.mode == "context":
            self.BASE_PROMPT = prompts.BASE_PROMPT_EDM_QA

            self.EVAL_PROMPT = PromptTemplate(
                input_variables=["query", "context", "result"], template=prompts.EVAL_PROMPT_TEMPLATE_EDM_QA
            )

            self.eval_config = RunEvalConfig(
                evaluators=[
                    RunEvalConfig.ContextQA(llm=self.eval_llm, prompt=self.EVAL_PROMPT),
                ]
            )

        elif self.mode == "nestor_elmi":
            self.BASE_PROMPT = prompts.BASE_PROMPT_EDM_QA_NESTOR_ELMI

            self.EVAL_PROMPT = PromptTemplate(
                input_variables=["query", "context", "result"], template=prompts.EVAL_PROMPT_TEMPLATE_EDM_QA
            )

            self.eval_config = RunEvalConfig(
                custom_evaluators = [RelevanceEvaluator(self.mode)],
                # evaluators=[
                #     RelevanceEvaluator(),
                #     # RunEvalConfig.ContextQA(llm=self.eval_llm, prompt=self.EVAL_PROMPT),
                # ]
            )

        elif self.mode == "nestor_cgpt":
            self.BASE_PROMPT = prompts.BASE_PROMPT_EDM_QA_NESTOR_CGPT

            self.EVAL_PROMPT = PromptTemplate(
                input_variables=["query", "context", "result"], template=prompts.EVAL_PROMPT_TEMPLATE_EDM_QA
            )

            self.eval_config = RunEvalConfig(
                custom_evaluators = [RelevanceEvaluator(self.mode)],
                # evaluators=[
                #     RelevanceEvaluator(),
                #     # RunEvalConfig.ContextQA(llm=self.eval_llm, prompt=self.EVAL_PROMPT),
                # ]
            )

    @staticmethod
    def format_context(prompt_input: dict):
        context = list(prompt_input.values())[0].split("CONTEXT")[1]
        context = context.replace("\\n\\n", "\n\n")
        context_gp = context.split("\n\n")
        context = "".join([x + "\n\n" for x in context_gp])
        return context


    def ask_elmib(self, input_: dict, categories:list = None) -> dict:
        url = "https://elmilab.expertcustomers.ai/elmib/generate"

        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        # text = list(input_.values())[0]
        
        if self.mode == "cat":
            if not categories:
                raise ValueError("No categories list provided")
            question = list(input_.values())[0]
            categories = ", ".join(categories)
            prompt = self.BASE_PROMPT.format(question, categories)
        elif self.mode in ["context", "nestor_elmi", "nestor_cgpt"]:
            question = list(input_.values())[0].split("CONTEXT")[0]
            context = self.format_context(input_)
            # context = list(input_.values())[0].split("CONTEXT")[1]
            # context_gp = context.split("\n\n")
            # context = "".join([x + "\n\n" for x in context_gp])
            prompt = self.BASE_PROMPT.format(context, question)
            print(prompt)
        # print(f"----\n############PROMPT:\n{prompt}\n")



        body = {
            "instruction": "",
            "input": "",
            "text": prompt,
            "use_beam_search": True,
            "num_beams": 2,
            "temperature": 0,
            "top_p": 0.75,
            "top_k": 40,
            "max_length": 256,
            "stream": False,
            "stop_tokens": [],
            "lang": "en",
            "input_auto_trunc": False,
            "custom_profile": "",
        }

        response = requests.post(url, json=body, headers=headers)

        if response.status_code == 200:
            result = response.json()
            print(result)
            return {"output": result, "prompt": prompt}
        else:
            print(f"Error: {response.status_code}, {response.text}")
            return {"output": response}
    

    def ask_chatgpt(self, input_: dict, categories:list = None) -> dict:
        client = OpenAI()
        

        if self.mode == "cat":
            if not list(categories):
                raise ValueError("No categories list provided")
            question = list(input_.values())[0]
            categories = ", ".join(list(categories))
            prompt = self.BASE_PROMPT.format(question, categories)
        elif self.mode in ["context", "nestor_elmi", "nestor_cgpt"]:
            question = list(input_.values())[0].split("CONTEXT")[0]
            context = self.format_context(input_)
            # context = list(input_.values())[0].split("CONTEXT")[1]
            prompt = self.BASE_PROMPT.format(context, question)
            print(prompt)

        completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant. Please respond to the user's request only based on the given context."},
                {"role": "user", "content": prompt}
            ]
        )

        content = completion.choices[0].message.content
        return {"output": content, "prompt": prompt}


    
    def run_on_dataset(self, dataset_name, llm_or_chain_factory=None, eval_config=None):
        self.client.run_on_dataset(
            dataset_name=dataset_name,
            llm_or_chain_factory=llm_or_chain_factory if llm_or_chain_factory else self.ask_elmib,
            evaluation=eval_config if eval_config else self.eval_config,
            verbose=True,
            # Any experiment metadata can be specified here
            project_name=f"{llm_or_chain_factory.__name__} - {os.getenv('LANGCHAIN_PROJECT')} - {uuid4()}",
            project_metadata={
                "project": os.getenv("LANGCHAIN_PROJECT"),
                "llm": llm_or_chain_factory.__name__,
                "base_prompt": self.BASE_PROMPT,
                "evaluation_prompt": self.EVAL_PROMPT
                },
        )

    def add_csv_dataset(self, csv_path, input_key, output_key, description=None, data_type=None, separator=None):
        path = Path(csv_path)
        sep = "," if not separator else separator
        try:
            df = pd.read_csv(path, sep=sep, encoding="utf8")
        except UnicodeDecodeError:
            df = pd.read_csv(path, sep=sep, encoding="cp1252")
        input_keys = input_key
        output_keys = output_key
        dataset = self.client.upload_dataframe(
            df=df,
            input_keys=input_keys,
            output_keys=output_keys,
            name=path.stem,
            description="Dataset created from a CSV file" if not description else description,
            data_type="kv" if not data_type else data_type
        )
    

