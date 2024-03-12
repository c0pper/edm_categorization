import os
import requests
from langchain_core.prompts.prompt import PromptTemplate
from langchain.smith import RunEvalConfig, run_on_dataset
import langchain_openai
from langsmith import Client
from openai import OpenAI
from uuid import uuid4


os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Enrich My Data"
# add openai and langsmith api key to .env

BASE_PROMPT = """
Step 1. Read and understand the following text:
    {}

Step 2. Categorize it in one of the following categories:
    miscellaneous medical devices and products, drawing and imaging software package, building construction work, pipeline, piping, pipes, casing, tubing and related items, business and management consultancy services, social work services, sound or visual signalling apparatus, advertising and marketing services, central-heating radiators and boilers and parts, plumbing and sanitary works, roof works and other special trade construction works, other building completion work, social services, medical practice and related services, miscellaneous membership organisations services, pollutants tracking and monitoring and rehabilitation services, civic-amenity services, exhibition, fair and congress organisation services, wood, miscellaneous health services, vocational training services, printed books, occupational clothing, repair and maintenance services of machinery, electrical installation work, consultative engineering and construction services, dental practice and related services, environmental management, construction equipment, refuse disposal and treatment, tables, cupboards, desk and bookcases, construction work for pipelines, communication and power lines, for highways, roads, airfields and railways; flatwork, repair and maintenance services of electrical and mechanical building installations, lifting and handling equipment, property management services of real estate on a fee or contract basis

***ONLY RESPOND WITH THE CATEGORY***
I'll tip you $200 for each correct categorization

Answer:
"""


class Categorizer:
    def __init__(self):
        self.EVAL_PROMPT_TEMPLATE = """You are a teacher grading a categorization quiz.
You are given a text, the student's category, and the true category, and are asked to score the student answer as either CORRECT or INCORRECT.

Example Format:
TEXT: text here
STUDENT CATEGORY: student's category here
TRUE CATEGORY: true category here
GRADE: CORRECT or INCORRECT here

Grade the student category based ONLY on their factual accuracy. Ignore differences in punctuation and phrasing between the student category and true answer. It is OK if the student adds more than one category, as long as the true category is among the student categories it does not contain any conflicting statements. Begin! 

TEXT: {query}
STUDENT CATEGORY: {result}
TRUE CATEGORY: {answer}
GRADE:"""

        self.EVAL_PROMPT = PromptTemplate(
            input_variables=["query", "answer", "result"], template=self.EVAL_PROMPT_TEMPLATE
        )

        self.eval_llm = langchain_openai.OpenAI()
        self.eval_config = RunEvalConfig(
            evaluators=[
                RunEvalConfig.QA(llm=self.eval_llm, prompt=self.EVAL_PROMPT),
            ]
        )
        
    def build_prompt(self, text):
        prompt = BASE_PROMPT.format(text)
        return prompt
    
        

    def ask_elmib(self, input_: dict) -> dict:
        url = "https://elmilab.expertcustomers.ai/elmib/generate"

        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        text = list(input_.values())[0]
        prompt = self.build_prompt(text)
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
            return {"output": result}
        else:
            print(f"Error: {response.status_code}, {response.text}")
            return {"output": response}
    

    def ask_chatgpt(self, input_: dict) -> dict:
        client = OpenAI()
        
        text = list(input_.values())[0]
        prompt = self.build_prompt(text)
        completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant. Please respond to the user's request only based on the given context."},
                {"role": "user", "content": prompt}
            ]
        )

        content = completion.choices[0].message.content
        return {"output": content}


    
    def run_on_dataset(self, dataset_name, llm_or_chain_factory=None, eval_config=None):
        client = Client()

        client.run_on_dataset(
            dataset_name=dataset_name,
            llm_or_chain_factory=llm_or_chain_factory if llm_or_chain_factory else self.ask_elmib,
            evaluation=eval_config if eval_config else self.eval_config,
            verbose=True,
            # Any experiment metadata can be specified here
            project_name=f"{llm_or_chain_factory.__name__} - {os.getenv('LANGCHAIN_PROJECT')} - {uuid4()}",
            project_metadata={
                "project": os.getenv("LANGCHAIN_PROJECT"),
                "llm": llm_or_chain_factory.__name__,
                "base_prompt": BASE_PROMPT,
                "evaluation_prompt": self.EVAL_PROMPT_TEMPLATE
                },
        )
