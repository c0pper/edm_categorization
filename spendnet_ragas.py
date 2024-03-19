from langchain.prompts import ChatPromptTemplate
from operator import itemgetter
from langchain_openai import ChatOpenAI
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from dotenv import load_dotenv
from langchain_community.document_loaders.csv_loader import CSVLoader
from ragas.testset.generator import TestsetGenerator
from ragas.testset.evolutions import simple, reasoning, multi_context
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas.metrics import (
    answer_relevancy,
    faithfulness,
    context_recall,
    context_precision,
)
from datasets import Dataset, Features, Value, Sequence, load_dataset
import pandas as pd
from Categorizer import Categorizer
load_dotenv()


df = pd.read_csv( "data/bosch-samples_for_ragas.csv", encoding='cp1252')


# Ask llm
categorizer = Categorizer(mode="nestor_cgpt")
df = df.head(3)
contexts_list = df['contexts'].str.split('\n\n').tolist()

responses = []
for index, row in df.iterrows():
    # Extract the "question" and "context" cells from the current row
    question = str(row['question'])
    context = str(row['contexts'])
    
    # Combine the "question" and "context" with "CONTEXT" in between
    input_text = f"{question} CONTEXT {context}"
    
    # Call the ask_cgpt method of the categorizer object with the combined input text
    response = categorizer.ask_chatgpt({"question": input_text})
    responses.append(response["output"])
df['answer'] = responses
df['contexts'] = contexts_list

# Load the dataset from the CSV file
features = Features({
    "question": Value("string"),
    "ground_truth": Value("string"),
    "contexts": Sequence(Value("string")),
    "answer": Value("string")
})
dataset = Dataset.from_pandas(df, features=features)
print(dataset[0])

from ragas import evaluate

result = evaluate(
    dataset,
    metrics=[
        context_precision,
        faithfulness,
        answer_relevancy,
        context_recall,
    ],
)

print(result) # {'context_precision': 1.0000, 'faithfulness': 1.0000, 'answer_relevancy': 0.9744, 'context_recall': 0.9333}