####### EDM CAT
BASE_PROMPT_EDM_CAT = """
Step 1. Read and understand the following text:
    {}

Step 2. Categorize it in one of the following categories:
    {}

***ONLY RESPOND WITH THE CATEGORY***
I'll tip you $200 for each correct categorization

Answer:
"""

BASE_PROMPT_EDM_QA = """
Step 1. Read and understand the following context:
    {}

Step 2. Answer the following question based on the context:
    {}

I'll tip you $200 if you answer correctly based on the context provided.

Answer:
"""


BASE_PROMPT_EDM_QA_NESTOR_ELMI = """
### INPUT:
{}
 
### TASK:
Given the above text passage, briefly answer the following question.
 
### QUESTION:
{}
 
### ANSWER:
"""

BASE_PROMPT_EDM_QA_NESTOR_CGPT = """
{}

Given the above context, answer the following question
{}
"""














EVAL_PROMPT_TEMPLATE_EDM_CAT = """You are a teacher grading a categorization quiz.
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

EVAL_PROMPT_TEMPLATE_EDM_QA = """You are a teacher grading a quiz.
You are given a question, the context the question is about, and the student's answer. You are asked to score the student's answer as either CORRECT or INCORRECT, based on the context.

Example Format:
QUESTION: question here
CONTEXT: context the question is about here
STUDENT ANSWER: student's answer here
GRADE: CORRECT or INCORRECT here

Grade the student answers based ONLY on their factual accuracy. Ignore differences in punctuation and phrasing between the student answer and true answer. It is OK if the student answer contains more information than the true answer, as long as it does not contain any conflicting statements. If the student answers says that the information is not specified or not mentioned, always grade it INCORRECT. Begin! 

QUESTION: {query}
CONTEXT: {context}
STUDENT ANSWER: {result}
GRADE:"""