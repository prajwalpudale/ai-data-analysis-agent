import os
import pandas as pd
from typing import Dict, TypedDict
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END

load_dotenv()

class DataState(TypedDict):
    csv_path: str
    user_query: str
    dataset_profile: str
    generated_code: str
    execution_result: str
    final_insights: str

def profile_dataset(state: DataState) -> Dict:
    df = pd.read_csv(state['csv_path'])
    profile = f"Columns: {', '.join(df.columns)}\n"
    profile += f"Shape: {df.shape}\n"
    profile += f"Sample data:\n{df.head(2).to_string()}"
    return {"dataset_profile": profile}

def generate_code(state: DataState) -> Dict:
    llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0)
    prompt = (
        f"Dataset Profile:\n{state['dataset_profile']}\n\n"
        f"User Question: {state['user_query']}\n\n"
        f"Write Python pandas code to answer the user question. "
        f"The dataframe is loaded as `df = pd.read_csv('{state['csv_path']}')`. "
        f"Store the final answer string in a variable named `result_str`. "
        f"Output ONLY valid python code."
    )
    response = llm.invoke([
        SystemMessage(content="You are a data scientist."),
        HumanMessage(content=prompt)
    ])

    code = response.content
    if code.startswith("```python"):
        code = code[9:]
    if code.endswith("```"):
        code = code[:-3]

    return {"generated_code": code.strip()}

def execute_code(state: DataState) -> Dict:
    # WARNING: Using exec() is unsafe in production. This is a simplified sandbox.
    code = state['generated_code']
    local_env = {}
    try:
        exec(code, globals(), local_env)
        result = local_env.get('result_str', "Code executed but `result_str` not found.")
    except Exception as e:
        result = f"Error executing code: {e}"

    return {"execution_result": str(result)}

def interpret_results(state: DataState) -> Dict:
    llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0.4)
    prompt = (
        f"User Question: {state['user_query']}\n"
        f"Data Output: {state['execution_result']}\n\n"
        f"Write a 3-4 sentence professional narrative insight explaining these findings."
    )
    response = llm.invoke([
        SystemMessage(content="You are a senior data analyst."),
        HumanMessage(content=prompt)
    ])
    return {"final_insights": response.content}

def build_graph():
    workflow = StateGraph(DataState)

    workflow.add_node("profile", profile_dataset)
    workflow.add_node("generate", generate_code)
    workflow.add_node("execute", execute_code)
    workflow.add_node("interpret", interpret_results)

    workflow.set_entry_point("profile")
    workflow.add_edge("profile", "generate")
    workflow.add_edge("generate", "execute")
    workflow.add_edge("execute", "interpret")
    workflow.add_edge("interpret", END)

    return workflow.compile()