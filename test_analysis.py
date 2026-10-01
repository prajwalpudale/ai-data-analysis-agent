import os
from graph import build_graph
from dotenv import load_dotenv

load_dotenv()

CSV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_data.csv")

def run():
    print("Initializing Data Analysis Agent...")
    graph = build_graph()

    query = "What is the total profit per region, and which region has the highest sales?"
    print(f"\nUser Query: {query}\n")

    initial_state = {
        "csv_path": CSV_PATH,
        "user_query": query,
        "dataset_profile": "",
        "generated_code": "",
        "execution_result": "",
        "final_insights": ""
    }

    print("Running Analysis...")
    final_state = graph.invoke(initial_state)

    print("\n" + "-"*40)
    print("GENERATED PANDAS CODE:")
    print("-"*40)
    print(final_state.get("generated_code"))

    print("\n" + "-"*40)
    print("RAW EXECUTION RESULT:")
    print("-"*40)
    print(final_state.get("execution_result"))

    print("\n" + "="*50)
    print("FINAL INSIGHTS REPORT:")
    print("="*50)
    print(final_state.get("final_insights"))

if __name__ == "__main__":
    run()