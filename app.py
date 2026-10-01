import streamlit as st
import pandas as pd
import os
import tempfile
from graph import build_graph

st.set_page_config(page_title="AI Data Analysis Agent", layout="wide")

st.title("📊 AI Data Analysis Agent")
st.caption("Upload a CSV, ask a question in plain English, and get AI-generated analysis.")

# --- File upload ---
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    # Save uploaded file to a temp path so pandas can read it by path
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
        tmp.write(uploaded_file.getvalue())
        csv_path = tmp.name

    # Preview the data
    df_preview = pd.read_csv(csv_path)
    st.subheader("Preview of uploaded data")
    st.dataframe(df_preview.head(10))
    st.caption(f"Shape: {df_preview.shape[0]} rows × {df_preview.shape[1]} columns")

    # --- Question input ---
    query = st.text_input(
        "Ask a question about this data",
        placeholder="e.g. What is the total profit per region, and which region has the highest sales?"
    )

    if st.button("Run Analysis", type="primary") and query:
        with st.spinner("Running analysis..."):
            graph = build_graph()
            initial_state = {
                "csv_path": csv_path,
                "user_query": query,
                "dataset_profile": "",
                "generated_code": "",
                "execution_result": "",
                "final_insights": ""
            }
            final_state = graph.invoke(initial_state)

        st.success("Analysis complete")

        with st.expander("Generated Pandas Code", expanded=False):
            st.code(final_state.get("generated_code"), language="python")

        st.subheader("Result")
        st.write(final_state.get("execution_result"))

        st.subheader("AI Insights")
        st.write(final_state.get("final_insights"))

    # Clean up temp file after the session ends (optional, basic housekeeping)
    # os.unlink(csv_path)  # don't delete immediately, Streamlit may re-run the script

else:
    st.info("Upload a CSV file to get started.")