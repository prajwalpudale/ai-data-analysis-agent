# AI Data Analysis Agent

An AI-powered agent that analyzes CSV data using natural language. Upload any dataset, ask a question in plain English, and get AI-generated pandas code, the computed result, and a narrative insight — all through a simple web interface.

## Demo

Upload a CSV → Ask a question → Get code, results, and AI-written insights.

## Features

- **Natural language data analysis** — no need to write pandas code yourself
- **Transparent reasoning** — see the exact generated code before trusting the result
- **Automated insight generation** — a second LLM pass turns raw output into a readable explanation
- **Simple web interface** — built with Streamlit, no setup needed beyond running one command

## Tech Stack

- **Python**
- **LangGraph** — orchestrates the multi-step agent workflow (profile → generate → execute → interpret)
- **LangChain** — LLM integration layer
- **Groq API** — fast LLM inference (OpenAI GPT-OSS models via Groq)
- **Streamlit** — web interface
- **Pandas** — data handling and generated code execution

## How It Works

This project uses LangGraph to orchestrate a 4-step agent pipeline:

1. **Profile** — reads the uploaded CSV and summarizes its structure (columns, shape, sample rows)
2. **Generate** — an LLM writes pandas code to answer the user's specific question, grounded in the dataset profile
3. **Execute** — the generated code runs in a sandboxed environment and produces a result
4. **Interpret** — a second LLM call turns the raw result into a clear, professional explanation
