# Generative AI with LangChain & Hugging Face

Hands-on notes and code from Krish Naik's *Complete Generative AI Course With Langchain and Huggingface*.

**Author:** Ishwar ([@IshwarRajChauhan](https://github.com/IshwarRajChauhan))
**Status:** 🚧 In progress

## Goal

Build the LangChain fundamentals needed for the AI Engineer track (agents, RAG, LangGraph next).

## Stack

- Python 3.11+
- LangChain, LangSmith
- Hugging Face (Transformers, Hub)
- OpenAI / Groq / Ollama (LLM providers)
- Vector stores: FAISS, Chroma
- Streamlit (UI)

## Structure

```
.
├── 01-basics/            # LLM calls, prompts, chains
├── 02-rag/               # loaders, splitters, embeddings, retrievers
├── 03-huggingface/       # HF models, pipelines, embeddings
├── 04-projects/          # end-to-end apps
├── requirements.txt
├── .env.example
└── README.md
```

## Progress

- [ ] LangChain basics (prompts, chains, output parsers)
- [ ] Data ingestion, text splitting, embeddings
- [ ] Vector stores and retrievers
- [ ] RAG pipeline
- [ ] Hugging Face integration
- [ ] Mini projects / deployment

## Setup

```bash
git clone https://github.com/IshwarRajChauhan/<repo-name>.git
cd <repo-name>
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env            # add your API keys
```

`.env` keys:

```
OPENAI_API_KEY=
GROQ_API_KEY=
HF_TOKEN=
LANGCHAIN_API_KEY=
```

## Key Learnings

- _Add one-liners per section as you finish them._

## Next

- Agentic AI Bootcamp (LangGraph)
- RAG Bootcamp (LangSmith evals)

## Credits

Course by [Krish Naik](https://www.youtube.com/@krishnaik06).
