"""
BrainBiz Chat — neuroscience-backed business insights.

A simple Streamlit chat app that answers business questions
using a curated corpus of neuromarketing/neuroforecasting papers.

Usage: streamlit run brainbiz/app.py
"""

import streamlit as st
import anthropic
import chromadb
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

DB_DIR = Path(__file__).parent / "vectorstore"

SYSTEM_PROMPT = """You are BrainBiz, a neuroscience research assistant that helps business people understand what brain science says about their marketing, content, and product decisions.

You answer questions using findings from peer-reviewed neuroscience papers. For every claim you make, cite the specific paper(s) that support it.

RULES:
1. Always cite papers with author, year, and journal. Include the DOI link if available.
2. State the strength of evidence clearly: "one study found..." vs "multiple studies replicate..." vs "meta-analysis shows..."
3. Always mention experimental caveats: sample size, whether it's fMRI/EEG, lab vs. real-world, whether the finding has been replicated.
4. When the evidence is weak or the question is outside the corpus, say so honestly. Don't speculate beyond what the papers show.
5. Translate neuroscience jargon into plain language. The user is a businessperson, not a neuroscientist.
6. When relevant, mention the brain region and what it does in plain terms (e.g., "the nucleus accumbens, a reward-processing area deep in the brain").
7. Keep answers concise but thorough. Lead with the actionable insight, then support with evidence.
8. If asked about something the papers don't cover, say "The current research doesn't address this directly" rather than making something up.

FORMAT:
- Lead with a clear, actionable answer
- Support with 2-4 most relevant papers
- End with caveats/limitations
- Use markdown for readability"""


@st.cache_resource
def load_db():
    ef = DefaultEmbeddingFunction()
    client = chromadb.PersistentClient(path=str(DB_DIR))
    collection = client.get_collection(name="papers", embedding_function=ef)
    return collection


def retrieve(collection, query: str, n_results: int = 8) -> str:
    """Retrieve relevant papers and format as context for the LLM."""
    results = collection.query(query_texts=[query], n_results=n_results)

    context_parts = []
    for i in range(len(results["ids"][0])):
        meta = results["metadatas"][0][i]
        doc = results["documents"][0][i]
        dist = results["distances"][0][i] if results.get("distances") else None

        header = f"**Paper {i+1}** (cited {meta['citations']}x)"
        if meta.get("doi"):
            header += f" | [DOI]({meta['doi']})"
        if meta.get("oa_url"):
            header += f" | [Full text]({meta['oa_url']})"

        context_parts.append(f"{header}\n{doc}")

    return "\n\n---\n\n".join(context_parts)


def main():
    st.set_page_config(
        page_title="BrainBiz",
        page_icon="🧠",
        layout="wide",
    )

    st.title("BrainBiz")
    st.caption("Ask business questions, get neuroscience-backed answers from 650+ peer-reviewed papers")

    # Sidebar
    with st.sidebar:
        st.markdown("### About")
        st.markdown(
            "BrainBiz searches a curated corpus of neuromarketing, "
            "neuroforecasting, and consumer neuroscience papers to answer "
            "your business questions with cited evidence."
        )
        st.markdown("---")
        st.markdown("### Example questions")
        examples = [
            "How do I make my Kickstarter video more compelling?",
            "Does brain activity predict ad effectiveness better than surveys?",
            "What makes content go viral according to neuroscience?",
            "How does the brain decide whether to keep watching a video?",
            "What brain regions predict purchasing decisions?",
        ]
        for ex in examples:
            if st.button(ex, key=ex):
                st.session_state["pending_example"] = ex

        st.markdown("---")
        st.markdown(
            "### Coming Soon: Content Scoring\n"
            "Upload your video or ad and get brain-predicted "
            "attention, memorability, and engagement scores."
        )
        email = st.text_input("Get notified:", placeholder="your@email.com")
        if st.button("Notify me") and email:
            # TODO: store email somewhere (Supabase, sheet, etc.)
            st.success("We'll let you know when it's ready!")

    # Chat state
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Handle example button clicks
    if "pending_example" in st.session_state:
        prompt = st.session_state.pop("pending_example")
        st.session_state.messages.append({"role": "user", "content": prompt})

    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input
    if prompt := st.chat_input("Ask a business question..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

    # Generate response for the last user message
    if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
        user_msg = st.session_state.messages[-1]["content"]

        collection = load_db()
        context = retrieve(collection, user_msg)

        messages = []
        # Include recent chat history for context (last 6 messages)
        history = st.session_state.messages[:-1][-6:]
        for msg in history:
            messages.append({"role": msg["role"], "content": msg["content"]})

        messages.append({
            "role": "user",
            "content": f"""The user asked: {user_msg}

Here are the most relevant papers from our corpus:

{context}

Based on these papers, answer the user's question. Follow the rules in your system prompt.""",
        })

        with st.chat_message("assistant"):
            client = anthropic.Anthropic()
            with st.spinner("Searching 650+ neuroscience papers..."):
                response = client.messages.create(
                    model="claude-sonnet-4-20250514",
                    max_tokens=2048,
                    system=SYSTEM_PROMPT,
                    messages=messages,
                )
            answer = response.content[0].text
            st.markdown(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()
