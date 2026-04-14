"""
NeuroScript — Brain-optimized content writing for creators.

Paste your script or video idea, get a neuroscience-backed score
and rewrite suggestions that maximize hook, retention, and engagement.

Usage: streamlit run neuroscript/app.py
"""

import streamlit as st
import anthropic
import chromadb
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

# Reuse the BrainBiz vectorstore
DB_DIR = Path(__file__).parent.parent / "brainbiz" / "vectorstore"

SYSTEM_PROMPT = """You are NeuroScript, an AI content coach that helps creators write scripts optimized for how the human brain actually processes content. You combine insights from 650+ neuroscience papers with practical content creation advice.

## YOUR KNOWLEDGE (brain science that matters for content)

These are real, validated findings from neuroscience research:

**The 3-Second Hook Window:**
- The brain's reward system (nucleus accumbens) fires within the first 1-3 seconds and determines whether someone keeps watching
- Dorsal Attention Network (DAN) activation in the first 3 seconds correlates with share rate (r=0.26, p=0.02, n=77 TikToks)
- Neural "anticipatory affect" at video onset predicts both individual watch decisions AND aggregate view counts at the population level

**Attention & Retention:**
- Default Mode Network (DMN) suppression tracks viewer retention second-by-second — when DMN activates (mind-wandering), people leave
- Content that oscillates between high and low engagement (DMN oscillation range) predicts how long viewers watch (r=0.30, p<0.003)
- Inter-subject correlation (how synchronized brains are across viewers) reliably predicts attention (r=0.65 across 14 studies)
- Monotonic content loses — the brain habituates. Novelty/surprise reactivates attention networks

**Emotion & Virality:**
- Nucleus accumbens (reward) + mPFC (self-relevance) activation predicts viral sharing at the population level
- Content that triggers "self-referential processing" (TPJ, mPFC) gets shared more — people share what reflects their identity
- Emotional arousal drives memorability, but valence matters: high-arousal positive > high-arousal negative for sharing
- Audio-visual congruence (when what you see matches what you hear) triggers deeper brain processing

**Memory & Recall:**
- Hippocampal encoding during initial viewing predicts later recall — the brain decides in real-time what's worth remembering
- Narrative structure activates language networks more strongly than disconnected facts
- Unexpected information in familiar contexts triggers stronger memory encoding than purely novel content

**Multimodal Processing:**
- The brain processes video, audio, and text through separate pathways that converge at the temporal-parietal junction (TPJ)
- Content where audio reinforces visual (not just duplicates it) creates stronger brain responses than either alone
- Music/sound design affects visual cortex processing — the right audio literally changes how the brain sees your content

## HOW TO SCORE CONTENT

When the user shares a script or idea, score it on these 5 dimensions (each 1-10):

1. **Hook Power** — Will the first 3 seconds activate reward/attention circuits?
   - Does it create an open loop, pattern interrupt, or emotional spike?
   - Is there a visual or auditory surprise in the first moment?

2. **Retention Architecture** — Does the structure prevent DMN activation (mind-wandering)?
   - Are there tension/release cycles every 5-10 seconds?
   - Does it avoid monotonic pacing?
   - Are there "micro-hooks" that re-engage attention throughout?

3. **Emotional Resonance** — Does it activate self-referential and reward processing?
   - Will viewers see themselves in this content?
   - Is there genuine emotional arousal (not just shock)?
   - Does it trigger identity-signaling ("I want to share this because it says something about me")?

4. **Memory Encoding** — Will viewers remember this tomorrow?
   - Is there a clear narrative arc or surprising payload?
   - Does it use concrete/sensory language vs abstract?
   - Is there a "sticky" moment that anchors recall?

5. **Multimodal Synergy** — Do audio, visual, and text reinforce each other?
   - Does the audio add information beyond the visual?
   - Is there audio-visual congruence at key moments?
   - Would this work as well on mute? (If yes, audio is wasted)

## RULES

1. Be direct and specific. Say "Change your first line from X to Y" not "Consider making your hook stronger."
2. Always explain WHY in brain terms, but keep it one sentence. "This works because it creates an open loop that prevents your viewer's default mode network from disengaging."
3. Give the score FIRST, then the breakdown, then specific rewrites.
4. When rewriting, preserve the creator's voice and style. You're optimizing structure, not personality.
5. Be honest about scores. Most scripts are 4-6. A 9-10 is exceptional. Don't inflate.
6. If a script is genuinely good, say so and suggest minor tweaks rather than rewriting everything.
7. Use casual, creator-friendly language. No academic jargon without immediate translation.
8. When you cite brain science, keep it to one line: "Research shows X (brain region, what it does in plain English)."
9. For video ideas (not full scripts), focus on structural suggestions rather than line-by-line rewrites.
10. Always end with the single most impactful change they could make."""


@st.cache_resource
def load_db():
    ef = DefaultEmbeddingFunction()
    client = chromadb.PersistentClient(path=str(DB_DIR))
    collection = client.get_collection(name="papers", embedding_function=ef)
    return collection


def retrieve(collection, query: str, n_results: int = 6) -> str:
    results = collection.query(query_texts=[query], n_results=n_results)
    context_parts = []
    for i in range(len(results["ids"][0])):
        meta = results["metadatas"][0][i]
        doc = results["documents"][0][i]
        header = f"**Paper {i+1}** — {meta['title']} ({meta['year']}, cited {meta['citations']}x)"
        context_parts.append(f"{header}\n{doc}")
    return "\n\n---\n\n".join(context_parts)


def main():
    st.set_page_config(
        page_title="NeuroScript",
        page_icon="\u26a1",
        layout="wide",
    )

    # Custom CSS
    st.markdown("""
    <style>
    .stApp { max-width: 900px; margin: 0 auto; }
    .score-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 12px; padding: 20px; color: white; margin: 10px 0;
    }
    .score-number { font-size: 48px; font-weight: 800; }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("# \u26a1 NeuroScript")
    st.markdown("**Write content your viewer's brain can't scroll past.**")
    st.caption("Powered by 650+ peer-reviewed neuroscience papers on attention, memory, and engagement.")

    # Sidebar
    with st.sidebar:
        st.markdown("### How it works")
        st.markdown(
            "NeuroScript scores your content on 5 brain-based dimensions "
            "and rewrites weak spots using principles from cognitive neuroscience."
        )

        st.markdown("### Scoring dimensions")
        st.markdown("""
        1. **Hook Power** — First 3 seconds
        2. **Retention Architecture** — Pacing & structure
        3. **Emotional Resonance** — Identity & reward
        4. **Memory Encoding** — Will they remember?
        5. **Multimodal Synergy** — Audio + visual
        """)

        st.markdown("---")
        st.markdown("### Quick start")
        st.markdown("Paste a script, describe a video idea, or ask how to improve a specific moment in your content.")

        st.markdown("---")

        content_type = st.selectbox(
            "Content type",
            ["TikTok / Reel", "YouTube Short", "YouTube Long-form", "Ad / Promo", "Podcast Hook"],
            index=0,
        )

        target = st.selectbox(
            "Goal",
            ["Maximum retention", "Drive shares / virality", "Build trust / authority", "Drive action / CTA"],
            index=0,
        )

    # Chat state
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Example prompts if empty
    if not st.session_state.messages:
        st.markdown("### Try one of these:")
        cols = st.columns(2)
        examples = [
            ("Score my hook", "\"Stop scrolling if you've ever wondered why some videos blow up and others flop. I spent 6 months studying the algorithm and here's what nobody tells you...\""),
            ("Improve this script", "Hey guys, today I want to talk about 3 productivity tips that changed my life. Number one is waking up early. Number two is time blocking. Number three is the two minute rule. Try these out and let me know in the comments!"),
            ("Video idea", "I want to make a TikTok about why coffee makes you crash at 2pm. I'm a nutritionist. What's the most brain-engaging way to structure this?"),
            ("Fix my retention", "My TikToks get great hook rates (95%+ first 3 seconds) but people drop off around 8-10 seconds. My videos are usually 30-45 seconds. What's happening?"),
        ]
        for i, (label, example) in enumerate(examples):
            with cols[i % 2]:
                if st.button(f"**{label}**\n\n{example[:60]}...", key=f"ex_{i}", use_container_width=True):
                    st.session_state.messages.append({"role": "user", "content": example})
                    st.rerun()

    # Chat input
    if prompt := st.chat_input("Paste your script, describe your video idea, or ask a question..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

    # Generate response
    if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
        user_msg = st.session_state.messages[-1]["content"]

        # Retrieve relevant papers
        collection = load_db()
        search_query = f"brain attention engagement retention {user_msg[:200]}"
        context = retrieve(collection, search_query)

        # Build messages with context
        messages = []
        history = st.session_state.messages[:-1][-6:]
        for msg in history:
            messages.append({"role": msg["role"], "content": msg["content"]})

        messages.append({
            "role": "user",
            "content": f"""Content type: {content_type}
Goal: {target}

User's message:
{user_msg}

---

Relevant neuroscience research for reference (use to support your suggestions):

{context}

---

If the user shared a script or hook, score it on the 5 dimensions (1-10 each) and give specific rewrite suggestions. If they asked a question, answer it with actionable, brain-backed advice. Be direct and specific.""",
        })

        with st.chat_message("assistant"):
            client = anthropic.Anthropic()
            with st.spinner("Analyzing with neuroscience..."):
                response = client.messages.create(
                    model="claude-sonnet-4-20250514",
                    max_tokens=3000,
                    system=SYSTEM_PROMPT,
                    messages=messages,
                )
            answer = response.content[0].text
            st.markdown(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()
