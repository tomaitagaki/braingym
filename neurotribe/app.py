"""
NeuroTribe — Interpretable TRIBEv2 brain encoding.

Upload a stimulus (video, audio, text), get predicted fMRI,
visualize the brain response, and chat about what it means.

Usage: streamlit run neurotribe/app.py
"""

import streamlit as st
import anthropic
import chromadb
import numpy as np
import json
import logging
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("neurotribe")

# Reuse BrainBiz vectorstore for paper retrieval
DB_DIR = Path(__file__).parent.parent / "brainbiz" / "vectorstore"

SYSTEM_PROMPT = """You are NeuroTribe, an AI neuroscience interpreter that explains predicted brain responses to content.

You have access to TRIBEv2, a state-of-the-art brain encoding model from Meta AI that predicts how the average human brain would respond to any video, audio, or text stimulus. The model predicts fMRI activity across 20,484 cortical vertices at 1Hz (1 prediction per second).

## BRAIN NETWORKS (Schaefer 7-Network Parcellation)

| Network | Abbreviation | Function | Content interpretation |
|---------|-------------|----------|----------------------|
| Dorsal Attention | DAN | Focused, top-down attention | High = viewer is actively paying attention |
| Default Mode | DMN | Mind-wandering, self-reflection | High = viewer is disengaged/zoning out. LOW DMN = good engagement |
| Frontoparietal Control | FPN | Executive control, cognitive effort | High = content requires thinking/processing |
| Ventral Attention / Salience | VAN | Surprise, salience detection | High = something unexpected happened |
| Visual | Vis | Visual processing | High = visually stimulating content |
| Somatomotor | SomMot | Motor/physical processing | High = content involves physical movement |
| Limbic | Limbic | Emotion, reward | High = emotionally evocative content |

## KEY DERIVED METRICS

- **Attention (DAN - DMN)**: Positive = engaged, negative = mind-wandering. This is the most important single metric.
- **Engagement (-DMN)**: How suppressed the default mode network is. More suppression = more engagement.
- **Cognitive Load (FPN + DAN - DMN)**: How mentally demanding the content is.
- **DMN oscillation range**: How much engagement varies. Higher range → viewers who stay watch longer (r=0.30, p<0.003).

## IMPORTANT CAVEATS (always mention relevant ones)

1. **5-second hemodynamic lag**: TRIBE's prediction at time t reflects the stimulus at t-5 seconds. When interpreting "the brain response at t=5," that's actually about what happened at t=0.
2. **Population average**: This is the predicted response of an average healthy adult brain. Individual responses vary.
3. **Passive viewing**: The model was trained on passive viewing in an MRI scanner, not active phone scrolling.
4. **Correlation ≠ causation**: Brain network activation patterns are correlated with engagement metrics, but we can't say "high DAN causes retention."
5. **Effect sizes are modest**: After controlling for shared temporal trends, brain-retention correlations are real but modest (r~0.10). The brain signal adds information beyond content features, but it's not a crystal ball.

## HOW TO INTERPRET FOR TIKTOK/CONTENT

Based on our research (n=77 TikToks, n=97 Tsinghua retention videos):

- **First 3 seconds attention (DAN-DMN)** predicts share rate (r=+0.26, p=0.02)
- **Peak attention** correlates negatively with engagement rate (r=-0.35, p=0.002) — attention-grabbing content gets fewer likes per view due to reach dilution
- **-DMN with 5s lag** tracks retention curves (median r=0.72 raw, r~0.10 after detrending for shared trends)
- **DMN oscillation range** predicts half_life (r=+0.30, p<0.003) — content that cycles between high and low engagement keeps people watching

## CITATION FORMAT

You MUST use inline numbered citations like Perplexity. When you reference a finding from the provided papers, cite it as [1], [2], etc. At the END of your response, include a "Sources" section listing each cited paper with its number, title, authors, year, and DOI link.

Example inline: "The brain's reward system activates within the first 3 seconds of viewing [1], and this early response predicts whether content gets shared [2]."

Example sources section:
**Sources**
[1] Neural Predictors of Purchases — Knutson et al., 2007 [DOI](https://doi.org/...)
[2] Brain activity forecasts video engagement — Tong et al., 2020 [DOI](https://doi.org/...)

Only cite papers that were provided in the context. Do not fabricate citations. If no paper is relevant to a claim, don't cite it — just state it as a finding from "our research" or general neuroscience knowledge.

## RULES

1. Always start by describing what the brain response looks like in plain language.
2. Highlight the most interesting/unusual patterns — don't just list all 7 networks.
3. When asked "how would this do on TikTok?", give a specific, honest assessment based on the metrics above.
4. Point out specific timestamps where interesting things happen in the brain response.
5. Always mention the hemodynamic lag when discussing timing.
6. If the response is flat/uninteresting, say so — don't oversell.
7. Cite relevant papers inline using numbered references [1], [2], etc. Include a Sources section at the end."""


@st.cache_resource
def load_paper_db():
    if DB_DIR.exists():
        ef = DefaultEmbeddingFunction()
        client = chromadb.PersistentClient(path=str(DB_DIR))
        return client.get_collection(name="papers", embedding_function=ef)
    return None


def retrieve_papers(collection, query: str, n_results: int = 4) -> str:
    if collection is None:
        return ""
    results = collection.query(query_texts=[query], n_results=n_results)
    parts = []
    for i in range(len(results["ids"][0])):
        meta = results["metadatas"][0][i]
        doc = results["documents"][0][i]
        doi = meta.get("doi", "")
        doi_link = f" | DOI: {doi}" if doi else ""
        parts.append(f"[Paper {i+1}] {meta['title']} ({meta['year']}, {meta['citations']} citations{doi_link})\n{doc}")
    return "\n\n---\n\n".join(parts)


def call_tribe_modal(file_bytes: bytes, filename: str, modality: str) -> dict:
    """Call Modal endpoint to run TRIBE inference."""
    import modal
    cls = modal.Cls.from_name("neurotribe-api", "TribeEndpoint")
    result = cls().predict.remote(
        file_bytes=file_bytes,
        filename=filename,
        modality=modality,
    )
    return result


def format_brain_summary(result: dict) -> str:
    """Format brain response as a readable summary for the chat."""
    nets = result["net_means"]
    n_tp = result["n_timepoints"]

    summary = f"""## Brain Response Summary

**Duration:** {n_tp} seconds | **Modality:** {result['modality']} | **Inference time:** {result['elapsed_seconds']}s

### Network Activation (mean across time)
| Network | Activation | Role |
|---------|-----------|------|
| DAN (attention) | {nets['DAN']:+.4f} | Focused attention |
| DMN (default mode) | {nets['DMN']:+.4f} | Mind-wandering (low = good) |
| FPN (control) | {nets['FPN']:+.4f} | Cognitive effort |
| VAN (salience) | {nets['VAN']:+.4f} | Surprise detection |
| Vis (visual) | {nets['Vis']:+.4f} | Visual processing |
| SomMot (motor) | {nets['SomMot']:+.4f} | Physical processing |
| Limbic (emotion) | {nets['Limbic']:+.4f} | Emotional response |

### Derived Metrics
- **Attention (DAN-DMN):** {result['attention']:+.4f}
- **Engagement (-DMN):** {result['engagement']:+.4f}
- **Cognitive Load:** {result['cognitive_load']:+.4f}
"""
    # Attention dynamics
    attn_ts = np.array(result["attention_ts"])
    summary += f"""
### Temporal Dynamics
- **First 3s attention:** {np.mean(attn_ts[:min(3, len(attn_ts))]):+.4f}
- **Peak attention:** {np.max(attn_ts):+.4f} at t={np.argmax(attn_ts)}s
- **Min attention:** {np.min(attn_ts):+.4f} at t={np.argmin(attn_ts)}s
- **Attention variability (std):** {np.std(attn_ts):.4f}
- **DMN oscillation range:** {np.ptp(np.array(result['timeseries']['DMN'])):.4f}
"""
    return summary


def _run_encoding(file_bytes, filename, modality):
    """Run TRIBE encoding with progress bar. Returns result or None."""
    import threading, time as _time

    progress = st.progress(0, text="Uploading to Modal GPU...")
    _result_holder = [None]
    _error_holder = [None]

    def _run():
        try:
            _result_holder[0] = call_tribe_modal(file_bytes, filename, modality)
        except Exception as e:
            _error_holder[0] = e

    t = threading.Thread(target=_run)
    t.start()

    _steps = [
        (5, "Uploading to Modal GPU..."),
        (15, "Spinning up A10G GPU..."),
        (30, "Loading TRIBEv2 model (V-JEPA2 + LLaMA + Wav2Vec)..."),
        (50, "Extracting video/audio/text features..."),
        (65, "Running transformer encoder..."),
        (80, "Computing brain network timeseries..."),
        (90, "Packaging results..."),
    ]
    _step_idx = 0
    _elapsed = 0
    while t.is_alive():
        _time.sleep(1)
        _elapsed += 1
        while _step_idx < len(_steps) - 1 and _elapsed > _steps[_step_idx + 1][0] * 1.5:
            _step_idx += 1
        _pct = min(_steps[_step_idx][0] + (_elapsed % 10), 95)
        progress.progress(_pct / 100, text=_steps[_step_idx][1])

    t.join()
    progress.progress(1.0, text="Done!")
    _time.sleep(0.3)
    progress.empty()

    if _error_holder[0]:
        st.error(f"Error calling TRIBE: {_error_holder[0]}")
        return None
    if _result_holder[0] and "error" in _result_holder[0]:
        st.error(f"TRIBE error: {_result_holder[0]['error']}")
        return None
    return _result_holder[0]


def _render_viz(msg, result):
    """Render embedded visualizations in a chat message."""
    for viz_type in msg.get("viz", []):
        if viz_type == "summary":
            st.markdown(format_brain_summary(result))
        elif viz_type == "networks":
            from neurotribe.brain_viz import render_network_timeseries
            fig = render_network_timeseries(
                result["timeseries"], attention_ts=result["attention_ts"],
            )
            st.pyplot(fig)
            plt.close(fig)
        elif viz_type == "brain":
            if result.get("viz_preds"):
                viz_indices = result["viz_timepoint_indices"]
                t_idx = msg.get("brain_t", viz_indices[len(viz_indices) // 2])
                if t_idx in viz_indices:
                    idx = viz_indices.index(t_idx)
                    vertex_data = np.array(result["viz_preds"][idx])
                    from neurotribe.brain_viz import render_brain_surface
                    fig = render_brain_surface(
                        vertex_data,
                        title=f"Brain at t={t_idx}s (stimulus at t={max(0, t_idx-5)}s)",
                    )
                    st.pyplot(fig)
                    plt.close(fig)


def _generate_response(result, user_msg):
    """Generate an LLM response with optional brain viz. Returns (answer, viz_list, brain_t)."""
    brain_summary = format_brain_summary(result)

    collection = load_paper_db()
    papers = retrieve_papers(collection, f"brain {user_msg[:200]}")

    # Build message history
    api_messages = []
    history = st.session_state.get("messages", [])[:-1][-6:]
    for msg in history:
        if msg["content"] not in ("__INITIAL_ANALYSIS__", "__UPLOAD__"):
            api_messages.append({"role": msg["role"], "content": msg["content"]})

    api_messages.append({
        "role": "user",
        "content": f"""Here is the predicted brain response for the user's content:

{brain_summary}

Full network timeseries (JSON):
DAN: {json.dumps(result['timeseries']['DAN'][:20])}{'...' if len(result['timeseries']['DAN']) > 20 else ''}
DMN: {json.dumps(result['timeseries']['DMN'][:20])}{'...' if len(result['timeseries']['DMN']) > 20 else ''}
VAN: {json.dumps(result['timeseries']['VAN'][:20])}{'...' if len(result['timeseries']['VAN']) > 20 else ''}
FPN: {json.dumps(result['timeseries']['FPN'][:20])}{'...' if len(result['timeseries']['FPN']) > 20 else ''}

Relevant neuroscience research:
{papers}

User's question: {user_msg}

Answer based on the actual brain response data above. Be specific about timestamps, network values, and patterns. Remember the 5-second hemodynamic lag.""",
    })

    # Brain surface if requested
    _q = user_msg.lower()
    wants_brain = any(w in _q for w in [
        "brain surface", "show brain", "show me the brain", "visualize",
        "brain map", "cortical", "which areas", "brain at", "peak attention",
        "surface", "brain render", "fmri", "activation map",
    ])
    viz_list = []
    brain_t = None

    if wants_brain and result.get("viz_preds"):
        attn_ts = np.array(result["attention_ts"])
        peak_t = int(np.argmax(attn_ts))
        viz_indices = result["viz_timepoint_indices"]
        brain_t = min(viz_indices, key=lambda x: abs(x - peak_t))
        viz_list.append("brain")

    client = anthropic.Anthropic()
    with st.spinner("Interpreting..."):
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2500,
            system=SYSTEM_PROMPT,
            messages=api_messages,
        )
    return response.content[0].text, viz_list, brain_t


SIGNALS = [
    ("DAN-DMN", "Attention", "Fox et al. 2005, PNAS", "https://doi.org/10.1073/pnas.0504136102",
     "DAN and DMN are intrinsically anticorrelated. When focused attention increases, mind-wandering decreases."),
    ("-DMN", "Engagement", "Raichle et al. 2001, PNAS", "https://doi.org/10.1073/pnas.98.2.676",
     "Named the Default Mode Network. DMN suppression scales with how engaged the viewer is."),
    ("VAN", "Surprise", "Corbetta & Shulman 2002, Nat Rev Neurosci", "https://doi.org/10.1038/nrn755",
     "Ventral attention network acts as a 'circuit breaker' for unexpected, salient stimuli."),
    ("FPN", "Cognitive load", "Duncan & Owen 2000, Trends Neurosci", "https://doi.org/10.1016/S0166-2236(00)01633-7",
     "Frontoparietal regions respond to cognitive difficulty regardless of task type."),
]


def main():
    st.set_page_config(page_title="NeuroTribe", page_icon="\U0001f9e0", layout="centered")

    # ── Styles ─────────────────────────────────────────────────────────
    st.markdown("""<style>
    /* Typography */
    body, .stApp { -webkit-font-smoothing: antialiased; }
    h1, h2, h3 { text-wrap: balance; letter-spacing: -0.02em; }

    /* Layout */
    .block-container { max-width: 780px; padding-top: 2.5rem; padding-bottom: 1rem; }

    /* Hide file uploader decoration, keep functional */
    [data-testid="stFileUploader"] {
        border: 1.5px dashed #d0d0d0;
        border-radius: 10px;
        padding: 0.6rem;
        transition: border-color 0.15s ease;
    }
    [data-testid="stFileUploader"]:hover { border-color: #888; }
    [data-testid="stFileUploader"] section > div:first-child { display: none; }

    /* Sidebar */
    [data-testid="stSidebar"] { padding-top: 1.5rem; }
    [data-testid="stSidebar"] .block-container { padding-top: 0; }

    /* Signal items in sidebar */
    .sig-item {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 6px 0;
        border-bottom: 1px solid rgba(0,0,0,0.05);
        font-size: 0.88rem;
        line-height: 1.4;
    }
    .sig-item:last-child { border-bottom: none; }
    .sig-code {
        font-family: 'SF Mono', ui-monospace, monospace;
        font-size: 0.8rem;
        font-weight: 600;
        color: #1a1a1a;
        background: #f0f0f0;
        padding: 1px 6px;
        border-radius: 4px;
        white-space: nowrap;
    }
    .sig-meaning { color: #666; margin-left: 6px; flex: 1; }
    .sig-info {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 20px;
        height: 20px;
        border-radius: 50%;
        border: 1px solid #ccc;
        font-size: 0.7rem;
        color: #888;
        text-decoration: none;
        flex-shrink: 0;
        margin-left: 8px;
        transition: border-color 0.15s ease, color 0.15s ease;
    }
    .sig-info:hover { border-color: #666; color: #333; }

    /* Prompt suggestion pills */
    .prompt-grid {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin: 1rem 0 1.5rem;
    }
    .prompt-pill {
        font-size: 0.85rem;
        color: #444;
        padding: 8px 14px;
        border: 1px solid #e0e0e0;
        border-radius: 20px;
        cursor: pointer;
        transition: border-color 0.15s ease, background-color 0.15s ease;
        line-height: 1.3;
    }
    .prompt-pill:hover { border-color: #999; background: #fafafa; }

    /* Hero section */
    .hero-sub {
        font-size: 1.05rem;
        color: #555;
        line-height: 1.6;
        margin: 0.5rem 0 0.25rem;
    }
    .hero-meta {
        font-size: 0.78rem;
        color: #999;
        letter-spacing: 0.02em;
        margin-bottom: 1.5rem;
    }
    </style>""", unsafe_allow_html=True)

    # ── Sidebar ────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("**Key signals**")
        for code, meaning, paper, doi, detail in SIGNALS:
            with st.popover(f"`{code}` = {meaning}"):
                st.markdown(f"**{paper}**")
                st.markdown(detail)
                st.link_button("View paper", doi)

        st.divider()
        if st.button("New session", use_container_width=True):
            for k in ["messages", "tribe_result", "_last_upload"]:
                st.session_state.pop(k, None)
            st.rerun()

    # ── State ──────────────────────────────────────────────────────────
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    has_result = "tribe_result" in st.session_state

    # ── Chat history ──────────────────────────────────────────────────
    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            if msg["content"] == "__UPLOAD__":
                st.markdown(f"Uploaded **{msg.get('filename', 'file')}** ({msg.get('modality', 'video')})")
                if msg.get("user_message"):
                    st.markdown(msg["user_message"])
                continue

            result = st.session_state.get("tribe_result")
            if result and msg.get("viz"):
                _render_viz(msg, result)

            if msg["content"] not in ("__INITIAL_ANALYSIS__", "__UPLOAD__"):
                st.markdown(msg["content"])

    # ── Hero (empty state) ─────────────────────────────────────────────
    if not st.session_state["messages"]:
        st.markdown("## NeuroTribe")
        st.markdown('<p class="hero-sub">See how the brain responds to your content.<br>Upload a video, audio, or script and ask a question about it.</p>', unsafe_allow_html=True)
        st.markdown('<p class="hero-meta">TRIBEv2 (Meta AI) on Modal GPUs &middot; 650 neuroscience papers</p>', unsafe_allow_html=True)

    # ── Post-encode suggestions ────────────────────────────────────────
    if has_result and len(st.session_state["messages"]) == 2:
        cols = st.columns(3)
        for i, q in enumerate([
            "How would this do on TikTok?",
            "When does attention drop?",
            "Boost engagement?",
            "Show me the brain surface at peak attention",
            "Explain for a creator",
            "Most interesting pattern?",
        ]):
            with cols[i % 3]:
                if st.button(q, key=f"post_{i}", use_container_width=True):
                    st.session_state["messages"].append({"role": "user", "content": q})
                    st.rerun()

    # ── First-time upload (hero state only) ──────────────────────────
    if not st.session_state["messages"]:
        st.file_uploader(
            "Drop a video, audio, or text file",
            type=["mp4", "mov", "avi", "webm", "mp3", "wav", "m4a", "txt", "mpeg4"],
            key="file_upload",
        )
        cols = st.columns(2)
        for i, q in enumerate([
            "What brain signals predict TikTok retention?",
            "How does the brain decide to keep watching?",
            "What makes a hook work, neurologically?",
            "Does emotional content actually perform better?",
        ]):
            with cols[i % 2]:
                if st.button(q, key=f"pre_{i}", use_container_width=True):
                    st.session_state["messages"].append({"role": "user", "content": q})
                    st.rerun()

    # ── Compact attach button (after first message) ────────────────────
    if st.session_state["messages"]:
        with st.popover("+ Attach new content", use_container_width=True):
            st.file_uploader(
                "Upload video, audio, or text",
                type=["mp4", "mov", "avi", "webm", "mp3", "wav", "m4a", "txt", "mpeg4"],
                key="file_upload",
            )
            st.caption("Attach a file, then type your question below and hit enter.")

    # ── Chat input ─────────────────────────────────────────────────────
    pending_file = st.session_state.get("file_upload")
    has_pending = pending_file is not None and hasattr(pending_file, "name") and pending_file.name != st.session_state.get("_last_upload")

    if has_pending:
        placeholder = f"Ask about {pending_file.name}..."
    elif has_result:
        placeholder = "Ask about the brain response..."
    else:
        placeholder = "Ask a question or attach content above..."

    prompt = st.chat_input(placeholder)

    if not prompt:
        return

    log.info(f"User prompt: {prompt[:80]}...")

    # Re-check file state after prompt
    uploaded_file = st.session_state.get("file_upload")
    is_new_upload = (
        uploaded_file is not None
        and hasattr(uploaded_file, "name")
        and uploaded_file.name != st.session_state.get("_last_upload")
    )

    if is_new_upload:
        # ── Upload + encode + answer ───────────────────────────────
        log.info(f"New upload: {uploaded_file.name}")
        st.session_state["_last_upload"] = uploaded_file.name
        file_bytes = uploaded_file.read()
        filename = uploaded_file.name
        ext = filename.rsplit(".", 1)[-1].lower()
        modality = "video" if ext in ("mp4", "mov", "avi", "webm", "mpeg4") else "audio" if ext in ("mp3", "wav", "m4a") else "text"

        st.session_state["messages"].append({
            "role": "user", "content": "__UPLOAD__",
            "filename": filename, "modality": modality,
            "user_message": prompt,
        })

        log.info(f"Encoding {filename} ({modality})...")
        result = _run_encoding(file_bytes, filename, modality)

        if result:
            log.info(f"Encoding done: {result['n_timepoints']} timepoints in {result['elapsed_seconds']}s")
            st.session_state["tribe_result"] = result

            # Generate answer to user's question
            log.info(f"Generating response for: {prompt[:60]}...")
            answer, viz_list, brain_t = _generate_response(result, prompt)
            log.info(f"Response generated ({len(answer)} chars), viz={viz_list}")

            combined_msg = {
                "role": "assistant",
                "content": answer,
                "viz": ["summary", "networks"],
            }
            if viz_list:
                combined_msg["viz"].extend(viz_list)
                combined_msg["brain_t"] = brain_t
            st.session_state["messages"].append(combined_msg)
        else:
            log.error("Encoding failed")

        st.rerun()

    else:
        # ── Follow-up chat (no new file) ───────────────────────────
        log.info(f"Follow-up prompt (has_result={has_result})")
        st.session_state["messages"].append({"role": "user", "content": prompt})

        result = st.session_state.get("tribe_result")

        if result:
            log.info("Generating response with brain data...")
            answer, viz_list, brain_t = _generate_response(result, prompt)
            log.info(f"Response generated ({len(answer)} chars), viz={viz_list}")

            new_msg = {"role": "assistant", "content": answer}
            if viz_list:
                new_msg["viz"] = viz_list
                new_msg["brain_t"] = brain_t
            st.session_state["messages"].append(new_msg)
        else:
            log.info("No brain data, answering from papers only...")
            collection = load_paper_db()
            papers = retrieve_papers(collection, f"brain {prompt[:200]}", n_results=6)
            client = anthropic.Anthropic()
            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=2000,
                system=SYSTEM_PROMPT + "\n\nThe user has NOT uploaded content yet. Answer from neuroscience knowledge and papers. Suggest uploading content for specific analysis.",
                messages=[{"role": "user", "content": f"Papers:\n{papers}\n\nQuestion: {prompt}"}],
            )
            answer = response.content[0].text
            log.info(f"Paper-only response generated ({len(answer)} chars)")
            st.session_state["messages"].append({"role": "assistant", "content": answer})

        st.rerun()




# Need this import at module level for matplotlib in viz functions
import matplotlib.pyplot as plt

if __name__ == "__main__":
    main()
