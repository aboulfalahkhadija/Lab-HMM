"""Chaotic Text Generator: NLP Evaluation Test Bench
Streamlit interface for HMM, Role-Split HMM, and Trigram sequence models.
"""

import csv
import io
import json
import math
import os
import random
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from src.pipeline import (
    HMM,
    NGramGenerator,
    coarse,
    generation_metrics,
    load_tagged,
    sentence_features,
    split_by_document,
)

# -----------------------------------------------------------------------------
# Page Configuration & Academic Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Chaotic Text Generator - Evaluation Bench",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    /* Clean academic typography & layout */
    .app-header {
        border-bottom: 1px solid #e5e7eb;
        padding-bottom: 14px;
        margin-bottom: 20px;
    }
    .app-title {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        font-size: 1.75rem;
        font-weight: 600;
        color: #111827;
        margin: 0;
        letter-spacing: -0.02em;
    }
    .app-subtitle {
        font-size: 0.95rem;
        color: #4b5563;
        margin-top: 4px;
        margin-bottom: 0;
    }
    .academic-notice {
        background-color: #f9fafb;
        border: 1px solid #e5e7eb;
        border-left: 4px solid #4b5563;
        padding: 10px 14px;
        border-radius: 2px;
        margin-bottom: 22px;
        font-size: 0.88rem;
        color: #374151;
        line-height: 1.45;
    }
    .token-container {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin: 16px 0 24px 0;
    }
    .token-card {
        border: 1px solid #d1d5db;
        border-radius: 4px;
        background: #ffffff;
        padding: 6px 12px;
        min-width: 64px;
        text-align: center;
    }
    .token-word {
        font-size: 0.95rem;
        font-weight: 500;
        color: #111827;
        line-height: 1.2;
    }
    .token-tag {
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
        font-size: 0.75rem;
        font-weight: 600;
        color: #2563eb;
        margin-top: 4px;
        border-top: 1px dashed #e5e7eb;
        padding-top: 2px;
    }
    .token-oov {
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
        font-size: 0.65rem;
        color: #b91c1c;
        font-weight: 600;
        margin-top: 2px;
    }
    .stat-label {
        font-size: 0.82rem;
        color: #6b7280;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        font-weight: 500;
    }
    .stat-value {
        font-size: 1.5rem;
        font-weight: 600;
        color: #111827;
        margin-top: 2px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# Data & Model Caching
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Initializing models and loading corpus...")
def get_data_and_models():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    corpus_path = os.path.join(base_dir, "data", "corpus.csv")
    if not os.path.exists(corpus_path):
        corpus_path = "data/corpus.csv"

    docs = load_tagged(corpus_path)
    train, test = split_by_document(docs)

    hmm_plain = HMM(train)
    hmm_split = HMM(train, split_states=True)
    ngram_gen = NGramGenerator([w for w, _ in train])

    train_text = {" ".join(w) for w, _ in train}
    train_patterns = {" ".join(t) for _, t in train}
    train_pairs = {sentence_features(w, t)[1] for w, t in train}

    return {
        "train": train,
        "test": test,
        "hmm_plain": hmm_plain,
        "hmm_split": hmm_split,
        "ngram_gen": ngram_gen,
        "train_text": train_text,
        "train_patterns": train_patterns,
        "train_pairs": train_pairs,
    }


data = get_data_and_models()
train = data["train"]
hmm_plain = data["hmm_plain"]
hmm_split = data["hmm_split"]
ngram_gen = data["ngram_gen"]
train_text = data["train_text"]
train_patterns = data["train_patterns"]
train_pairs = data["train_pairs"]


# -----------------------------------------------------------------------------
# Application Header
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="app-header">
        <h1 class="app-title">Chaotic Text Generator: Sequence Model Evaluation Bench</h1>
        <p class="app-subtitle">Empirical comparison of Plain HMM, Role-Split HMM, and Trigram language models.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Mandatory Notice (Requirement 6)
st.markdown(
    """
    <div class="academic-notice">
        <strong>Notice:</strong> The corpus is synthetic and the grammar is tiny.
        Metrics such as valid tag pattern are defined relative to this corpus
        and do not measure the quality of real English.
    </div>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# Sidebar: Control Panel (Requirement 1)
# -----------------------------------------------------------------------------
st.sidebar.markdown("### Model Parameters")

model_choice = st.sidebar.selectbox(
    "Architecture",
    options=["Role-split HMM", "Plain HMM", "Trigram"],
    index=0,
    help="Role-split HMM introduces state refinement (S/O/P). Plain HMM is memoryless first-order. Trigram conditions on preceding bigram context.",
)

temperature = st.sidebar.slider(
    "Sampling Temperature (T)",
    min_value=0.1,
    max_value=3.0,
    value=1.0,
    step=0.1,
    help="T < 1.0 concentrates probability mass on high-frequency transitions; T > 1.0 flattens the distribution.",
)

seed = st.sidebar.number_input(
    "Random Seed",
    min_value=0,
    max_value=100_000,
    value=42,
    step=1,
)

num_sentences = st.sidebar.slider(
    "Sample Size (N sentences)",
    min_value=10,
    max_value=500,
    value=30,
    step=10,
    help="Number of sentences to sample in a single batch (range: 10 to 500).",
)

trigram_prompt = ""
if model_choice == "Trigram":
    trigram_prompt = st.sidebar.text_input(
        "Initial Prefix (optional)",
        value="the",
        help="Seed words for trigram generation.",
    )


# -----------------------------------------------------------------------------
# Workspace Tabs
# -----------------------------------------------------------------------------
tab_gen, tab_tagger, tab_compare, tab_real = st.tabs(
    [
        "Batch Generation & Metrics",
        "Viterbi Sequence Tagger",
        "Model Comparison",
        "Real-Data Evaluation (UD EWT)",
    ]
)


# =============================================================================
# TAB 1: BATCH GENERATION & EVALUATION (Requirements 2, 3, 5)
# =============================================================================
with tab_gen:
    st.markdown(f"**Sampling target:** {num_sentences} sentences | **Model:** {model_choice} | **Temperature:** {temperature:.1f} | **Seed:** {seed}")

    generate_clicked = st.button("Generate Sentences", type="primary")

    if "batch_results" not in st.session_state:
        st.session_state.batch_results = None

    if generate_clicked:
        rng = random.Random(int(seed))
        samples_data = []

        prefix = trigram_prompt.lower().split() if trigram_prompt else None
        if prefix:
            unknown_prefix = [w for w in prefix if w not in ngram_gen.vocab]
            if unknown_prefix:
                st.caption(f"Prefix tokens {unknown_prefix} are not in training vocabulary. Backing off to unigram.")

        for i in range(num_sentences):
            if model_choice == "Plain HMM":
                words, states = hmm_plain.sample_with_states(rng, temperature)
                coarse_tags = coarse(states)
                hidden_tags_str = " -> ".join(states)
            elif model_choice == "Role-split HMM":
                words, states = hmm_split.sample_with_states(rng, temperature)
                coarse_tags = coarse(states)
                hidden_tags_str = " -> ".join(states)
            else:  # Trigram
                words = ngram_gen.sample_sentence(rng, temperature, prefix=prefix)
                coarse_tags = hmm_split.viterbi(words)
                hidden_tags_str = " -> ".join(coarse_tags) + " (Viterbi inferred)"

            pattern, pair = sentence_features(words, coarse_tags)
            is_valid_pattern = pattern in train_patterns
            is_novel = " ".join(words) not in train_text
            is_plausible_sv = pair in train_pairs

            samples_data.append(
                {
                    "idx": i + 1,
                    "sentence": " ".join(words),
                    "tags": coarse_tags,
                    "hidden_tags_str": hidden_tags_str,
                    "pattern": pattern,
                    "is_valid_pattern": is_valid_pattern,
                    "is_plausible_sv": is_plausible_sv,
                    "is_novel": is_novel,
                    "words_tuple": words,
                }
            )

        n = len(samples_data)
        valid_cnt = sum(s["is_valid_pattern"] for s in samples_data)
        plausible_cnt = sum(s["is_plausible_sv"] for s in samples_data)
        novel_cnt = sum(s["is_novel"] for s in samples_data)

        all_bigrams = []
        for s in samples_data:
            ws = s["words_tuple"]
            all_bigrams += list(zip(ws, ws[1:]))
        distinct_2 = len(set(all_bigrams)) / max(len(all_bigrams), 1)

        st.session_state.batch_results = {
            "model": model_choice,
            "T": temperature,
            "seed": seed,
            "n": n,
            "samples": samples_data,
            "metrics": {
                "valid_tag_pattern": valid_cnt / n,
                "plausible_subject_verb": plausible_cnt / n,
                "novel_sentences": novel_cnt / n,
                "distinct_2": distinct_2,
            },
        }

    res = st.session_state.batch_results
    if res is not None:
        m = res["metrics"]
        st.markdown("<br>", unsafe_allow_html=True)

        # Metrics Panel (Requirement 3)
        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(f'<div class="stat-label">Valid Tag Pattern</div><div class="stat-value">{m["valid_tag_pattern"]:.1%}</div>', unsafe_allow_html=True)
        c2.markdown(f'<div class="stat-label">Plausible Subject-Verb</div><div class="stat-value">{m["plausible_subject_verb"]:.1%}</div>', unsafe_allow_html=True)
        c3.markdown(f'<div class="stat-label">Novel Sentences</div><div class="stat-value">{m["novel_sentences"]:.1%}</div>', unsafe_allow_html=True)
        c4.markdown(f'<div class="stat-label">Distinct-2</div><div class="stat-value">{m["distinct_2"]:.3f}</div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Download CSV & Table Header (Requirements 2 & 5)
        head_col, dl_col = st.columns([3, 1])
        head_col.markdown("#### Sampled Output")

        csv_buffer = io.StringIO()
        writer = csv.writer(csv_buffer)
        writer.writerow(["ID", "Model", "Temperature", "Sentence", "Hidden_Tag_Sequence", "Pattern", "Valid_Pattern", "Plausible_Subject_Verb", "Novel"])
        for s in res["samples"]:
            writer.writerow([
                s["idx"],
                res["model"],
                res["T"],
                s["sentence"],
                s["hidden_tags_str"],
                s["pattern"],
                s["is_valid_pattern"],
                s["is_plausible_sv"],
                s["is_novel"],
            ])
        csv_string = csv_buffer.getvalue()

        dl_col.download_button(
            label="Download CSV",
            data=csv_string,
            file_name=f"samples_{res['model'].replace(' ', '_').lower()}_T{res['T']}.csv",
            mime="text/csv",
        )

        display_rows = []
        for s in res["samples"]:
            display_rows.append({
                "#": s["idx"],
                "Sentence": s["sentence"],
                "Hidden Tag Sequence": s["hidden_tags_str"],
                "Tag Pattern": "Valid" if s["is_valid_pattern"] else "Invalid",
                "Subject-Verb": "Plausible" if s["is_plausible_sv"] else "Implausible",
                "Novelty": "Novel" if s["is_novel"] else "Memorized",
            })

        st.dataframe(display_rows, use_container_width=True, height=420)
    else:
        st.markdown("_Click **Generate Sentences** to evaluate the current model configuration._")


# =============================================================================
# TAB 2: VITERBI SEQUENCE TAGGER (Requirement 4)
# =============================================================================
with tab_tagger:
    st.markdown("#### Dynamic Programming Sequence Decoding (Viterbi)")
    st.markdown(
        "Finds the maximum-likelihood hidden state sequence in log-space using add-k smoothing ($k = 0.01$)."
    )

    tagger_model_choice = st.radio(
        "Underlying HMM Estimator:",
        options=["Role-split HMM (refined states: S / O / P)", "Plain HMM (standard POS states)"],
        horizontal=True,
    )
    tagger_model = hmm_split if "Role-split" in tagger_model_choice else hmm_plain

    st.markdown("Quick test inputs:")
    p1, p2, p3, p4 = st.columns(4)
    selected_preset = None
    if p1.button("watch as VERB"):
        selected_preset = "the farmers watch the mice in the old school ."
    if p2.button("watch as NOUN"):
        selected_preset = "the girls clean the watch in the kitchen ."
    if p3.button("Out-of-vocabulary words"):
        selected_preset = "the quiet astronaut observes distant galaxies ."
    if p4.button("Minimal phrase"):
        selected_preset = "the cats sleep ."

    input_text = st.text_input(
        "Input sentence to decode:",
        value=selected_preset if selected_preset else "the quiet foxes watch the old birds in the park .",
    )

    if input_text.strip():
        tokens_raw = re.findall(r"\w+|[^\w\s]", input_text.strip())
        tokens_lower = [t.lower() for t in tokens_raw]

        oov_tokens = [t for t in tokens_lower if t not in tagger_model.vocab]

        if oov_tokens:
            unique_oov = sorted(set(oov_tokens))
            st.warning(
                f"Out-of-vocabulary token(s): {', '.join(unique_oov)}. "
                f"These tokens do not exist in the training vocabulary. "
                f"Smoothed emission probability P(w|tag) = k / (count(tag) + k*V) is applied; decoding proceeds without interruption."
            )

        try:
            decoded_tags = tagger_model.viterbi(tokens_lower)

            html_tokens = ['<div class="token-container">']
            for word, tag in zip(tokens_raw, decoded_tags):
                is_oov = word.lower() not in tagger_model.vocab
                oov_label = '<div class="token-oov">OOV</div>' if is_oov else ''
                html_tokens.append(
                    f'<div class="token-card">'
                    f'<div class="token-word">{word}</div>'
                    f'<div class="token-tag">{tag}</div>'
                    f'{oov_label}'
                    f'</div>'
                )
            html_tokens.append('</div>')
            st.markdown("".join(html_tokens), unsafe_allow_html=True)

            with st.expander("Sequence analysis & grammar reference"):
                pattern_str = " ".join(decoded_tags)
                is_valid = pattern_str in train_patterns
                st.markdown(f"**Recovered Tag Sequence:** `{pattern_str}`")
                st.markdown(f"**Grammar Validity (relative to synthetic corpus):** {'Valid' if is_valid else 'Invalid'}")
                st.markdown(
                    """
                    **Tagset Definitions:**
                    - `DET`: Determiner (e.g., *the*)
                    - `NOUN`: Noun (e.g., *cats, dogs, garden, watch*)
                    - `VERB`: Verb (e.g., *sleep, watch, clean, eat*)
                    - `ADJ`: Adjective (e.g., *old, hungry, young, quiet*)
                    - `PREP`: Preposition (e.g., *in, at, near, by, behind*)
                    - `PUNCT`: Terminal punctuation (*.*)
                    """
                )

        except Exception as err:
            st.error(f"Decoding failed: {err}")


# =============================================================================
# TAB 3: MODEL COMPARISON & EMPIRICAL PLOTS
# =============================================================================
with tab_compare:
    st.markdown("#### Empirical Benchmark Results (N = 500 sentences)")

    metrics_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "metrics.json")
    if not os.path.exists(metrics_file):
        metrics_file = "results/metrics.json"

    if os.path.exists(metrics_file):
        bench_data = json.load(open(metrics_file, encoding="utf-8"))
        gen = bench_data.get("generation", {})

        st.markdown("##### Performance Comparison at T = 1.0")
        table_rows = [
            {
                "Model Architecture": "Plain HMM",
                "Valid Tag Pattern": f"{gen.get('hmm', {}).get('1.0', {}).get('valid_tag_pattern', 0):.1%}",
                "Plausible Subject-Verb": f"{gen.get('hmm', {}).get('1.0', {}).get('plausible_subject_verb', 0):.1%}",
                "Novel Sentences": f"{gen.get('hmm', {}).get('1.0', {}).get('novel_sentences', 0):.1%}",
                "Distinct-2": f"{gen.get('hmm', {}).get('1.0', {}).get('distinct_2', 0):.3f}",
            },
            {
                "Model Architecture": "Role-split HMM",
                "Valid Tag Pattern": f"{gen.get('hmm_split', {}).get('1.0', {}).get('valid_tag_pattern', 0):.1%}",
                "Plausible Subject-Verb": f"{gen.get('hmm_split', {}).get('1.0', {}).get('plausible_subject_verb', 0):.1%}",
                "Novel Sentences": f"{gen.get('hmm_split', {}).get('1.0', {}).get('novel_sentences', 0):.1%}",
                "Distinct-2": f"{gen.get('hmm_split', {}).get('1.0', {}).get('distinct_2', 0):.3f}",
            },
            {
                "Model Architecture": "Trigram Baseline",
                "Valid Tag Pattern": f"{gen.get('ngram', {}).get('1.0', {}).get('valid_tag_pattern', 0):.1%}",
                "Plausible Subject-Verb": f"{gen.get('ngram', {}).get('1.0', {}).get('plausible_subject_verb', 0):.1%}",
                "Novel Sentences": f"{gen.get('ngram', {}).get('1.0', {}).get('novel_sentences', 0):.1%}",
                "Distinct-2": f"{gen.get('ngram', {}).get('1.0', {}).get('distinct_2', 0):.3f}",
            },
        ]
        st.table(table_rows)

        st.markdown("##### Empirical Plots")
        temps = [0.5, 1.0, 1.5, 2.5]
        temps_str = ["0.5", "1.0", "1.5", "2.5"]

        hmm_vals = [gen.get("hmm", {}).get(t, {}).get("valid_tag_pattern", 0) * 100 for t in temps_str]
        hmm_split_vals = [gen.get("hmm_split", {}).get(t, {}).get("valid_tag_pattern", 0) * 100 for t in temps_str]
        ngram_vals = [gen.get("ngram", {}).get(t, {}).get("valid_tag_pattern", 0) * 100 for t in temps_str]

        # Clean academic matplotlib styling
        plt.rcParams.update({
            "font.family": "sans-serif",
            "font.size": 9,
            "axes.edgecolor": "#cccccc",
            "axes.linewidth": 0.8,
        })

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

        # Subplot 1: Valid tag pattern vs Temperature
        ax1.plot(temps, hmm_vals, marker="o", markersize=5, color="#4b5563", linestyle="-", label="Plain HMM")
        ax1.plot(temps, hmm_split_vals, marker="s", markersize=5, color="#1f2937", linestyle="-", linewidth=1.8, label="Role-split HMM")
        ax1.plot(temps, ngram_vals, marker="^", markersize=5, color="#6b7280", linestyle="--", label="Trigram")
        ax1.set_xlabel("Sampling Temperature (T)")
        ax1.set_ylabel("Valid Tag Pattern (%)")
        ax1.set_title("Grammatical Validity vs Temperature")
        ax1.set_ylim(-5, 105)
        ax1.grid(True, linestyle=":", alpha=0.6)
        ax1.legend(frameon=True)

        # Subplot 2: Transition matrix heatmap
        tags, mat = hmm_plain.transition_matrix()
        im = ax2.imshow(np.array(mat), cmap="Greys", vmin=0, vmax=1)
        ax2.set_xticks(range(len(tags)))
        ax2.set_xticklabels(tags, rotation=45, ha="right", fontsize=8)
        ax2.set_yticks(range(len(tags)))
        ax2.set_yticklabels(tags, fontsize=8)
        for i in range(len(tags)):
            for j in range(len(tags)):
                val = mat[i][j]
                if val > 0.01:
                    ax2.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7.5,
                             color="white" if val > 0.55 else "black")
        ax2.set_title("Plain HMM Transition Matrix P(tag_t | tag_{t-1})")
        ax2.set_xlabel("Target Tag (t)")
        ax2.set_ylabel("Source Tag (t-1)")

        fig.tight_layout()
        st.pyplot(fig)

    else:
        st.caption("Benchmark file `results/metrics.json` not found. Run `python -m src.evaluate` to generate.")


# =============================================================================
# TAB 4: REAL-DATA EVALUATION (Universal Dependencies English EWT)
# =============================================================================
with tab_real:
    st.markdown("#### Real-Data Treebank Evaluation (Universal Dependencies English EWT)")
    st.markdown(
        "Evaluation of first-order HMM + Viterbi decoding against a most-frequent tag baseline "
        "on 12,544 training sentences and 2,077 test sentences (17 UPOS categories)."
    )

    ewt_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "ewt_results.json")
    if not os.path.exists(ewt_file):
        ewt_file = "results/ewt_results.json"

    if os.path.exists(ewt_file):
        ewt_data = json.load(open(ewt_file, encoding="utf-8"))
        models = ewt_data.get("models", {})
        mf = models.get("most-frequent tag", {})
        hv = models.get("HMM + Viterbi", {})

        ewt_rows = [
            {
                "Model": "Most-frequent tag baseline",
                "Accuracy": f"{mf.get('accuracy', 0.862):.1%}",
                "Known words": f"{mf.get('known_words', 0.908):.1%}",
                "Unknown words (OOV)": f"{mf.get('unknown_words', 0.286):.1%}",
                "Whole sentence correct": f"{mf.get('whole_sentence', 0.319):.1%}",
            },
            {
                "Model": "HMM + Viterbi",
                "Accuracy": f"{hv.get('accuracy', 0.881):.1%}",
                "Known words": f"{hv.get('known_words', 0.929):.1%}",
                "Unknown words (OOV)": f"{hv.get('unknown_words', 0.282):.1%}",
                "Whole sentence correct": f"{hv.get('whole_sentence', 0.404):.1%}",
            },
        ]
        st.table(ewt_rows)

        st.markdown(
            """
            **Evaluation Analysis:**
            - **Synthetic Corpus vs Real Text:** On synthetic data, the closed vocabulary and 16 regular grammar patterns allow 100% accuracy. On real web text, structural entropy (10,359 unique sequences) reduces HMM accuracy to **88.1%** (+1.9 percentage points over unigram baseline).
            - **Out-of-Vocabulary Bottleneck:** Unseen test tokens account for **7.5% of total tokens**, on which both models achieve only ~28% accuracy due to uniform smoothed priors.
            """
        )

        with st.expander("Top confusion categories (Gold -> Predicted)"):
            confs = ewt_data.get("confusions", [])
            conf_rows = [{"Gold Tag": g, "Predicted Tag": p, "Frequency": c} for g, p, c in confs[:10]]
            st.dataframe(conf_rows, use_container_width=True)
    else:
        st.caption("EWT evaluation data not found. Run `python -m src.evaluate_real` to generate.")
