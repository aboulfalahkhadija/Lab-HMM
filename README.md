# Lab 03: Chaotic Text Generator — Test Bench

An interactive test bench web application comparing generative sequence models:
- **Plain HMM** (first-order Hidden Markov Model over part-of-speech tags)
- **Role-Split HMM** (syntactic role refinement: Subject, Object, Prepositional phrase)
- **Trigram Baseline** (word-level n-gram with back-off)
- **Viterbi Tagger** (dynamic programming decoding with add-k smoothing, resilient to out-of-vocabulary words)

Detailed guide: `Lab03_Chaotic_Text_Generator_Guide.pdf`.

---

## Features

1. **Batch Generation & Quality Evaluation**:
   - Model selector (Plain HMM, Role-split HMM, Trigram), temperature slider ($0.1 \le T \le 3.0$), random seed, and sample size ($10 \le N \le 500$).
   - Real-time batch evaluation table: sentence text, hidden POS tag sequence, and validation badges (`Valid Tag Pattern`, `Plausible Subject-Verb`, `Novelty`).
   - Summary quality metrics cards: *Valid Tag Pattern*, *Plausible Subject-Verb*, *Novelty*, *Distinct-2*.
   - **Download CSV** button exporting the full evaluation batch.
2. **Interactive Viterbi POS Tagger**:
   - Type any natural language sentence to inspect its decoded tag sequence.
   - Out-of-vocabulary (OOV) word detector: warns the user without crashing the app.
   - Disambiguation test for polysemic words (e.g. *watch* as NOUN vs VERB).
3. **Model Comparison & Visualization**:
   - Side-by-side benchmark comparison at $T = 1.0$.
   - Interactive line plot of *Valid Tag Pattern vs Temperature*.
   - Transition probability matrix heatmap $P(\text{tag}_t \mid \text{tag}_{t-1})$.
4. **Real-Text Benchmark (Universal Dependencies English EWT)**:
   - Results on 12,544 training and 2,077 test sentences (17 UPOS tags).
   - Analysis of known vs unknown words and top confusion pairs.

---

## File Structure

| Path | Purpose |
|---|---|
| `app.py` | Complete Streamlit test bench web application |
| `src/pipeline.py` | HMM, Viterbi decoding, role splitting, trigram model, and metric functions |
| `src/evaluate.py` | Evaluation script on the synthetic corpus (generates `results/metrics.json`) |
| `src/evaluate_real.py` | Evaluation on real text (UD English EWT, generates `results/ewt_results.json`) |
| `data/corpus.csv` | Synthetic tagged corpus (1,200 sentences, 6 tags) |
| `results/` | Benchmark metrics and error analysis logs |

---

## Quickstart

### 1. Installation
```bash
pip install -r requirements.txt
```

### 2. Run Evaluations (Optional)
```bash
python -m src.evaluate
python -m src.evaluate_real
```

### 3. Launch Web Application
```bash
streamlit run app.py
```

---

## Deployment Note

When deploying to **Streamlit Community Cloud**:
- The repository must include `requirements.txt` and `data/corpus.csv`.
- The `data/ewt/` directory (real text dataset, ~17 MB) is ignored via `.gitignore` and downloaded automatically if needed.
