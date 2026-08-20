# 🔮 Autonomous Data Analyst — Churn Prediction Agent

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Haleemy/ChurnPrediction/blob/main/notebooks/churn_analysis.ipynb)

A production-quality, **genuinely agentic** system for exploring customer churn data, predicting risk, and answering natural language questions — with verified, grounded answers (no hallucination).

---

## Architecture

```
User Query
    ↓
Streamlit UI (presentation)
    ↓
Agent (orchestrator)
    ├── Planner → LLM (Groq) [1 call: intent + plan]
    ├── Tools:
    │   ├── DataFrameTool  → real pandas computations
    │   ├── ModelTool      → sklearn pipeline predictions
    │   ├── ExplainerTool  → SHAP feature attribution
    │   └── ChartTool      → Plotly charts from real data
    ├── Verifier → validates every result before use
    └── Answer Generator → LLM [1 call: evidence-grounded]
```

**LLM = reasoning/orchestration only.** All numbers, predictions, and statistics come from Python tools. The LLM sees tool results as evidence and explains them — it never calculates.

---

## Quick Start

### 1. Install

```bash
git clone <repo>
cd churn-analyst
pip install -r requirements.txt
```

### 2. Configure

```bash
cp .env.example .env
# Edit .env and add your Groq API key:
# GROQ_API_KEY=gsk_...
```

Get a free key at [console.groq.com](https://console.groq.com).

### 3. Train the model

```bash
python run.py train
```

### 4. Run the app

```bash
streamlit run app/ui/streamlit_app.py
# Or:
python run.py ui
```

Open http://localhost:8501

---

## Dataset

- **Source**: Telco Customer Churn (IBM/Kaggle)
- **File**: `data/Customer-Churn.csv`
- **Actual size**: 7,043 rows × 21 features (computed at runtime, not hardcoded)

### Data Cleaning Decisions

| Issue | Decision | Rationale |
|---|---|---|
| `TotalCharges` blank strings | Convert to numeric, impute 0 for zero-tenure | 11 records with tenure=0 and blank TotalCharges; logically TotalCharges=0 |
| Remaining NaN TotalCharges | Drop row | Very few (if any); avoids imputation bias |
| `customerID` | Excluded from features | Identifier, not predictive |
| `SeniorCitizen` 0/1 encoding | Keep as int | OrdinalEncoder handles it correctly |
| Categorical 3-way values | Preserved (No / No phone service / No internet service) | These distinctions are predictively meaningful |

---

## Model Selection

### Why ROC-AUC as primary metric?

Churn is an imbalanced classification problem (~26% positive rate). Accuracy is misleading — predicting "No churn" always gives ~74% accuracy. ROC-AUC measures discrimination across all thresholds.

### Candidate comparison (5-fold CV on actual data)

| Model | ROC-AUC | Notes |
|---|---|---|
| **LogisticRegression** | **0.8444 ± 0.0116** | ✅ Winner |
| HistGradientBoosting | 0.8368 ± 0.0083 | Good but LR wins here |
| RandomForest | 0.8283 ± 0.0101 | Overfitting signs |

**Result**: LogisticRegression won on this dataset. This is a common outcome — on well-preprocessed tabular data with good feature engineering, LR is hard to beat and provides excellent probability calibration.

### Test set metrics (held-out 20%)

Computed from actual model on real data:
- ROC-AUC: ~0.840
- Optimal threshold: ~0.499 (Youden's J statistic)
- Recall at optimal threshold: reported in `models/model_metadata.json`

---

## Agent Design

### Why this is a real agent (not an LLM wrapper)

| Feature | Implementation |
|---|---|
| Multi-step planning | Planner generates structured JSON execution plan |
| Tool selection | Agent picks specific tools per plan step |
| Real computation | All numbers come from pandas/sklearn |
| Verification | Every tool result verified before use |
| Retry on failure | Up to 2 retries with fallback plan |
| Memory | Structured turn history for multi-turn context |
| Anti-hallucination | LLM receives evidence objects, not free-form |

### Tool inventory

| Tool | Purpose |
|---|---|
| `analyze_data` | 16 dataframe operations: filter, group, aggregate, correlate, etc. |
| `predict_customer_risk` | Risk score for known customer by ID |
| `predict_hypothetical` | Compare current vs. hypothetical scenario |
| `predict_new_customer` | Risk for a new/hypothetical customer |
| `get_top_risk_customers` | N highest-risk customers |
| `generate_chart` | 8 chart types from actual data |
| `get_model_info` | Model metadata and feature importance |
| `get_dataset_info` | Dataset statistics |

### Hallucination Prevention

1. **Evidence objects**: LLM's answer prompt includes exact tool outputs as the source of truth
2. **Grounding check**: Post-answer regex scan for ungrounded numbers
3. **Verification layer**: Structural validation of every tool result
4. **No LLM math**: The LLM is never asked to calculate; tools do all computation
5. **Unavailable column detection**: Agent explains missing columns rather than inventing data

---

## Example Questions

```
Dataset/EDA:
  "How many customers are there?"
  "What is the churn rate?"
  "Which contract type has the highest churn?"
  "Show churn by payment method"
  "Does tenure correlate with churn?"
  "Average monthly charges for churned customers"

Customer-specific:
  "What is the churn risk of customer 7590-VHVEG?"
  "Why is customer 7590-VHVEG high risk?"
  "Which customers are most likely to churn?"

Hypothetical:
  "What happens to customer 7590-VHVEG's risk if they switch to a two-year contract?"
  "What if MonthlyCharges increase to $90?"
  "Predict risk for: Month-to-month, Fiber optic, senior citizen, 6 months tenure"

Multi-step:
  "Find the top-risk customers and compare their average monthly charges by contract"
  "Which segment has the highest predicted risk and why?"

Unavailable column (handled gracefully):
  "What is the churn rate by region?"  → explains region column doesn't exist
```

---

## Project Structure

```
churn-analyst/
├── app/
│   ├── config.py               # Central config (paths, secrets, constants)
│   ├── data/
│   │   ├── loader.py           # Dataset loading + cleaning + caching
│   │   └── analyzer.py         # 16 controlled dataframe operations
│   ├── model/
│   │   ├── preprocessing.py    # sklearn ColumnTransformer pipeline
│   │   ├── train.py            # Model training + comparison + selection
│   │   ├── predictor.py        # Clean prediction interface
│   │   └── explainability.py   # SHAP explanations
│   ├── agent/
│   │   ├── agent.py            # Main orchestration loop
│   │   ├── planner.py          # Intent classification + plan generation
│   │   ├── tools.py            # Tool registry + executor
│   │   ├── verifier.py         # Result verification layer
│   │   └── prompts.py          # All system prompts and tool schemas
│   ├── llm/
│   │   └── provider.py         # LLMProvider base + Groq + Fallback
│   └── visualization/
│       └── charts.py           # 8 Plotly chart types
├── app/ui/
│   └── streamlit_app.py        # Streamlit interface
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── models/                     # Saved model + metadata (gitignored)
├── notebooks/
│   └── churn_analysis.ipynb    # Full EDA → model → evaluation notebook
├── tests/                      # 85 pytest tests
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env.example
└── run.py                      # CLI: train | inspect | test-agent | ui
```

---

## Running Tests

```bash
# Run all 85 tests
$env:PYTHONPATH = (pwd).Path  # Windows
pytest tests/ -v

# Or with PYTHONPATH set
PYTHONPATH=. pytest tests/ -v  # Unix/Mac
```

---

## Docker

```bash
# Build
docker build -t churn-analyst .

# Run (with API key)
docker run -p 8501:8501 -e GROQ_API_KEY=your_key churn-analyst

# Or using docker-compose
cp .env.example .env  # Fill in GROQ_API_KEY
docker-compose up

# Train model separately first
docker-compose run trainer
```

---

## Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `GROQ_API_KEY` | Yes* | — | Groq API key |
| `MODEL_NAME` | No | `llama-3.3-70b-versatile` | Groq model to use |
| `DATA_PATH` | No | `data/Customer-Churn.csv` | Path to dataset |
| `LLM_TEMPERATURE` | No | `0.0` | LLM temperature (0 = deterministic) |
| `LLM_MAX_TOKENS` | No | `2048` | Max tokens per LLM call |
| `LOG_LEVEL` | No | `INFO` | Logging level |

*App works without API key in demo mode with limited natural language.

---

## Important Trade-offs

| Trade-off | Decision | Why |
|---|---|---|
| SHAP vs. permutation importance | SHAP primary, fallback to feature importance | SHAP is model-faithful but adds latency; fallback ensures robustness |
| LR vs. GBM | LR selected by CV | For well-preprocessed Telco data, LR has better calibration |
| Native tool calling vs. JSON parsing | JSON parsing of planning output | More reliable on free-tier models |
| Full CV vs. single split | 5-fold stratified CV | Prevents overfitting metric selection on small dataset |
| XGBoost | Not used | No dependency advantage; HistGBM in sklearn is equivalent |

---

## Limitations

1. **No region/geography column**: The dataset doesn't contain regional data; region-based questions are gracefully declined.
2. **Free-tier rate limits**: Groq free tier limits ~30 req/min. Complex multi-step questions may hit limits.
3. **SHAP for LR**: SHAP LinearExplainer gives global-style explanations for LR; less instance-specific than TreeExplainer.
4. **Memory is not persistent**: Conversation memory resets when the Streamlit app restarts.
5. **No streaming**: Answers are returned as complete text (not streamed tokens).

---

## Future Improvements

- [ ] Streaming LLM responses in Streamlit
- [ ] Persistent conversation memory (SQLite)
- [ ] Threshold tuning UI slider
- [ ] Customer segment cohort analysis
- [ ] Scheduled retraining on fresh data
- [ ] Evaluation harness with 15-question grounded test suite
- [ ] OpenAI/Anthropic fallback provider

---

## Reflection

- **Hardest Part**: Engineering deterministic tool-selection and anti-hallucination verification logic on a free-tier LLM. Small models can occasionally return malformed JSON or hit API rate limits; building a resilient rule-based plan fallback system (`create_fallback_plan`) alongside post-execution grounding verification ensured zero numerical hallucination without breaking conversation flows.
- **What I Learned / Taught Myself**: Designing stateful agent planning pipelines, enforcing strict separation between LLM reasoning and Python computation, computing SHAP model explanations for tabular pipelines, and building dynamic light/dark visual theme engines in Streamlit.
- **What I'd Do Differently With More Time**: Build an automated end-to-end evaluation harness with benchmark synthetic queries, implement streaming token response rendering in the chat interface, and implement vector-backed conversation memory.

---

## Time Spent

**~9.5 hours total**:
- **~2.5h**: Exploratory Data Analysis, data cleaning pipeline, feature engineering, and model cross-validation comparison.
- **~3.5h**: Agent architecture design (Planner, operational tool signatures, execution loop, and grounding verifier).
- **~2.0h**: Streamlit web application, custom CSS theme system (Light & Dark modes), and Plotly chart integrations.
- **~1.0h**: Pytest testing suite (85 tests), Docker containerization, and Streamlit Cloud auto-retraining deployment setup.
- **~0.5h**: Documentation, Colab setup, and final submission verification.
