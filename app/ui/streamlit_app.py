"""
streamlit_app.py — Streamlit UI for the Autonomous Data Analyst.

Panels:
- Left sidebar: dataset + model info
- Main area: chat interface with history
- Right: tables, charts, predictions

Never exposes raw tracebacks to users.
"""
import json
import logging
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Autonomous Churn Analyst",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.main-header {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    padding: 2rem;
    border-radius: 12px;
    margin-bottom: 1.5rem;
    color: white;
}

.main-header h1 { 
    font-size: 2rem; 
    font-weight: 700; 
    margin: 0; 
    color: white;
}

.main-header p { 
    color: #a0aec0; 
    margin: 0.5rem 0 0 0; 
    font-size: 0.95rem;
}

.metric-card {
    background: linear-gradient(135deg, #1e293b, #0f172a);
    border: 1px solid #334155;
    border-radius: 10px;
    padding: 1rem;
    text-align: center;
    color: white;
}

.metric-card .metric-value {
    font-size: 1.8rem;
    font-weight: 700;
    color: #60a5fa;
}

.metric-card .metric-label {
    font-size: 0.8rem;
    color: #94a3b8;
    margin-top: 0.25rem;
}

.risk-high { color: #ef4444; font-weight: 600; }
.risk-medium { color: #f59e0b; font-weight: 600; }
.risk-low { color: #22c55e; font-weight: 600; }

.chat-message-user {
    background: #1e40af;
    color: white;
    border-radius: 12px 12px 2px 12px;
    padding: 0.75rem 1rem;
    margin: 0.5rem 0;
    max-width: 85%;
    margin-left: auto;
}

.chat-message-assistant {
    background: #1e293b;
    color: #e2e8f0;
    border-radius: 12px 12px 12px 2px;
    padding: 0.75rem 1rem;
    margin: 0.5rem 0;
    max-width: 90%;
    border-left: 3px solid #3b82f6;
}

.stButton > button {
    background: linear-gradient(135deg, #2563eb, #1d4ed8);
    color: white;
    border: none;
    border-radius: 8px;
    padding: 0.5rem 1.5rem;
    font-weight: 500;
    transition: all 0.2s;
}

.stButton > button:hover {
    background: linear-gradient(135deg, #1d4ed8, #1e40af);
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(37, 99, 235, 0.4);
}

.warning-box {
    background: #1c1917;
    border: 1px solid #92400e;
    border-radius: 8px;
    padding: 0.75rem;
    color: #fbbf24;
    font-size: 0.85rem;
}

.tool-badge {
    display: inline-block;
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 4px;
    padding: 0.2rem 0.6rem;
    font-size: 0.75rem;
    color: #94a3b8;
    margin: 0.1rem;
}
</style>
""", unsafe_allow_html=True)


# ── Session state initialization ───────────────────────────────────────────────

def init_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "agent" not in st.session_state:
        st.session_state.agent = None
    if "dataset_info" not in st.session_state:
        st.session_state.dataset_info = None
    if "model_info" not in st.session_state:
        st.session_state.model_info = None
    if "initialization_error" not in st.session_state:
        st.session_state.initialization_error = None
    if "charts_to_show" not in st.session_state:
        st.session_state.charts_to_show = []
    if "last_tool_results" not in st.session_state:
        st.session_state.last_tool_results = []


# ── Agent initialization ───────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def get_agent():
    """Initialize and cache the agent."""
    from app.llm.provider import create_provider
    from app.agent.agent import ChurnAnalystAgent
    llm = create_provider()
    return ChurnAnalystAgent(llm)


@st.cache_data(show_spinner=False)
def get_dataset_info_cached():
    from app.data.loader import load_dataset, get_dataset_info
    df = load_dataset()
    return get_dataset_info(df)


@st.cache_data(show_spinner=False)
def get_model_info_cached():
    from app.agent.tools import tool_get_model_info
    result = tool_get_model_info()
    if result.get("success"):
        return result["data"]
    return None


# ── Sidebar ────────────────────────────────────────────────────────────────────

def render_sidebar():
    with st.sidebar:
        st.markdown("## 🔮 Churn Analyst")
        st.markdown("---")

        # Dataset info
        st.markdown("### 📊 Dataset")
        try:
            info = get_dataset_info_cached()
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Customers", f"{info['n_rows']:,}")
            with col2:
                st.metric("Churn Rate", f"{info['churn_rate_pct']}%")
            st.metric("Features", f"{info['n_cols'] - 2}")  # exclude ID + target
            if info.get("n_missing_total", 0) == 0:
                st.success("✓ No missing values")
            else:
                st.warning(f"⚠ {info['n_missing_total']} missing values")
        except Exception as e:
            st.error("Dataset not loaded")
            logger.error(f"Dataset info error: {e}")

        st.markdown("---")

        # Model info
        st.markdown("### 🤖 Model")
        try:
            model_info = get_model_info_cached()
            if model_info:
                st.markdown(f"**Type:** `{model_info.get('model_name', 'Unknown')}`")
                metrics = model_info.get("test_metrics", {})
                st.metric("ROC-AUC", f"{metrics.get('roc_auc', 0):.3f}")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("F1", f"{metrics.get('f1_at_0.5', 0):.3f}")
                with col2:
                    st.metric("Recall", f"{metrics.get('recall_at_0.5', 0):.3f}")
                st.metric("Threshold", f"{metrics.get('optimal_threshold', 0.5):.3f}")
            else:
                st.warning("⚠ Model not trained. Run: `python run.py train`")
        except Exception as e:
            st.warning("Model not available")
            logger.error(f"Model info error: {e}")

        st.markdown("---")

        # Example queries
        st.markdown("### 💡 Example Queries")
        examples = [
            "What is the overall churn rate?",
            "Which contract type has the highest churn?",
            "Show churn by payment method",
            "What is the risk for customer 7590-VHVEG?",
            "Top 10 highest risk customers",
            "What if customer changes to 2-year contract?",
            "Average monthly charges for churned customers",
            "Does tenure correlate with churn?",
        ]
        for example in examples:
            if st.button(example, key=f"ex_{hash(example)}", use_container_width=True):
                st.session_state["inject_query"] = example

        st.markdown("---")

        # Controls
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.charts_to_show = []
            st.session_state.last_tool_results = []
            agent = st.session_state.get("agent")
            if agent:
                agent.clear_memory()
            st.rerun()

        # API status
        st.markdown("---")
        st.markdown("### ⚙️ Status")
        from app.config import GROQ_API_KEY, MODEL_NAME
        if GROQ_API_KEY:
            st.success(f"✓ Groq API connected")
            st.caption(f"Model: `{MODEL_NAME}`")
        else:
            st.warning("⚠ No API key — limited mode")
            st.caption("Add GROQ_API_KEY to .env")


# ── Chat rendering ─────────────────────────────────────────────────────────────

def render_message(msg: Dict):
    role = msg["role"]
    content = msg["content"]

    if role == "user":
        st.markdown(
            f'<div class="chat-message-user">{content}</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="chat-message-assistant">{content}</div>',
            unsafe_allow_html=True,
        )

        # Show tool results if available
        if msg.get("tool_results"):
            with st.expander("🔧 Tool Results", expanded=False):
                for tr in msg["tool_results"]:
                    tool_name = tr.get("tool", "unknown")
                    result = tr.get("result", {})
                    if result.get("success"):
                        st.markdown(f"**`{tool_name}`** ✓")
                        data = result.get("data")
                        if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
                            st.dataframe(pd.DataFrame(data).head(20), use_container_width=True)
                        elif isinstance(data, dict):
                            st.json(data)
                        else:
                            st.write(data)
                    else:
                        st.markdown(f"**`{tool_name}`** ❌ {result.get('error', 'Failed')}")

        # Show warnings
        if msg.get("warnings"):
            for w in msg["warnings"]:
                st.markdown(f'<div class="warning-box">⚠️ {w}</div>', unsafe_allow_html=True)

        # Show charts
        if msg.get("charts"):
            for chart_data in msg["charts"]:
                try:
                    fig = go.Figure(chart_data)
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    logger.warning(f"Could not render chart: {e}")


# ── Prediction widget ──────────────────────────────────────────────────────────

def render_prediction_widget():
    """Inline customer prediction widget."""
    with st.expander("🎯 Quick Customer Lookup", expanded=False):
        customer_id = st.text_input(
            "Customer ID",
            placeholder="e.g. 7590-VHVEG",
            key="quick_lookup_id",
        )
        if st.button("Predict Risk", key="predict_btn"):
            if customer_id.strip():
                with st.spinner("Predicting..."):
                    try:
                        from app.model.predictor import predict_churn_risk
                        result = predict_churn_risk(customer_id.strip())
                        if "error" in result:
                            st.error(f"❌ {result['error']}")
                        else:
                            risk_level = result["risk_level"]
                            risk_class = f"risk-{risk_level.lower()}"
                            score = result["risk_score"]

                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Risk Score", f"{score:.3f}")
                            with col2:
                                st.markdown(f"**Risk Level:** <span class='{risk_class}'>{risk_level}</span>", unsafe_allow_html=True)
                            with col3:
                                st.markdown(f"**Prediction:** {result['prediction']}")

                            if result.get("top_factors"):
                                st.markdown("**Top Factors:**")
                                for f in result["top_factors"][:5]:
                                    direction = "⬆️" if f.get("direction") == "increases_risk" else "⬇️"
                                    val = f.get("raw_value", "")
                                    shap = f.get("shap_value") or f.get("importance", "")
                                    st.markdown(f"  {direction} **{f['feature']}** = `{val}` (contribution: {shap:.4f})")
                    except FileNotFoundError:
                        st.error("Model not trained. Run `python run.py train` first.")
                    except Exception as e:
                        logger.error(f"Prediction widget error: {e}")
                        st.error("Prediction failed. Check logs for details.")


# ── Main app ───────────────────────────────────────────────────────────────────

def main():
    init_session_state()

    # ── Header ────────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="main-header">
        <h1>🔮 Autonomous Data Analyst</h1>
        <p>Ask questions about customer churn — powered by a real agentic ML system with grounded, verified answers.</p>
    </div>
    """, unsafe_allow_html=True)

    render_sidebar()

    # ── Quick prediction widget ───────────────────────────────────────────────
    render_prediction_widget()

    # ── Chat history ──────────────────────────────────────────────────────────
    st.markdown("### 💬 Chat")

    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.messages:
            render_message(msg)

    # ── Input area ────────────────────────────────────────────────────────────
    # Check for injected query (from sidebar example buttons)
    injected = st.session_state.pop("inject_query", None)

    col1, col2 = st.columns([9, 1])
    with col1:
        user_input = st.chat_input(
            "Ask anything about churn... e.g. 'Which contract type has highest churn?'",
            key="chat_input",
        )
    with col2:
        pass  # Placeholder for layout

    query = injected or user_input

    if query:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": query})

        # Initialize agent if needed
        if st.session_state.agent is None:
            try:
                st.session_state.agent = get_agent()
            except Exception as e:
                logger.error(f"Agent init failed: {e}")
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": "⚠️ Could not initialize the agent. Check that the model is trained (`python run.py train`) and dependencies are installed.",
                    "tool_results": [],
                    "warnings": [],
                    "charts": [],
                })
                st.rerun()

        # Process query
        with st.spinner("🤔 Analyzing..."):
            try:
                agent = st.session_state.agent
                result = agent.answer(query)
                
                # Store assistant response with metadata
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result["answer"],
                    "tool_results": result.get("tool_results", []),
                    "warnings": result.get("warnings", []),
                    "charts": result.get("charts", []),
                    "plan": result.get("plan", {}),
                })
                st.session_state.last_tool_results = result.get("tool_results", [])

            except ValueError as e:
                # API key or config issue
                error_msg = str(e)
                if "api" in error_msg.lower() or "key" in error_msg.lower():
                    user_msg = f"🔑 **API Configuration Error:** {error_msg}\n\nPlease add your `GROQ_API_KEY` to the `.env` file."
                else:
                    user_msg = f"⚠️ **Configuration error:** {error_msg}"
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": user_msg,
                    "tool_results": [],
                    "warnings": [],
                    "charts": [],
                })

            except RuntimeError as e:
                error_msg = str(e)
                if "rate limit" in error_msg.lower():
                    user_msg = "⏳ **Rate limit reached.** Please wait a minute and try again. (Groq free tier has a request limit.)"
                else:
                    user_msg = f"⚠️ **Processing error.** The system encountered a temporary issue. Please try rephrasing your question."
                logger.error(f"Agent runtime error: {e}")
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": user_msg,
                    "tool_results": [],
                    "warnings": [],
                    "charts": [],
                })

            except FileNotFoundError as e:
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": "⚠️ **Model not trained.** Please run `python run.py train` first, then refresh this page.",
                    "tool_results": [],
                    "warnings": [],
                    "charts": [],
                })

            except Exception as e:
                logger.error(f"Unexpected agent error: {traceback.format_exc()}")
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": "⚠️ An unexpected error occurred. Please try a different question or check the logs.",
                    "tool_results": [],
                    "warnings": [],
                    "charts": [],
                })

        st.rerun()


if __name__ == "__main__":
    main()
