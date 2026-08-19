"""
streamlit_app.py — "Churn Analyst Agent

Design language: near-black canvas, glowing emerald / amber / magenta accents,
sharp-edged cards with colored left rails, sidebar page navigation (not tabs),
monospace numerics for data-heavy panels. Built as a fresh structure rather
than a re-skin — page routing, layout, and component design all differ from
prior versions.

Pages (sidebar-routed):
  1.  Overview      — portfolio KPIs + at-a-glance charts
  2.  Ask the Agent  — natural-language chat, grounded in real tool calls
  3.  Customer Lens  — single customer inspection + what-if simulator
  4.  Risk Ledger    — sortable/filterable full-portfolio risk table + export
  5.  System         — model card, agent architecture, environment status


"""
import logging
import sys
import traceback
from pathlib import Path
from typing import Dict

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="Churn Analyst",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════════════════════
# DESIGN SYSTEM — Dark & Light Themes
# ═══════════════════════════════════════════════════════════════════════════
def render_theme_css(theme: str = "Dark"):
    if theme == "Light":
        MINT = "#059669"
        AMBER = "#D97706"
        MAGENTA = "#E11D48"
        INK = "#F8FAFC"
        PANEL = "#FFFFFF"
        TEXT = "#0F172A"
        MUTED = "#64748B"
        BORDER = "#E2E8F0"

        st.markdown(f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Sora:wght@500;600;700;800&family=JetBrains+Mono:wght@400;500;600&family=Inter:wght@400;500;600&display=swap');

        html, body, [class*="css"] {{
            font-family: 'Inter', sans-serif;
            color: {TEXT};
        }}
        h1, h2, h3, h4, h5, h6 {{
            font-family: 'Sora', sans-serif;
            letter-spacing: -0.015em;
            color: {TEXT} !important;
        }}
        code, .mono {{ font-family: 'JetBrains Mono', monospace; }}

        .stApp {{
            background:
                radial-gradient(circle at 12% 8%, rgba(5,150,105,0.06) 0%, transparent 40%),
                radial-gradient(circle at 88% 15%, rgba(225,29,72,0.05) 0%, transparent 40%),
                radial-gradient(circle at 50% 95%, rgba(217,119,6,0.05) 0%, transparent 45%),
                {INK};
        }}

        .block-container {{ padding-top: 3.8rem; padding-bottom: 3rem; max-width: 1240px; }}

        .topstrip {{
            display: flex; justify-content: space-between; align-items: center;
            padding: 0.55rem 1.1rem;
            background: {PANEL};
            border: 1px solid {BORDER};
            border-radius: 10px;
            margin-bottom: 1.4rem;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.74rem;
            color: {MUTED};
            box-shadow: 0 1px 3px rgba(0,0,0,0.03);
        }}
        .topstrip .dot {{ color: {MINT}; }}

        .page-eyebrow {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.72rem;
            letter-spacing: 0.14em;
            text-transform: uppercase;
            color: {MINT};
            margin-bottom: 0.3rem;
        }}
        .page-title {{
            font-size: 2.05rem;
            font-weight: 800;
            margin: 0 0 0.35rem 0;
            color: {TEXT};
        }}
        .page-sub {{ color: {MUTED}; font-size: 0.95rem; margin-bottom: 1.6rem; }}

        .rail-card {{
            background: {PANEL};
            border: 1px solid {BORDER};
            border-left: 3px solid var(--rail-color, {MINT});
            border-radius: 4px 12px 12px 4px;
            padding: 1.1rem 1.3rem;
            height: 100%;
            box-shadow: 0 2px 8px rgba(0,0,0,0.04);
        }}
        .rail-label {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.7rem;
            text-transform: uppercase;
            letter-spacing: 0.09em;
            color: {MUTED};
            margin-bottom: 0.3rem;
        }}
        .rail-value {{
            font-family: 'Sora', sans-serif;
            font-size: 1.85rem;
            font-weight: 700;
            color: {TEXT};
            line-height: 1.05;
        }}
        .rail-foot {{ font-size: 0.78rem; color: {MUTED}; margin-top: 0.35rem; }}

        .pill {{
            display: inline-block; padding: 3px 12px; border-radius: 999px;
            font-size: 0.76rem; font-weight: 700; font-family: 'JetBrains Mono', monospace;
            letter-spacing: 0.03em;
        }}
        .pill-high {{ background: #fee2e2; color: #dc2626; border: 1px solid #fca5a5; }}
        .pill-medium {{ background: #fef3c7; color: #d97706; border: 1px solid #fde68a; }}
        .pill-low {{ background: #d1fae5; color: #059669; border: 1px solid #6ee7b7; }}

        .msg-row {{ display: flex; margin: 0.5rem 0; }}
        .msg-row.user {{ justify-content: flex-end; }}
        .bubble-user {{
            background: #eff6ff;
            border: 1px solid #bfdbfe;
            color: #1e3a8a;
            padding: 0.75rem 1.1rem;
            border-radius: 14px 14px 2px 14px;
            max-width: 78%;
            font-size: 0.93rem;
            box-shadow: 0 1px 4px rgba(0,0,0,0.03);
        }}
        .bubble-agent {{
            background: {PANEL};
            border: 1px solid {BORDER};
            border-left: 3px solid {MAGENTA};
            color: {TEXT};
            padding: 0.85rem 1.15rem;
            border-radius: 2px 14px 14px 14px;
            max-width: 88%;
            font-size: 0.93rem;
            box-shadow: 0 2px 6px rgba(0,0,0,0.04);
        }}
        .agent-tag {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.68rem;
            color: {MAGENTA};
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin-bottom: 0.3rem;
            display: block;
        }}

        .stButton > button {{
            background: {PANEL};
            border: 1px solid {BORDER};
            color: {TEXT};
            border-radius: 9px;
            font-size: 0.85rem;
            font-weight: 500;
            transition: all 0.15s ease;
            box-shadow: 0 1px 3px rgba(0,0,0,0.02);
        }}
        .stButton > button:hover {{
            border-color: {MINT};
            color: {MINT};
            background: #f1f5f9;
            box-shadow: 0 2px 8px rgba(5,150,105,0.12);
        }}

        .stChatInputContainer, div[data-testid="stChatInput"] {{
            border-color: #cbd5e1 !important;
            background-color: #ffffff !important;
        }}

        section[data-testid="stSidebar"] {{
            background: #ffffff;
            border-right: 1px solid {BORDER};
        }}
        section[data-testid="stSidebar"] .stRadio label {{
            font-size: 0.92rem;
            color: {TEXT};
        }}
        .sidebar-brand {{
            display: flex; align-items: center; gap: 0.6rem;
            padding: 0.4rem 0 1rem 0;
        }}
        .sidebar-brand .mark {{
            width: 34px; height: 34px; border-radius: 9px;
            background: linear-gradient(135deg, {MINT}, {MAGENTA});
            display: flex; align-items: center; justify-content: center;
            font-family: 'Sora', sans-serif; font-weight: 800; color: #ffffff; font-size: 1.05rem;
        }}
        .sidebar-brand .name {{ font-family: 'Sora', sans-serif; font-weight: 700; font-size: 1.05rem; color: {TEXT}; }}
        .sidebar-brand .tag {{ font-family: 'JetBrains Mono', monospace; font-size: 0.68rem; color: {MINT}; }}

        .sidebar-mini {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.72rem;
            color: {MUTED};
            background: #f8fafc;
            border: 1px solid {BORDER};
            border-radius: 8px;
            padding: 0.6rem 0.75rem;
            margin-top: 0.5rem;
        }}

        [data-testid="stDataFrame"] {{ border-radius: 10px; overflow: hidden; border: 1px solid {BORDER}; }}
        hr {{ border-color: {BORDER} !important; }}
        [data-testid="stMetricValue"] {{ font-family: 'Sora', sans-serif; color: {TEXT}; }}
        [data-testid="stMetricLabel"] {{ color: {MUTED}; }}

        ::-webkit-scrollbar {{ width: 9px; }}
        ::-webkit-scrollbar-track {{ background: #f1f5f9; }}
        ::-webkit-scrollbar-thumb {{ background: #cbd5e1; border-radius: 8px; }}
        </style>
        """, unsafe_allow_html=True)
    else:
        # Dark Theme (Aurora Noir)
        MINT = "#00E5A0"
        AMBER = "#FFC857"
        MAGENTA = "#FF3D9A"
        INK = "#080A0B"
        PANEL = "#101317"
        TEXT = "#EAF2EF"
        MUTED = "#7C8B87"

        st.markdown(f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Sora:wght@500;600;700;800&family=JetBrains+Mono:wght@400;500;600&family=Inter:wght@400;500;600&display=swap');

        html, body, [class*="css"] {{
            font-family: 'Inter', sans-serif;
            color: {TEXT};
        }}
        h1, h2, h3, h4, h5, h6 {{
            font-family: 'Sora', sans-serif;
            letter-spacing: -0.015em;
            color: {TEXT} !important;
        }}
        code, .mono {{ font-family: 'JetBrains Mono', monospace; }}

        .stApp {{
            background:
                radial-gradient(circle at 12% 8%, rgba(0,229,160,0.10) 0%, transparent 38%),
                radial-gradient(circle at 88% 15%, rgba(255,61,154,0.09) 0%, transparent 40%),
                radial-gradient(circle at 50% 95%, rgba(255,200,87,0.07) 0%, transparent 45%),
                {INK};
        }}

        .block-container {{ padding-top: 3.8rem; padding-bottom: 3rem; max-width: 1240px; }}

        .topstrip {{
            display: flex; justify-content: space-between; align-items: center;
            padding: 0.55rem 1.1rem;
            background: {PANEL};
            border: 1px solid rgba(255,255,255,0.06);
            border-radius: 10px;
            margin-bottom: 1.4rem;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.74rem;
            color: {MUTED};
        }}
        .topstrip .dot {{ color: {MINT}; }}

        .page-eyebrow {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.72rem;
            letter-spacing: 0.14em;
            text-transform: uppercase;
            color: {MINT};
            margin-bottom: 0.3rem;
        }}
        .page-title {{
            font-size: 2.05rem;
            font-weight: 800;
            margin: 0 0 0.35rem 0;
            color: {TEXT};
        }}
        .page-sub {{ color: {MUTED}; font-size: 0.95rem; margin-bottom: 1.6rem; }}

        .rail-card {{
            background: {PANEL};
            border: 1px solid rgba(255,255,255,0.06);
            border-left: 3px solid var(--rail-color, {MINT});
            border-radius: 4px 12px 12px 4px;
            padding: 1.1rem 1.3rem;
            height: 100%;
        }}
        .rail-label {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.7rem;
            text-transform: uppercase;
            letter-spacing: 0.09em;
            color: {MUTED};
            margin-bottom: 0.3rem;
        }}
        .rail-value {{
            font-family: 'Sora', sans-serif;
            font-size: 1.85rem;
            font-weight: 700;
            color: {TEXT};
            line-height: 1.05;
        }}
        .rail-foot {{ font-size: 0.78rem; color: {MUTED}; margin-top: 0.35rem; }}

        .pill {{
            display: inline-block; padding: 3px 12px; border-radius: 999px;
            font-size: 0.76rem; font-weight: 700; font-family: 'JetBrains Mono', monospace;
            letter-spacing: 0.03em;
        }}
        .pill-high {{ background: rgba(255,61,154,0.14); color: {MAGENTA}; border: 1px solid rgba(255,61,154,0.35); }}
        .pill-medium {{ background: rgba(255,200,87,0.14); color: {AMBER}; border: 1px solid rgba(255,200,87,0.35); }}
        .pill-low {{ background: rgba(0,229,160,0.14); color: {MINT}; border: 1px solid rgba(0,229,160,0.35); }}

        .msg-row {{ display: flex; margin: 0.5rem 0; }}
        .msg-row.user {{ justify-content: flex-end; }}
        .bubble-user {{
            background: linear-gradient(135deg, #0d3d31, #0a2620);
            border: 1px solid rgba(0,229,160,0.28);
            color: {TEXT};
            padding: 0.75rem 1.1rem;
            border-radius: 14px 14px 2px 14px;
            max-width: 78%;
            font-size: 0.93rem;
        }}
        .bubble-agent {{
            background: {PANEL};
            border: 1px solid rgba(255,255,255,0.07);
            border-left: 3px solid {MAGENTA};
            color: {TEXT};
            padding: 0.85rem 1.15rem;
            border-radius: 2px 14px 14px 14px;
            max-width: 88%;
            font-size: 0.93rem;
        }}
        .agent-tag {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.68rem;
            color: {MAGENTA};
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin-bottom: 0.3rem;
            display: block;
        }}

        .stButton > button {{
            background: {PANEL};
            border: 1px solid rgba(255,255,255,0.09);
            color: {TEXT};
            border-radius: 9px;
            font-size: 0.85rem;
            font-weight: 500;
            transition: all 0.15s ease;
        }}
        .stButton > button:hover {{
            border-color: {MINT};
            color: {MINT};
            box-shadow: 0 0 0 1px rgba(0,229,160,0.25), 0 4px 16px rgba(0,229,160,0.12);
        }}

        .stChatInputContainer, div[data-testid="stChatInput"] {{
            border-color: rgba(0,229,160,0.25) !important;
        }}

        section[data-testid="stSidebar"] {{
            background: #0a0c0e;
            border-right: 1px solid rgba(255,255,255,0.06);
        }}
        section[data-testid="stSidebar"] .stRadio label {{
            font-size: 0.92rem;
            color: {TEXT};
        }}
        .sidebar-brand {{
            display: flex; align-items: center; gap: 0.6rem;
            padding: 0.4rem 0 1rem 0;
        }}
        .sidebar-brand .mark {{
            width: 34px; height: 34px; border-radius: 9px;
            background: linear-gradient(135deg, {MINT}, {MAGENTA});
            display: flex; align-items: center; justify-content: center;
            font-family: 'Sora', sans-serif; font-weight: 800; color: {INK}; font-size: 1.05rem;
        }}
        .sidebar-brand .name {{ font-family: 'Sora', sans-serif; font-weight: 700; font-size: 1.05rem; color: {TEXT}; }}
        .sidebar-brand .tag {{ font-family: 'JetBrains Mono', monospace; font-size: 0.68rem; color: {MINT}; }}

        .sidebar-mini {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.72rem;
            color: {MUTED};
            background: {PANEL};
            border: 1px solid rgba(255,255,255,0.06);
            border-radius: 8px;
            padding: 0.6rem 0.75rem;
            margin-top: 0.5rem;
        }}

        [data-testid="stDataFrame"] {{ border-radius: 10px; overflow: hidden; }}
        hr {{ border-color: rgba(255,255,255,0.07) !important; }}
        [data-testid="stMetricValue"] {{ font-family: 'Sora', sans-serif; color: {TEXT}; }}
        [data-testid="stMetricLabel"] {{ color: {MUTED}; }}

        ::-webkit-scrollbar {{ width: 9px; }}
        ::-webkit-scrollbar-track {{ background: {INK}; }}
        ::-webkit-scrollbar-thumb {{ background: linear-gradient(180deg, {MINT}, {MAGENTA}); border-radius: 8px; }}
        </style>
        """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ═══════════════════════════════════════════════════════════════════════════
def init_state():
    defaults = {
        "messages": [],
        "agent": None,
        "inject_query": None,
        "lookup_id": "7590-VHVEG",
        "page": "Overview",
        "theme": "Light",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ═══════════════════════════════════════════════════════════════════════════
# CACHED BACKEND ACCESSORS
# ═══════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def get_agent():
    from app.llm.provider import create_provider
    from app.agent.agent import ChurnAnalystAgent
    return ChurnAnalystAgent(create_provider())


@st.cache_data(show_spinner=False)
def get_dataset_info_cached():
    from app.data.loader import load_dataset, get_dataset_info
    return get_dataset_info(load_dataset())


@st.cache_data(show_spinner=False)
def get_model_info_cached():
    from app.agent.tools import tool_get_model_info
    result = tool_get_model_info()
    return result["data"] if result.get("success") else None


@st.cache_data(show_spinner=False)
def get_all_predictions_cached():
    from app.model.predictor import predict_all_customers
    return predict_all_customers()


def risk_pill(level: str) -> str:
    cls = {"High": "pill-high", "Medium": "pill-medium", "Low": "pill-low"}.get(level, "pill-low")
    return f'<span class="pill {cls}">{level.upper()} RISK</span>'


# ═══════════════════════════════════════════════════════════════════════════
# SIDEBAR — brand + navigation + live status + theme toggle
# ═══════════════════════════════════════════════════════════════════════════
def render_sidebar():
    with st.sidebar:
        st.markdown("""
        <div class="sidebar-brand">
            <div class="mark">◆</div>
            <div>
                <div class="name">Churn Analyst</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Theme Selector Radio
        selected_theme = st.radio(
            "Theme",
            ["Dark", "Light"],
            index=0 if st.session_state.theme == "Dark" else 1,
            horizontal=True,
            key="theme_radio_selector",
        )
        if selected_theme != st.session_state.theme:
            st.session_state.theme = selected_theme
            st.rerun()

        st.markdown("---")

        st.session_state.page = st.radio(
            "Navigate",
            ["Overview", "Ask the Agent", "Customer Lens", "Risk Ledger", "System"],
            label_visibility="collapsed",
        )

        st.markdown("---")

        TEXT_CLR = "#0F172A" if st.session_state.theme == "Light" else "#EAF2EF"
        MAGENTA_CLR = "#E11D48" if st.session_state.theme == "Light" else "#FF3D9A"
        MINT_CLR = "#059669" if st.session_state.theme == "Light" else "#00E5A0"
        AMBER_CLR = "#D97706" if st.session_state.theme == "Light" else "#FFC857"

        try:
            info = get_dataset_info_cached()
            st.markdown(f"""
            <div class="sidebar-mini">
                ROWS &nbsp;<span style="color:{TEXT_CLR}">{info['n_rows']:,}</span><br>
                CHURN &nbsp;<span style="color:{MAGENTA_CLR}">{info['churn_rate_pct']}%</span><br>
                MISSING &nbsp;<span style="color:{MINT_CLR}">{info.get('n_missing_total', 0)}</span>
            </div>
            """, unsafe_allow_html=True)
        except Exception:
            st.markdown('<div class="sidebar-mini">dataset unavailable</div>', unsafe_allow_html=True)

        try:
            mi = get_model_info_cached()
            if mi:
                m = mi.get("test_metrics", {})
                st.markdown(f"""
                <div class="sidebar-mini">
                    MODEL &nbsp;<span style="color:{TEXT_CLR}">{mi.get('model_name','?')}</span><br>
                    ROC-AUC &nbsp;<span style="color:{MINT_CLR}">{m.get('roc_auc',0):.3f}</span><br>
                    F1 &nbsp;<span style="color:{AMBER_CLR}">{m.get('f1_at_0.5', m.get('f1', 0)):.3f}</span>
                </div>
                """, unsafe_allow_html=True)
        except Exception:
            pass

        st.markdown("---")
        if st.button("Clear conversation", use_container_width=True):
            st.session_state.messages = []
            agent = st.session_state.get("agent")
            if agent and hasattr(agent, "clear_memory"):
                agent.clear_memory()
            st.rerun()


# ═══════════════════════════════════════════════════════════════════════════
# TOP STATUS STRIP
# ═══════════════════════════════════════════════════════════════════════════
def render_topstrip():
    from app.config import GROQ_API_KEY, MODEL_NAME
    llm_status = f'<span class="dot">●</span> groq · {MODEL_NAME}' if GROQ_API_KEY else '<span style="color:#FF3D9A">●</span> no llm key set'
    st.markdown(f"""
    <div class="topstrip">
        <div>{llm_status}</div>
        <div>autonomous-data-analyst // build local</div>
    </div>
    """, unsafe_allow_html=True)


def page_header(eyebrow: str, title: str, sub: str):
    st.markdown(f'<div class="page-eyebrow">{eyebrow}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="page-title">{title}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="page-sub">{sub}</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════
def page_overview():
    page_header("01 · Portfolio", "Overview", "Live KPIs computed directly from the dataset and trained model")

    theme = st.session_state.theme
    MINT_CLR = "#059669" if theme == "Light" else "#00E5A0"
    MAGENTA_CLR = "#E11D48" if theme == "Light" else "#FF3D9A"
    AMBER_CLR = "#D97706" if theme == "Light" else "#FFC857"

    try:
        from app.data.loader import load_dataset
        from app.visualization.charts import generate_chart
        df = load_dataset()
        info = get_dataset_info_cached()
        mi = get_model_info_cached()

        c1, c2, c3, c4 = st.columns(4)
        cards = [
            (c1, "Customers", f"{info['n_rows']:,}", "total portfolio", MINT_CLR),
            (c2, "Churn Rate", f"{info['churn_rate_pct']}%", f"{info.get('churn_yes', 0):,} churned", MAGENTA_CLR),
            (c3, "Retention", f"{100 - info['churn_rate_pct']:.1f}%", f"{info.get('churn_no', 0):,} retained", MINT_CLR),
            (c4, "Model ROC-AUC", f"{(mi or {}).get('test_metrics', {}).get('roc_auc', 0):.3f}", "held-out test set", AMBER_CLR),
        ]
        for col, label, val, foot, color in cards:
            with col:
                st.markdown(f"""
                <div class="rail-card" style="--rail-color:{color}">
                    <div class="rail-label">{label}</div>
                    <div class="rail-value">{val}</div>
                    <div class="rail-foot">{foot}</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        col_a, col_b = st.columns(2)
        with col_a:
            fig = generate_chart("churn_distribution", theme=theme)
            if fig:
                st.plotly_chart(go.Figure(fig), use_container_width=True)
        with col_b:
            fig = generate_chart("churn_by_column", column="Contract", theme=theme)
            if fig:
                st.plotly_chart(go.Figure(fig), use_container_width=True)

        col_c, col_d = st.columns(2)
        with col_c:
            fig = generate_chart("tenure_trend", theme=theme)
            if fig:
                st.plotly_chart(go.Figure(fig), use_container_width=True)
        with col_d:
            fig = generate_chart("monthly_charges_by_churn", theme=theme)
            if fig:
                st.plotly_chart(go.Figure(fig), use_container_width=True)

    except Exception as e:
        logger.error(f"Overview page error: {e}")
        st.error("Could not render the overview — check that the model has been trained (`python run.py train`).")


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 2 — ASK THE AGENT
# ═══════════════════════════════════════════════════════════════════════════
def page_chat():
    page_header("02 · Agent", "Ask the Agent", "Plans, calls tools, verifies its own output, and only reports numbers it actually computed.")

    prompts = [
        "What is the overall churn rate?",
        "Which contract type has the highest churn?",
        "Show the top 10 highest-risk customers.",
        "Does tenure correlate with churn?",
    ]
    cols = st.columns(len(prompts))
    for col, p in zip(cols, prompts):
        with col:
            if st.button(p, use_container_width=True, key=f"chip_{p}"):
                st.session_state.inject_query = p

    st.markdown("---")

    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(f'<div class="msg-row user"><div class="bubble-user">{msg["content"]}</div></div>', unsafe_allow_html=True)
        else:
            st.markdown(
                f'<div class="msg-row"><div class="bubble-agent"><span class="agent-tag">agent</span>{msg["content"]}</div></div>',
                unsafe_allow_html=True,
            )
            if msg.get("tool_results"):
                with st.expander("View tool calls & verification trace", expanded=False):
                    for tr in msg["tool_results"]:
                        name = tr.get("tool", "unknown")
                        result = tr.get("result", {})
                        ok = result.get("success", result.get("ok"))
                        st.markdown(f"`{name}` — {'✓ verified' if ok else '✗ failed'}")
                        data = result.get("data", result.get("result"))
                        if isinstance(data, list) and data and isinstance(data[0], dict):
                            st.dataframe(pd.DataFrame(data).head(15), use_container_width=True)
                        elif isinstance(data, dict):
                            st.json(data)
                        elif data is not None:
                            st.write(data)

    injected = st.session_state.pop("inject_query", None)
    user_input = st.chat_input("Ask about churn drivers, segments, or a specific customer...")
    query = injected or user_input

    if query:
        st.session_state.messages.append({"role": "user", "content": query})
        if st.session_state.agent is None:
            try:
                st.session_state.agent = get_agent()
            except Exception as e:
                logger.error(f"Agent init failed: {e}")
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": "Could not initialize the agent. Make sure the model is trained (`python run.py train`) and an LLM API key is set.",
                    "tool_results": [],
                })
                st.rerun()

        with st.spinner("planning → executing tools → verifying..."):
            try:
                result = st.session_state.agent.answer(query)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result["answer"],
                    "tool_results": result.get("tool_results", []),
                })
            except Exception:
                logger.error(traceback.format_exc())
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": "Something went wrong while processing that question. The error has been logged.",
                    "tool_results": [],
                })
        st.rerun()


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 3 — CUSTOMER LENS
# ═══════════════════════════════════════════════════════════════════════════
def page_customer_lens():
    page_header("03 · Individual", "Customer Lens", "Look up a real customer's churn risk, see what's driving it, and simulate hypothetical changes.")

    theme = st.session_state.theme
    MUTED_CLR = "#64748B" if theme == "Light" else "#7C8B87"

    samples = ["7590-VHVEG", "5575-GNVDE", "3668-QVRZG", "7795-CFOCW"]
    cols = st.columns(len(samples) + 1)
    for col, sid in zip(cols[:-1], samples):
        with col:
            if st.button(sid, use_container_width=True, key=f"sample_{sid}"):
                st.session_state.lookup_id = sid
    with cols[-1]:
        st.markdown(f'<div style="padding-top:0.5rem;color:{MUTED_CLR};font-size:0.8rem;">sample IDs</div>', unsafe_allow_html=True)

    lookup_id = st.text_input("Customer ID", value=st.session_state.lookup_id)
    st.session_state.lookup_id = lookup_id.strip()

    if not lookup_id:
        return

    from app.model.predictor import predict_churn_risk, predict_hypothetical
    from app.visualization.charts import chart_risk_gauge

    result = predict_churn_risk(lookup_id.strip())
    if "error" in result:
        st.error(result["error"])
        return

    score = result["risk_score"]
    level = result["risk_level"]
    feats = result.get("customer_features", {})

    st.markdown("---")
    col_gauge, col_info = st.columns([4, 6])
    with col_gauge:
        st.plotly_chart(go.Figure(chart_risk_gauge(score, lookup_id.strip(), theme=st.session_state.theme)), use_container_width=True)
        st.markdown(f'<div style="text-align:center">{risk_pill(level)}</div>', unsafe_allow_html=True)

    with col_info:
        st.markdown(f"**Prediction:** `{result['prediction']}` &nbsp;·&nbsp; **Score:** `{score:.4f}`")
        rows = [
            ("Contract", feats.get("Contract", "N/A")),
            ("Tenure (months)", feats.get("tenure", "N/A")),
            ("Monthly Charges", f"${feats.get('MonthlyCharges', 0):.2f}" if feats.get("MonthlyCharges") is not None else "N/A"),
            ("Payment Method", feats.get("PaymentMethod", "N/A")),
            ("Internet Service", feats.get("InternetService", "N/A")),
            ("Tech Support", feats.get("TechSupport", "N/A")),
        ]
        st.table(pd.DataFrame(rows, columns=["Attribute", "Value"]))

    st.markdown("---")
    st.markdown("#### Key factors")
    top_factors = result.get("top_factors", [])
    if top_factors:
        f_cols = st.columns(min(len(top_factors), 5))
        theme = st.session_state.theme
        MINT_CLR = "#059669" if theme == "Light" else "#00E5A0"
        MAGENTA_CLR = "#E11D48" if theme == "Light" else "#FF3D9A"
        for i, f in enumerate(top_factors[:5]):
            with f_cols[i]:
                direction = f.get("direction", "")
                up = direction == "increases_risk"
                color = MAGENTA_CLR if up else MINT_CLR
                arrow = "▲" if up else "▼"
                shap_val = f.get("shap_value") or f.get("importance", 0.0)
                st.markdown(f"""
                <div class="rail-card" style="--rail-color:{color}; text-align:center;">
                    <div class="rail-label">{f['feature']}</div>
                    <div class="rail-value" style="font-size:1.1rem;">{f.get('raw_value','')}</div>
                    <div style="color:{color}; font-weight:700; font-size:0.8rem; margin-top:0.3rem;">{arrow} {shap_val:+.3f}</div>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.info("No factor breakdown available for this customer.")

    st.markdown("---")
    st.markdown("#### What-if simulator")
    st.caption("Change contract, payment method, or tech support and see the risk shift — computed live by the model.")

    sc1, sc2, sc3 = st.columns(3)
    with sc1:
        new_contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"],
                                     index=["Month-to-month", "One year", "Two year"].index(feats.get("Contract", "Month-to-month")))
    with sc2:
        methods = ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
        new_payment = st.selectbox("Payment Method", methods,
                                    index=methods.index(feats.get("PaymentMethod", "Electronic check")))
    with sc3:
        ts_opts = ["Yes", "No", "No internet service"]
        new_ts = st.selectbox("Tech Support", ts_opts, index=ts_opts.index(feats.get("TechSupport", "No")))

    if st.button("Run simulation", use_container_width=True):
        changes = {}
        if new_contract != feats.get("Contract"):
            changes["Contract"] = new_contract
        if new_payment != feats.get("PaymentMethod"):
            changes["PaymentMethod"] = new_payment
        if new_ts != feats.get("TechSupport"):
            changes["TechSupport"] = new_ts

        if not changes:
            st.warning("Change at least one attribute to run a simulation.")
        else:
            sim = predict_hypothetical(lookup_id.strip(), changes)
            if "error" in sim:
                st.error(sim["error"])
            else:
                orig = sim["current"]["risk_score"]
                hypo = sim["hypothetical"]["risk_score"]
                delta = sim["risk_delta"]
                r1, r2, r3 = st.columns(3)
                r1.metric("Original", f"{orig*100:.1f}%")
                r2.metric("Simulated", f"{hypo*100:.1f}%", delta=f"{delta*100:+.1f}%", delta_color="inverse")
                r3.metric("New tier", sim["hypothetical"]["risk_level"])
                if delta < 0:
                    st.success(f"Risk drops by {abs(delta)*100:.1f} points under this configuration.")
                else:
                    st.warning(f"Risk rises by {delta*100:.1f} points under this configuration.")


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 4 — RISK LEDGER
# ═══════════════════════════════════════════════════════════════════════════
def page_risk_ledger():
    page_header("04 · Full Portfolio", "Risk Ledger", "Every customer, scored by the model. Filter, search, and export.")

    try:
        risk_df = get_all_predictions_cached()

        c1, c2 = st.columns([3, 5])
        with c1:
            tiers = st.multiselect("Risk tier", ["High", "Medium", "Low"], default=["High", "Medium", "Low"])
        with c2:
            search = st.text_input("Search customer ID", placeholder="e.g. 7590-")

        filtered = risk_df[risk_df["risk_level"].isin(tiers)]
        if search.strip():
            filtered = filtered[filtered["customerID"].str.contains(search.strip(), case=False, na=False)]
        filtered = filtered.sort_values("risk_score", ascending=False)

        st.markdown(f'<div class="sidebar-mini" style="display:inline-block;">showing {len(filtered):,} of {len(risk_df):,}</div>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        st.dataframe(
            filtered,
            column_config={
                "customerID": "Customer ID",
                "risk_score": st.column_config.ProgressColumn("Risk Score", format="%.4f", min_value=0.0, max_value=1.0),
                "risk_level": "Tier",
                "prediction": "Prediction",
            },
            use_container_width=True,
            height=480,
        )

        e1, e2 = st.columns(2)
        with e1:
            st.download_button("Export CSV", data=filtered.to_csv(index=False).encode("utf-8"),
                                file_name="churn_risk_predictions.csv", mime="text/csv", use_container_width=True)
        with e2:
            st.download_button("Export JSON", data=filtered.to_json(orient="records", indent=2).encode("utf-8"),
                                file_name="churn_risk_predictions.json", mime="application/json", use_container_width=True)
    except Exception as e:
        logger.error(f"Risk ledger error: {e}")
        st.error("Could not compute portfolio-wide predictions.")


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 5 — SYSTEM
# ═══════════════════════════════════════════════════════════════════════════
def page_system():
    page_header("05 · Under the hood", "System", "Model card, agent architecture, and environment status.")

    theme = st.session_state.theme
    MINT_CLR = "#059669" if theme == "Light" else "#00E5A0"
    MAGENTA_CLR = "#E11D48" if theme == "Light" else "#FF3D9A"
    AMBER_CLR = "#D97706" if theme == "Light" else "#FFC857"

    mi = get_model_info_cached()
    if mi:
        m = mi.get("test_metrics", {})
        cols = st.columns(5)
        specs = [
            ("Model", mi.get("model_name", "?")),
            ("ROC-AUC", f"{m.get('roc_auc', 0):.3f}"),
            ("PR-AUC", f"{m.get('pr_auc', 0):.3f}"),
            ("F1", f"{m.get('f1_at_0.5', m.get('f1', 0)):.3f}"),
            ("Recall", f"{m.get('recall_at_0.5', m.get('recall', 0)):.3f}"),
        ]
        for col, (label, val) in zip(cols, specs):
            with col:
                st.markdown(f"""
                <div class="rail-card" style="--rail-color:{MINT_CLR}">
                    <div class="rail-label">{label}</div>
                    <div class="rail-value" style="font-size:1.4rem;">{val}</div>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.warning("Model not trained yet — run `python run.py train`.")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### Agent loop")
    st.markdown(f"""
    <div class="rail-card" style="--rail-color:{MAGENTA_CLR}">
    <span class="mono" style="color:{MAGENTA_CLR}">question</span> → intent understanding → planning → tool selection →
    tool execution → <span class="mono" style="color:{MINT_CLR}">verification</span> → retry/re-plan if invalid →
    grounded final answer
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    from app.config import GROQ_API_KEY, MODEL_NAME, DATA_PATH
    st.markdown("#### Environment")
    st.markdown(f"""
    <div class="rail-card" style="--rail-color:{AMBER_CLR}">
    <span class="mono">LLM provider:</span> {'groq · ' + MODEL_NAME if GROQ_API_KEY else 'not configured'}<br>
    <span class="mono">Data path:</span> {DATA_PATH}
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════
def main():
    init_state()
    render_theme_css(st.session_state.theme)
    render_sidebar()
    render_topstrip()

    page = st.session_state.page
    if page == "Overview":
        page_overview()
    elif page == "Ask the Agent":
        page_chat()
    elif page == "Customer Lens":
        page_customer_lens()
    elif page == "Risk Ledger":
        page_risk_ledger()
    elif page == "System":
        page_system()


if __name__ == "__main__":
    main()