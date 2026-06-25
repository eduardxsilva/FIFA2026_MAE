from __future__ import annotations

import tempfile
from html import escape
from io import BytesIO
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

try:
    from streamlit_option_menu import option_menu
except Exception:  # fallback para deploys sem o pacote opcional
    option_menu = None

from fifa2026_core import (
    InternetDataExtractor,
    MatchDataLoader,
    DynamicEloModel,
    BasePoissonModel,
    MLMatchOutcomeModel,
    EnsemblePredictionModel,
    MonteCarloChampionSimulator,
    WorldCupFormatSimulator,
    ModelBacktester,
)


# ============================================================
# CONFIGURAÇÃO STREAMLIT
# ============================================================

st.set_page_config(
    page_title="FIFA 2026 Analytics",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# ÍCONES OUTLINE SVG — estilo Lucide / liquid glass
# Sem emoji, sem preenchimento, apenas contorno.
# ============================================================

ICON_PATHS: dict[str, str] = {
    "activity": '<path d="M22 12h-4l-3 9L9 3l-3 9H2"/>',
    "arrow-up-right": '<path d="M7 7h10v10"/><path d="M7 17 17 7"/>',
    "bar-chart": '<path d="M3 3v18h18"/><path d="M7 16V8"/><path d="M12 16V5"/><path d="M17 16v-4"/>',
    "brain": '<path d="M12 5a3 3 0 0 0-5.83-1"/><path d="M12 5a3 3 0 0 1 5.83-1"/><path d="M7 4a4 4 0 0 0-2.5 7"/><path d="M17 4a4 4 0 0 1 2.5 7"/><path d="M5 11a4 4 0 0 0 1.5 7"/><path d="M19 11a4 4 0 0 1-1.5 7"/><path d="M8 18a4 4 0 0 0 4 3"/><path d="M16 18a4 4 0 0 1-4 3"/><path d="M12 5v16"/>',
    "calendar": '<path d="M8 2v4"/><path d="M16 2v4"/><rect x="3" y="4" width="18" height="18" rx="2"/><path d="M3 10h18"/>',
    "cloud-download": '<path d="M12 13v8"/><path d="m8 17 4 4 4-4"/><path d="M20.39 18.39A5 5 0 0 0 18 9h-1.26A8 8 0 1 0 3 16.3"/>',
    "database": '<ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M3 5v14c0 1.66 4.03 3 9 3s9-1.34 9-3V5"/><path d="M3 12c0 1.66 4.03 3 9 3s9-1.34 9-3"/>',
    "download": '<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><path d="M7 10l5 5 5-5"/><path d="M12 15V3"/>',
    "gauge": '<path d="M12 14l4-4"/><path d="M3.34 19a10 10 0 1 1 17.32 0"/>',
    "globe": '<circle cx="12" cy="12" r="10"/><path d="M2 12h20"/><path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z"/>',
    "home": '<path d="m3 10 9-7 9 7"/><path d="M5 10v10a1 1 0 0 0 1 1h12a1 1 0 0 0 1-1V10"/><path d="M9 21v-6h6v6"/>',
    "layers": '<path d="m12 2 9 5-9 5-9-5 9-5Z"/><path d="m3 12 9 5 9-5"/><path d="m3 17 9 5 9-5"/>',
    "line-chart": '<path d="M3 3v18h18"/><path d="m19 9-5 5-4-4-3 3"/>',
    "medal": '<path d="M7.21 15 2.66 7.14A2 2 0 0 1 4.39 4h15.22a2 2 0 0 1 1.73 3.14L16.79 15"/><path d="M11 12 5.12 4"/><path d="m13 12 5.88-8"/><circle cx="12" cy="17" r="5"/>',
    "sparkles": '<path d="m12 3-1.9 5.8L4 11l6.1 2.2L12 19l1.9-5.8L20 11l-6.1-2.2L12 3Z"/><path d="M5 3v4"/><path d="M3 5h4"/><path d="M19 17v4"/><path d="M17 19h4"/>',
    "target": '<circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/>',
    "trophy": '<path d="M8 21h8"/><path d="M12 17v4"/><path d="M7 4h10v5a5 5 0 0 1-10 0V4Z"/><path d="M5 5H3v2a4 4 0 0 0 4 4"/><path d="M19 5h2v2a4 4 0 0 1-4 4"/>',
    "upload": '<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><path d="M17 8l-5-5-5 5"/><path d="M12 3v12"/>',
    "wand": '<path d="M15 4V2"/><path d="M15 16v-2"/><path d="M8 9H6"/><path d="M20 9h-2"/><path d="M17.8 6.2 19 5"/><path d="M6.2 17.8 5 19"/><path d="m3 21 9-9"/><path d="M12.2 6.2 11 5"/>',
}


def icon_svg(name: str, size: int = 22, stroke: float = 1.8) -> str:
    paths = ICON_PATHS.get(name, ICON_PATHS["sparkles"])
    return (
        f'<svg class="line-icon" xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="{stroke}" '
        f'stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">{paths}</svg>'
    )


# ============================================================
# CSS LIQUID GLASS
# ============================================================

CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@20..48,300..500,0,0&display=swap');

    :root {
        --ink: rgba(255,255,255,0.92);
        --muted: rgba(255,255,255,0.66);
        --faint: rgba(255,255,255,0.46);
        --glass: rgba(255,255,255,0.075);
        --glass-strong: rgba(255,255,255,0.115);
        --glass-soft: rgba(255,255,255,0.045);
        --stroke: rgba(255,255,255,0.155);
        --stroke-strong: rgba(255,255,255,0.25);
        --accent: #7dd3fc;
        --accent-2: #a78bfa;
        --accent-3: #34d399;
        --danger: #fb7185;
        --warning: #fbbf24;
        --shadow: rgba(0,0,0,0.42);
    }

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }

    /* Correção crítica: não deixe o CSS global transformar ícones internos do Streamlit
       em texto bruto como "keyboard_double_arrow_left". */
    .material-symbols-rounded,
    .material-symbols-outlined,
    .material-icons,
    span[data-testid="stIconMaterial"],
    [class*="material-symbols"] {
        font-family: 'Material Symbols Rounded' !important;
        font-weight: normal !important;
        font-style: normal !important;
        font-size: 20px !important;
        line-height: 1 !important;
        letter-spacing: normal !important;
        text-transform: none !important;
        display: inline-flex !important;
        white-space: nowrap !important;
        word-wrap: normal !important;
        direction: ltr !important;
        -webkit-font-feature-settings: 'liga' !important;
        -webkit-font-smoothing: antialiased !important;
        font-feature-settings: 'liga' !important;
        font-variation-settings: 'FILL' 0, 'wght' 400, 'GRAD' 0, 'opsz' 24 !important;
    }

    /* Correção para Bootstrap Icons usados pelo streamlit-option-menu. */
    .bi,
    i.bi,
    [class^="bi-"],
    [class*=" bi-"] {
        font-family: bootstrap-icons !important;
        font-style: normal !important;
        font-weight: normal !important;
        line-height: 1 !important;
        text-transform: none !important;
        letter-spacing: normal !important;
    }

    .stApp {
        color: var(--ink);
        background:
            radial-gradient(circle at 8% 5%, rgba(125,211,252,0.26), transparent 30%),
            radial-gradient(circle at 86% 12%, rgba(167,139,250,0.25), transparent 34%),
            radial-gradient(circle at 45% 90%, rgba(52,211,153,0.16), transparent 38%),
            linear-gradient(135deg, #030712 0%, #08111f 42%, #0c1222 100%);
        background-attachment: fixed;
    }

    .stApp::before {
        content: "";
        position: fixed;
        inset: 0;
        pointer-events: none;
        background-image:
            linear-gradient(rgba(255,255,255,0.028) 1px, transparent 1px),
            linear-gradient(90deg, rgba(255,255,255,0.024) 1px, transparent 1px);
        background-size: 38px 38px;
        mask-image: radial-gradient(circle at 50% 30%, black, transparent 72%);
        opacity: .5;
        z-index: 0;
    }

    .block-container {
        padding-top: 1.35rem;
        padding-bottom: 2.7rem;
        max-width: 1440px;
        position: relative;
        z-index: 1;
    }

    [data-testid="stSidebar"] {
        background:
            linear-gradient(180deg, rgba(3,7,18,0.72), rgba(8,13,28,0.78)),
            radial-gradient(circle at 20% 0%, rgba(125,211,252,0.20), transparent 38%);
        border-right: 1px solid rgba(255,255,255,0.13);
        box-shadow: 18px 0 40px rgba(0,0,0,0.22);
        backdrop-filter: blur(28px) saturate(160%);
        -webkit-backdrop-filter: blur(28px) saturate(160%);
    }

    [data-testid="stSidebar"] :not(.material-symbols-rounded):not(.material-symbols-outlined):not(.material-icons):not(.bi):not(svg):not(path) {
        font-family: 'Inter', sans-serif;
    }

    .brand-card {
        padding: 18px 16px 14px 16px;
        margin-bottom: 14px;
        border-radius: 24px;
        background: linear-gradient(145deg, rgba(255,255,255,0.13), rgba(255,255,255,0.045));
        border: 1px solid var(--stroke);
        box-shadow: inset 0 1px rgba(255,255,255,0.22), 0 16px 38px rgba(0,0,0,0.22);
    }

    .brand-title {
        display: flex;
        align-items: center;
        gap: 10px;
        font-size: 1.03rem;
        font-weight: 900;
        letter-spacing: -0.035em;
    }

    .brand-subtitle {
        color: var(--muted);
        font-size: .78rem;
        margin-top: 5px;
    }

    .hero {
        position: relative;
        overflow: hidden;
        padding: 34px 34px;
        border-radius: 34px;
        background:
            linear-gradient(135deg, rgba(255,255,255,0.135), rgba(255,255,255,0.045)),
            radial-gradient(circle at 12% 20%, rgba(125,211,252,0.42), transparent 26%),
            radial-gradient(circle at 80% 10%, rgba(167,139,250,0.34), transparent 32%),
            radial-gradient(circle at 62% 90%, rgba(52,211,153,0.20), transparent 36%);
        border: 1px solid rgba(255,255,255,0.18);
        box-shadow: inset 0 1px rgba(255,255,255,0.22), 0 30px 90px rgba(0,0,0,0.34);
        backdrop-filter: blur(24px) saturate(170%);
        -webkit-backdrop-filter: blur(24px) saturate(170%);
        margin-bottom: 24px;
    }

    .hero::after {
        content: "";
        position: absolute;
        inset: 1px;
        border-radius: 33px;
        background: linear-gradient(120deg, rgba(255,255,255,0.18), transparent 22%, transparent 70%, rgba(255,255,255,0.08));
        pointer-events: none;
    }

    .hero-title {
        display: flex;
        align-items: center;
        gap: 14px;
        margin: 0 0 10px 0;
        font-size: clamp(2rem, 4vw, 3.6rem);
        letter-spacing: -0.07em;
        line-height: .98;
        font-weight: 950;
    }

    .hero-title .line-icon {
        width: 44px;
        height: 44px;
        color: var(--accent);
        filter: drop-shadow(0 0 18px rgba(125,211,252,0.52));
    }

    .hero p {
        color: var(--muted);
        font-size: 1.03rem;
        max-width: 900px;
        margin: 0;
        line-height: 1.6;
    }

    .chip-row {
        display: flex;
        flex-wrap: wrap;
        gap: 9px;
        margin-top: 22px;
    }

    .chip {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 9px 12px;
        border-radius: 999px;
        background: rgba(255,255,255,0.085);
        border: 1px solid rgba(255,255,255,0.15);
        color: rgba(255,255,255,0.86);
        font-size: 0.83rem;
        font-weight: 650;
        white-space: nowrap;
        box-shadow: inset 0 1px rgba(255,255,255,0.14);
    }

    .chip .line-icon {
        width: 17px;
        height: 17px;
        color: var(--accent);
    }

    .glass-card, .metric-card, div[data-testid="stMetric"] {
        border-radius: 24px !important;
        background: linear-gradient(145deg, rgba(255,255,255,0.12), rgba(255,255,255,0.045)) !important;
        border: 1px solid rgba(255,255,255,0.16) !important;
        box-shadow: inset 0 1px rgba(255,255,255,0.20), 0 18px 48px rgba(0,0,0,0.24) !important;
        backdrop-filter: blur(22px) saturate(165%);
        -webkit-backdrop-filter: blur(22px) saturate(165%);
    }

    .metric-card {
        padding: 20px 20px;
        height: 100%;
        position: relative;
        overflow: hidden;
    }

    .metric-card::before {
        content: "";
        position: absolute;
        inset: -40% -20% auto auto;
        width: 160px;
        height: 160px;
        border-radius: 999px;
        background: radial-gradient(circle, rgba(125,211,252,0.24), transparent 70%);
        pointer-events: none;
    }

    .metric-icon {
        width: 42px;
        height: 42px;
        display: flex;
        align-items: center;
        justify-content: center;
        border-radius: 15px;
        background: rgba(125,211,252,0.12);
        border: 1px solid rgba(125,211,252,0.23);
        color: var(--accent);
        margin-bottom: 13px;
        box-shadow: inset 0 1px rgba(255,255,255,0.16);
    }

    .metric-label {
        color: var(--muted);
        font-size: 0.82rem;
        font-weight: 680;
        margin-bottom: 4px;
        text-transform: uppercase;
        letter-spacing: .07em;
    }

    .metric-value {
        font-size: 1.72rem;
        font-weight: 900;
        letter-spacing: -0.055em;
        margin-bottom: 3px;
    }

    .metric-note {
        color: var(--faint);
        font-size: 0.80rem;
        line-height: 1.4;
    }

    .section-title {
        display: flex;
        align-items: center;
        gap: 12px;
        font-size: 1.35rem;
        font-weight: 900;
        letter-spacing: -0.045em;
        margin: 26px 0 12px 0;
    }

    .section-title .line-icon {
        width: 24px;
        height: 24px;
        color: var(--accent);
    }

    .soft-text {
        color: var(--muted);
        font-size: 0.96rem;
        line-height: 1.6;
    }

    .stButton > button,
    .stDownloadButton > button {
        border-radius: 16px !important;
        border: 1px solid rgba(255,255,255,0.18) !important;
        background: linear-gradient(145deg, rgba(255,255,255,0.13), rgba(255,255,255,0.055)) !important;
        color: rgba(255,255,255,0.94) !important;
        font-weight: 800 !important;
        min-height: 46px !important;
        box-shadow: inset 0 1px rgba(255,255,255,0.20), 0 14px 36px rgba(0,0,0,0.20) !important;
        backdrop-filter: blur(18px) saturate(155%);
        -webkit-backdrop-filter: blur(18px) saturate(155%);
        transition: transform .18s ease, border-color .18s ease, background .18s ease;
    }

    .stButton > button:hover,
    .stDownloadButton > button:hover {
        transform: translateY(-1px);
        border-color: rgba(125,211,252,0.44) !important;
        background: linear-gradient(145deg, rgba(125,211,252,0.20), rgba(255,255,255,0.075)) !important;
    }

    button[kind="primary"],
    .stDownloadButton button[kind="primary"] {
        background: linear-gradient(135deg, rgba(125,211,252,0.35), rgba(167,139,250,0.26)) !important;
        border-color: rgba(125,211,252,0.42) !important;
    }

    [data-testid="stDataFrame"],
    [data-testid="stTable"] {
        border-radius: 22px;
        overflow: hidden;
        border: 1px solid rgba(255,255,255,0.12);
        box-shadow: 0 18px 48px rgba(0,0,0,0.18);
    }

    div[data-testid="stTabs"] button {
        border-radius: 999px;
        padding: 10px 16px;
        color: rgba(255,255,255,0.72);
    }

    div[data-testid="stTabs"] button[aria-selected="true"] {
        color: rgba(255,255,255,0.96);
        background: rgba(255,255,255,0.08);
        border: 1px solid rgba(255,255,255,0.14);
    }

    [data-testid="stSelectbox"], [data-testid="stNumberInput"], [data-testid="stTextInput"], [data-testid="stFileUploader"] {
        border-radius: 18px;
    }

    .stAlert {
        border-radius: 18px;
        backdrop-filter: blur(18px);
    }


    /* Evita textos/ícones internos estourando a página em telas pequenas. */
    .main .block-container, [data-testid="stSidebar"] {
        overflow-x: hidden;
    }

    /* Mobile polish */
    @media (max-width: 900px) {
        .hero { padding: 24px 20px; border-radius: 28px; }
        .hero-title { font-size: 2.05rem; }
        .brand-card { border-radius: 20px; }
        .chip { font-size: .78rem; padding: 8px 10px; }
        .metric-value { font-size: 1.35rem; }
    }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ============================================================
# ESTADO E HELPERS
# ============================================================


def init_state() -> None:
    defaults = {
        "df_matches": None,
        "internet_extras": {},
        "internet_log": [],
        "elo_model": None,
        "poisson_model": None,
        "ml_model": None,
        "ensemble_model": None,
        "simulador_campeao": None,
        "simulador_copa": None,
        "equipes_copa_2026": [],
        "df_copa2026_classificacao": None,
        "df_copa2026_classificados": None,
        "ultima_previsao": None,
        "ultima_previsao_ml": None,
        "ultima_previsao_ensemble": None,
        "df_elo": None,
        "df_team_stats": None,
        "df_simulacao_campeao": None,
        "df_simulacao_copa": None,
        "ultima_copa_simulada": None,
        "df_validacao_resumo": None,
        "df_validacao_previsoes": None,
        "fonte_base": None,
        "fifa_live_last_update": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def has_data() -> bool:
    return st.session_state.df_matches is not None and not st.session_state.df_matches.empty


def has_models() -> bool:
    return st.session_state.poisson_model is not None and st.session_state.df_team_stats is not None


def metric_card(icon_name: str, label: str, value: str, note: str = "") -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-icon">{icon_svg(icon_name, 23)}</div>
            <div class="metric-label">{escape(str(label))}</div>
            <div class="metric-value">{escape(str(value))}</div>
            <div class="metric-note">{escape(str(note))}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def hero() -> None:
    st.markdown(
        f"""
        <div class="hero">
            <div class="hero-title">{icon_svg('sparkles', 44, 1.7)} FIFA 2026 Analytics Engine</div>
            <p>
                Plataforma premium para previsão de partidas, Elo dinâmico, xG,
                Poisson bivariada, Dixon-Coles, Machine Learning, Ensemble e simulações Monte Carlo.
            </p>
            <div class="chip-row">
                <span class="chip">{icon_svg('brain', 17)} Ensemble estatístico + ML</span>
                <span class="chip">{icon_svg('line-chart', 17)} Validação temporal</span>
                <span class="chip">{icon_svg('trophy', 17)} Simulação de campeão</span>
                <span class="chip">{icon_svg('cloud-download', 17)} Extração online</span>
                <span class="chip">{icon_svg('download', 17)} Exportação Excel</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section(title: str, icon_name: str = "sparkles") -> None:
    st.markdown(
        f"<div class='section-title'>{icon_svg(icon_name, 24)} {escape(title)}</div>",
        unsafe_allow_html=True,
    )


def get_teams() -> list[str]:
    if not has_models():
        return []
    return sorted(st.session_state.df_team_stats["Equipe"].astype(str).tolist())


def get_copa_teams_for_model() -> list[str]:
    """Times da Copa 2026 detectados pela FIFA e presentes no modelo treinado."""
    if not has_models():
        return []
    model_teams = set(st.session_state.df_team_stats["Equipe"].astype(str).tolist())
    copa_teams = st.session_state.get("equipes_copa_2026", []) or []
    return sorted([team for team in copa_teams if team in model_teams])


def sync_copa_live_state_from_extras(extras: dict) -> None:
    """Atualiza o estado da Copa 2026 usando a última consulta solicitada à FIFA."""
    st.session_state.df_copa2026_classificacao = extras.get("copa2026_classificacao_atual")
    st.session_state.df_copa2026_classificados = extras.get("copa2026_classificados_atuais")

    df_update = extras.get("fifa_atualizacao")
    if isinstance(df_update, pd.DataFrame) and not df_update.empty and "Atualizado_UTC" in df_update.columns:
        st.session_state.fifa_live_last_update = str(df_update.iloc[0]["Atualizado_UTC"])

    equipes = []
    df_eq = extras.get("copa2026_equipes_oficiais")
    if isinstance(df_eq, pd.DataFrame) and not df_eq.empty and "Equipe" in df_eq.columns:
        equipes = df_eq["Equipe"].astype(str).dropna().drop_duplicates().tolist()
    elif isinstance(st.session_state.df_copa2026_classificacao, pd.DataFrame) and not st.session_state.df_copa2026_classificacao.empty:
        equipes = st.session_state.df_copa2026_classificacao["Equipe"].astype(str).dropna().drop_duplicates().tolist()

    st.session_state.equipes_copa_2026 = sorted([e for e in equipes if e and e.lower() != "nan"])


def style_plotly(fig):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.025)",
        font=dict(color="rgba(255,255,255,0.86)", family="Inter, Segoe UI, sans-serif"),
        title_font=dict(size=18, color="rgba(255,255,255,0.95)"),
        margin=dict(l=18, r=18, t=58, b=18),
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="rgba(255,255,255,0.10)"),
    )
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.075)", zerolinecolor="rgba(255,255,255,0.12)")
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.075)", zerolinecolor="rgba(255,255,255,0.12)")
    return fig


def reset_models() -> None:
    for key in [
        "elo_model",
        "poisson_model",
        "ml_model",
        "ensemble_model",
        "simulador_campeao",
        "simulador_copa",
        "df_elo",
        "df_team_stats",
        "ultima_previsao",
        "ultima_previsao_ml",
        "ultima_previsao_ensemble",
        "df_simulacao_campeao",
        "df_simulacao_copa",
        "ultima_copa_simulada",
        "df_validacao_resumo",
        "df_validacao_previsoes",
    ]:
        st.session_state[key] = None


def treinar_modelos() -> None:
    if not has_data():
        raise ValueError("Importe uma base antes de treinar.")

    df = st.session_state.df_matches.copy()

    elo_model = DynamicEloModel().treinar(df)
    poisson_model = BasePoissonModel(max_gols=10).treinar(df, elo_model=elo_model)

    ml_model = MLMatchOutcomeModel()
    try:
        ml_model.treinar(df, poisson_model=poisson_model, elo_model=elo_model)
    except Exception:
        ml_model.treinado = False

    ensemble_model = EnsemblePredictionModel()

    st.session_state.elo_model = elo_model
    st.session_state.poisson_model = poisson_model
    st.session_state.ml_model = ml_model
    st.session_state.ensemble_model = ensemble_model
    st.session_state.df_elo = elo_model.ranking()
    st.session_state.df_team_stats = poisson_model.df_team_stats.copy()

    st.session_state.simulador_campeao = MonteCarloChampionSimulator(
        poisson_model=poisson_model,
        ml_model=ml_model,
        ensemble_model=ensemble_model,
        random_state=42,
    )
    grupos_copa = st.session_state.get("df_copa2026_classificacao")
    equipes_copa = st.session_state.get("equipes_copa_2026", []) or []

    st.session_state.simulador_copa = WorldCupFormatSimulator(
        poisson_model=poisson_model,
        ml_model=ml_model,
        ensemble_model=ensemble_model,
        random_state=42,
        equipes_copa=equipes_copa,
        grupos_copa=grupos_copa if isinstance(grupos_copa, pd.DataFrame) and not grupos_copa.empty else None,
    )




def recriar_simulador_copa_com_estado_fifa() -> None:
    """Reaponta o simulador da Copa para as equipes/grupos da última atualização FIFA."""
    if not has_models():
        return

    grupos_copa = st.session_state.get("df_copa2026_classificacao")
    equipes_copa = st.session_state.get("equipes_copa_2026", []) or []

    st.session_state.simulador_copa = WorldCupFormatSimulator(
        poisson_model=st.session_state.poisson_model,
        ml_model=st.session_state.ml_model,
        ensemble_model=st.session_state.ensemble_model,
        random_state=42,
        equipes_copa=equipes_copa,
        grupos_copa=grupos_copa if isinstance(grupos_copa, pd.DataFrame) and not grupos_copa.empty else None,
    )

def dataframe_to_excel_bytes(sheets: dict[str, pd.DataFrame]) -> bytes:
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        used_names = set()
        for name, df in sheets.items():
            if df is None or not isinstance(df, pd.DataFrame) or df.empty:
                continue
            clean = str(name)[:31] or "sheet"
            original = clean
            idx = 1
            while clean in used_names:
                suffix = f"_{idx}"
                clean = (original[: 31 - len(suffix)] + suffix)[:31]
                idx += 1
            used_names.add(clean)
            df.to_excel(writer, sheet_name=clean, index=False)
    return buffer.getvalue()


init_state()


# ============================================================
# SIDEBAR
# ============================================================

PAGES = [
    "Dashboard",
    "Importar dados",
    "Treinar modelos",
    "Prever partida",
    "Simulações",
    "Validação",
    "Exportar",
]

with st.sidebar:
    st.markdown(
        f"""
        <div class="brand-card">
            <div class="brand-title">{icon_svg('layers', 22)} FIFA 2026</div>
            <div class="brand-subtitle">Liquid Glass Analytics Suite</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if option_menu is not None:
        page = option_menu(
            menu_title=None,
            options=PAGES,
            icons=["speedometer2", "cloud-download", "cpu", "crosshair", "trophy", "graph-up-arrow", "box-arrow-down"],
            default_index=0,
            styles={
                "container": {"padding": "0", "background-color": "transparent"},
                "icon": {"color": "#7dd3fc", "font-size": "18px"},
                "nav-link": {
                    "font-size": "14px",
                    "font-weight": "700",
                    "color": "rgba(255,255,255,0.74)",
                    "border-radius": "14px",
                    "padding": "12px 12px",
                    "margin": "4px 0",
                    "background-color": "transparent",
                },
                "nav-link-selected": {
                    "background": "linear-gradient(135deg, rgba(125,211,252,0.22), rgba(167,139,250,0.16))",
                    "color": "white",
                    "border": "1px solid rgba(255,255,255,0.14)",
                },
            },
        )
    else:
        page = st.radio("Navegação", PAGES, label_visibility="collapsed")

    st.divider()
    if has_data():
        st.success("Base carregada")
        st.caption(st.session_state.fonte_base or "Fonte não informada")
    else:
        st.warning("Sem base")

    if has_models():
        st.success("Modelos treinados")
        if st.session_state.ml_model and getattr(st.session_state.ml_model, "treinado", False):
            st.caption("Machine Learning ativo")
        else:
            st.caption("Machine Learning indisponível ou base pequena")
    else:
        st.info("Modelos não treinados")


# ============================================================
# DASHBOARD
# ============================================================

if page == "Dashboard":
    hero()

    if not has_data():
        col1, col2, col3 = st.columns(3)
        with col1:
            metric_card("cloud-download", "Passo 1", "Importe dados", "Use internet ou planilha local")
        with col2:
            metric_card("brain", "Passo 2", "Treine", "Elo, Poisson, ML e Ensemble")
        with col3:
            metric_card("trophy", "Passo 3", "Simule", "Partidas, Copa e campeão")

        st.info("Vá em Importar dados para começar.")
    else:
        df = st.session_state.df_matches.copy()
        teams = sorted(set(df["home_team"]).union(set(df["away_team"])))

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            metric_card("calendar", "Partidas", f"{len(df):,}".replace(",", "."), "jogos válidos")
        with col2:
            metric_card("globe", "Equipes", str(len(teams)), "seleções/equipes únicas")
        with col3:
            metric_card("line-chart", "Primeiro jogo", str(df["date"].min().date()), "início da série")
        with col4:
            metric_card("activity", "Último jogo", str(df["date"].max().date()), "fim da série")

        if st.session_state.equipes_copa_2026:
            section("Estado Copa 2026 — última atualização FIFA", "globe")
            c_live1, c_live2 = st.columns([1, 2])
            with c_live1:
                nota_update = st.session_state.fifa_live_last_update or "sem horário registrado"
                metric_card("trophy", "Seleções da Copa", str(len(st.session_state.equipes_copa_2026)), f"última consulta: {nota_update}")
            with c_live2:
                if isinstance(st.session_state.df_copa2026_classificacao, pd.DataFrame) and not st.session_state.df_copa2026_classificacao.empty:
                    st.dataframe(st.session_state.df_copa2026_classificacao.head(60), width="stretch", hide_index=True)
                else:
                    st.dataframe(pd.DataFrame({"Seleção": st.session_state.equipes_copa_2026}), width="stretch", hide_index=True)

        section("Visão geral da base", "bar-chart")
        df_year = df.assign(ano=df["date"].dt.year).groupby("ano").size().reset_index(name="partidas")
        fig = px.line(df_year, x="ano", y="partidas", markers=True, title="Partidas por ano")
        fig.update_traces(line_width=3, marker_size=8)
        fig.update_layout(height=370)
        st.plotly_chart(style_plotly(fig), width="stretch")

        c1, c2 = st.columns(2)
        with c1:
            top_home = df["home_team"].value_counts().head(15).reset_index()
            top_home.columns = ["Equipe", "Jogos como mandante"]
            fig_home = px.bar(top_home, x="Jogos como mandante", y="Equipe", orientation="h", title="Top mandantes")
            fig_home.update_layout(height=450, yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(style_plotly(fig_home), width="stretch")
        with c2:
            st.dataframe(df.tail(20), width="stretch", hide_index=True)


# ============================================================
# IMPORTAR DADOS
# ============================================================

elif page == "Importar dados":
    hero()
    section("Importar base", "cloud-download")

    tab_net, tab_file = st.tabs(["Extração da internet", "Planilha local"])

    with tab_net:
        st.markdown(
            "A base histórica treina o modelo. A FIFA é consultada separadamente quando você clicar para atualizar agora."
        )
        c1, c2, c3 = st.columns([1, 1, 2])
        with c1:
            ano_minimo = st.number_input("Ano mínimo", min_value=1872, max_value=2030, value=2018, step=1)
        with c2:
            incluir_fifa = st.checkbox("Consultar FIFA junto", value=True)
        with c3:
            renderizar_js = st.checkbox(
                "Renderizar JavaScript da FIFA com Selenium",
                value=False,
                help="Use somente se o HTML simples não trouxer dados. Requer Selenium e Chrome/Chromium no ambiente."
            )

        if st.button("Extrair base histórica da internet", type="primary", width="stretch", icon=":material/cloud_download:"):
            try:
                with st.spinner("Extraindo e padronizando base histórica..."):
                    extractor = InternetDataExtractor(timeout=45, renderizar_js=bool(renderizar_js))
                    result = extractor.extrair(ano_minimo=int(ano_minimo), incluir_fifa=bool(incluir_fifa))

                st.session_state.df_matches = result.matches.copy()
                st.session_state.internet_extras = result.extras
                st.session_state.internet_log = result.log
                st.session_state.fonte_base = "internet"
                sync_copa_live_state_from_extras(result.extras)
                reset_models()

                st.success("Base histórica extraída. Treine os modelos novamente antes de prever ou simular.")
                if st.session_state.equipes_copa_2026:
                    st.info(f"FIFA atualizada: {len(st.session_state.equipes_copa_2026)} seleções detectadas na última consulta.")
                    with st.expander("Ver seleções detectadas na FIFA"):
                        st.dataframe(pd.DataFrame({"Seleção": st.session_state.equipes_copa_2026}), width="stretch", hide_index=True)
                else:
                    st.warning("A consulta FIFA não retornou standings/grupos utilizáveis. A base histórica foi carregada normalmente.")

                st.dataframe(result.matches.head(30), width="stretch", hide_index=True)
                with st.expander("Ver log da extração"):
                    st.code("\n".join(result.log))
            except Exception as e:
                st.error(f"Erro na extração: {e}")

        st.divider()
        section("Atualizar FIFA agora", "globe")
        st.markdown(
            "Este botão consulta as páginas da FIFA no momento do clique e atualiza apenas seleções, grupos, standings, jogos e extras FIFA. "
            "Ele não substitui a base histórica de treino."
        )

        if st.button("Atualizar dados FIFA agora", type="secondary", width="stretch", icon=":material/sync:"):
            try:
                with st.spinner("Consultando páginas da FIFA agora..."):
                    extractor = InternetDataExtractor(timeout=45, renderizar_js=bool(renderizar_js))
                    result = extractor.extrair_fifa_ao_vivo(renderizar_js=bool(renderizar_js))

                extras_anteriores = st.session_state.internet_extras or {}
                st.session_state.internet_extras = {**extras_anteriores, **result.extras}
                st.session_state.internet_log = result.log
                sync_copa_live_state_from_extras(result.extras)
                recriar_simulador_copa_com_estado_fifa()

                if st.session_state.equipes_copa_2026:
                    st.success(
                        f"FIFA atualizada agora: {len(st.session_state.equipes_copa_2026)} seleções detectadas."
                    )
                    st.dataframe(pd.DataFrame({"Seleção": st.session_state.equipes_copa_2026}), width="stretch", hide_index=True)
                else:
                    st.warning(
                        "A FIFA foi consultada, mas não encontrei standings/grupos úteis no HTML/JSON retornado. "
                        "Ative Selenium se a página estiver vindo renderizada por JavaScript."
                    )

                with st.expander("Ver log da atualização FIFA"):
                    st.code("\n".join(result.log))
            except Exception as e:
                st.error(f"Erro ao atualizar FIFA: {e}")
    with tab_file:
        uploaded = st.file_uploader("Selecione .xlsx, .xls ou .csv", type=["xlsx", "xls", "csv"])
        if uploaded is not None:
            if st.button("Carregar planilha", type="primary", width="stretch", icon=":material/upload_file:"):
                suffix = Path(uploaded.name).suffix.lower()
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                        tmp.write(uploaded.getvalue())
                        tmp_path = tmp.name

                    loader = MatchDataLoader()
                    df = loader.carregar_arquivo(tmp_path)

                    st.session_state.df_matches = df.copy()
                    st.session_state.internet_extras = {}
                    st.session_state.internet_log = []
                    st.session_state.fonte_base = uploaded.name
                    st.session_state.equipes_copa_2026 = []
                    st.session_state.df_copa2026_classificacao = None
                    st.session_state.df_copa2026_classificados = None
                    reset_models()

                    st.success("Planilha carregada com sucesso.")
                    st.dataframe(df.head(30), width="stretch", hide_index=True)
                except Exception as e:
                    st.error(f"Erro ao carregar: {e}")


# ============================================================
# TREINAR MODELOS
# ============================================================

elif page == "Treinar modelos":
    hero()
    section("Treinamento", "brain")

    if not has_data():
        st.warning("Importe uma base primeiro.")
    else:
        c1, c2, c3 = st.columns([1, 1, 2])
        with c1:
            st.metric("Partidas", len(st.session_state.df_matches))
        with c2:
            st.metric("Equipes", len(set(st.session_state.df_matches["home_team"]).union(set(st.session_state.df_matches["away_team"]))))

        if st.button("Treinar todos os modelos", type="primary", width="stretch", icon=":material/rocket_launch:"):
            try:
                with st.spinner("Treinando Elo, xG, Poisson, Dixon-Coles, bivariada, ML e Ensemble..."):
                    treinar_modelos()
                st.success("Modelos treinados com sucesso.")
            except Exception as e:
                st.error(f"Erro no treinamento: {e}")

        if has_models():
            section("Ranking Elo", "medal")
            df_elo = st.session_state.df_elo.copy()
            c1, c2 = st.columns([1.25, 1])
            with c1:
                st.dataframe(df_elo.head(40), width="stretch", hide_index=True)
            with c2:
                top = df_elo.head(15).sort_values("Elo")
                fig = px.bar(top, x="Elo", y="Equipe", orientation="h", title="Top 15 por Elo")
                fig.update_layout(height=520)
                st.plotly_chart(style_plotly(fig), width="stretch")

            with st.expander("Ver forças ofensivas e defensivas"):
                st.dataframe(st.session_state.df_team_stats, width="stretch", hide_index=True)


# ============================================================
# PREVER PARTIDA
# ============================================================

elif page == "Prever partida":
    hero()
    section("Previsão de partida", "target")

    if not has_models():
        st.warning("Treine os modelos antes de prever.")
    else:
        all_teams = get_teams()
        copa_teams = get_copa_teams_for_model()
        usar_copa = False
        if copa_teams:
            usar_copa = st.toggle(
                "Modo Copa 2026: usar apenas seleções detectadas na última atualização FIFA",
                value=True,
                help="O histórico treina o modelo, mas a competição da Copa fica limitada às seleções da última consulta FIFA."
            )
        teams = copa_teams if usar_copa and copa_teams else all_teams

        if usar_copa and copa_teams:
            st.info(f"Modo Copa 2026 ativo: {len(copa_teams)} seleções disponíveis, vindas da última atualização FIFA.")

        c1, c2, c3 = st.columns([1, 1, 1])
        with c1:
            mandante = st.selectbox("Mandante / Time A", teams, index=0)
        with c2:
            default_idx = 1 if len(teams) > 1 else 0
            visitante = st.selectbox("Visitante / Time B", teams, index=default_idx)
        with c3:
            competicao = st.text_input("Competição", value="FIFA World Cup 2026" if usar_copa else "Não informado")

        if st.button("Calcular previsão", type="primary", width="stretch", icon=":material/bolt:"):
            if mandante == visitante:
                st.error("Escolha equipes diferentes.")
            else:
                try:
                    r = st.session_state.poisson_model.prever_partida(mandante, visitante)
                    r_ml = None
                    if st.session_state.ml_model and getattr(st.session_state.ml_model, "treinado", False):
                        try:
                            r_ml = st.session_state.ml_model.prever_partida(mandante, visitante, competicao=competicao)
                        except Exception:
                            r_ml = None
                    r_ens = st.session_state.ensemble_model.combinar(r, r_ml)

                    st.session_state.ultima_previsao = r
                    st.session_state.ultima_previsao_ml = r_ml
                    st.session_state.ultima_previsao_ensemble = r_ens
                    st.success("Previsão calculada.")
                except Exception as e:
                    st.error(f"Erro na previsão: {e}")

        r = st.session_state.ultima_previsao
        r_ml = st.session_state.ultima_previsao_ml
        r_ens = st.session_state.ultima_previsao_ensemble

        if r and r_ens:
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                metric_card("home", f"Vitória {r_ens['mandante']}", f"{r_ens['ensemble_prob_mandante']}%", "Ensemble final")
            with c2:
                metric_card("layers", "Empate", f"{r_ens['ensemble_prob_empate']}%", "Ensemble final")
            with c3:
                metric_card("arrow-up-right", f"Vitória {r_ens['visitante']}", f"{r_ens['ensemble_prob_visitante']}%", "Ensemble final")
            with c4:
                metric_card("target", "Placar provável", r_ens["placar_provavel"], f"Confiança: {r_ens['confianca']}")

            prob_df = pd.DataFrame({
                "Resultado": [f"Vitória {r_ens['mandante']}", "Empate", f"Vitória {r_ens['visitante']}"],
                "Probabilidade": [r_ens["ensemble_prob_mandante"], r_ens["ensemble_prob_empate"], r_ens["ensemble_prob_visitante"]],
            })
            lambda_df = pd.DataFrame({
                "Equipe": [r["mandante"], r["visitante"]],
                "xG esperado": [r["lambda_mandante"], r["lambda_visitante"]],
            })

            c1, c2 = st.columns(2)
            with c1:
                fig = px.bar(prob_df, x="Resultado", y="Probabilidade", title="Probabilidades 1X2 — Ensemble", text="Probabilidade")
                fig.update_layout(height=400, yaxis_title="%")
                st.plotly_chart(style_plotly(fig), width="stretch")
            with c2:
                fig2 = px.bar(lambda_df, x="Equipe", y="xG esperado", title="xG esperado / intensidade ofensiva", text="xG esperado")
                fig2.update_layout(height=400)
                st.plotly_chart(style_plotly(fig2), width="stretch")

            c1, c2 = st.columns(2)
            with c1:
                section("Parâmetros do modelo", "gauge")
                df_parametros = pd.DataFrame([
                    {"Métrica": "Rho Dixon-Coles", "Valor": r["rho_dixon_coles"]},
                    {"Métrica": "Lambda compartilhado", "Valor": r["lambda_compartilhado"]},
                    {"Métrica": "Peso estatístico", "Valor": r_ens["peso_estatistico"]},
                    {"Métrica": "Peso ML", "Valor": r_ens["peso_ml"]},
                    {"Métrica": "Over 2.5", "Valor": f"{r['prob_over_25']}%"},
                    {"Métrica": "Ambas marcam", "Valor": f"{r['prob_btts']}%"},
                ])
                df_parametros["Valor"] = df_parametros["Valor"].astype(str)
                st.dataframe(df_parametros, width="stretch", hide_index=True)
            with c2:
                section("Placares mais prováveis", "bar-chart")
                st.dataframe(pd.DataFrame(r["top_placares"]), width="stretch", hide_index=True)

            if r_ml:
                with st.expander("Ver previsão separada do Machine Learning"):
                    st.dataframe(pd.DataFrame([r_ml]), width="stretch", hide_index=True)


# ============================================================
# SIMULAÇÕES
# ============================================================

elif page == "Simulações":
    hero()
    section("Simulações Monte Carlo", "trophy")

    if not has_models():
        st.warning("Treine os modelos antes de simular.")
    else:
        tab_ko, tab_wc = st.tabs(["Mata-mata aleatório", "Copa 2026 aproximada"])

        with tab_ko:
            iteracoes = st.number_input("Simulações", min_value=100, max_value=30000, value=3000, step=500, key="it_ko")
            if st.button("Simular mata-mata", type="primary", width="stretch", icon=":material/trophy:"):
                try:
                    with st.spinner("Simulando torneios..."):
                        equipes_copa = get_copa_teams_for_model()
                        df_sim = st.session_state.simulador_campeao.simular_campeao(
                            iteracoes=int(iteracoes),
                            equipes=equipes_copa if equipes_copa else None,
                        )
                    st.session_state.df_simulacao_campeao = df_sim.copy()
                    st.success("Simulação concluída.")
                except Exception as e:
                    st.error(f"Erro na simulação: {e}")

            if st.session_state.df_simulacao_campeao is not None:
                df_sim = st.session_state.df_simulacao_campeao.head(25).copy()
                fig = px.bar(df_sim.sort_values("Probabilidade_Titulo_%"), x="Probabilidade_Titulo_%", y="Equipe", orientation="h", title="Probabilidade de título — mata-mata")
                fig.update_layout(height=650)
                st.plotly_chart(style_plotly(fig), width="stretch")
                st.dataframe(st.session_state.df_simulacao_campeao, width="stretch", hide_index=True)

        with tab_wc:
            iteracoes_copa = st.number_input("Simulações", min_value=100, max_value=20000, value=1000, step=500, key="it_wc")
            if st.button("Simular formato Copa 2026", type="primary", width="stretch", icon=":material/public:"):
                try:
                    with st.spinner("Simulando fase de grupos e mata-mata..."):
                        df_wc, ultima = st.session_state.simulador_copa.simular_campeao_formato_copa(iteracoes=int(iteracoes_copa))
                    st.session_state.df_simulacao_copa = df_wc.copy()
                    st.session_state.ultima_copa_simulada = ultima
                    st.success("Simulação da Copa concluída.")
                except Exception as e:
                    st.error(f"Erro na simulação da Copa: {e}")

            if st.session_state.df_simulacao_copa is not None:
                top = st.session_state.df_simulacao_copa.head(25).copy()
                fig = px.bar(top.sort_values("Prob_Titulo_%"), x="Prob_Titulo_%", y="Equipe", orientation="h", title="Probabilidade de título — Copa aproximada")
                fig.update_layout(height=650)
                st.plotly_chart(style_plotly(fig), width="stretch")
                st.dataframe(st.session_state.df_simulacao_copa, width="stretch", hide_index=True)

                if st.session_state.ultima_copa_simulada is not None:
                    with st.expander("Ver última Copa simulada"):
                        st.dataframe(st.session_state.ultima_copa_simulada["classificados"], width="stretch", hide_index=True)
                        st.dataframe(st.session_state.ultima_copa_simulada["mata_mata"], width="stretch", hide_index=True)


# ============================================================
# VALIDAÇÃO
# ============================================================

elif page == "Validação":
    hero()
    section("Validação temporal", "line-chart")

    if not has_data():
        st.warning("Importe uma base antes de validar.")
    else:
        c1, c2, c3 = st.columns([1, 1, 2])
        with c1:
            min_treino = st.number_input("Mínimo de treino", min_value=30, max_value=5000, value=80, step=10)
        with c2:
            max_teste = st.number_input("Máximo de testes", min_value=10, max_value=500, value=100, step=10)
        with c3:
            st.markdown("<div class='soft-text'>A validação treina com jogos antigos e prevê jogos posteriores, evitando vazamento de dados.</div>", unsafe_allow_html=True)

        if st.button("Rodar validação", type="primary", width="stretch", icon=":material/monitoring:"):
            try:
                with st.spinner("Rodando backtest temporal..."):
                    validator = ModelBacktester()
                    resumo, previsoes = validator.validar(
                        st.session_state.df_matches,
                        min_treino=int(min_treino),
                        max_partidas_teste=int(max_teste),
                        usar_ml=True,
                    )
                st.session_state.df_validacao_resumo = resumo.copy()
                st.session_state.df_validacao_previsoes = previsoes.copy()
                st.success("Validação concluída.")
            except Exception as e:
                st.error(f"Erro na validação: {e}")

        if st.session_state.df_validacao_resumo is not None:
            section("Métricas", "activity")
            st.dataframe(st.session_state.df_validacao_resumo, width="stretch", hide_index=True)

            resumo_long = st.session_state.df_validacao_resumo.melt(
                id_vars=["Modelo", "Partidas_Avaliadas"],
                value_vars=["Acuracia_%", "Log_Loss", "Brier_Score"],
                var_name="Métrica",
                value_name="Valor",
            )
            fig = px.bar(resumo_long, x="Modelo", y="Valor", color="Métrica", barmode="group", title="Comparação de métricas")
            fig.update_layout(height=440)
            st.plotly_chart(style_plotly(fig), width="stretch")

            with st.expander("Ver previsões testadas"):
                st.dataframe(st.session_state.df_validacao_previsoes, width="stretch", hide_index=True)


# ============================================================
# EXPORTAR
# ============================================================

elif page == "Exportar":
    hero()
    section("Exportação", "download")

    sheets: dict[str, pd.DataFrame] = {}

    if st.session_state.df_matches is not None:
        sheets["base_padronizada"] = st.session_state.df_matches
    if st.session_state.df_elo is not None:
        sheets["ranking_elo"] = st.session_state.df_elo
    if st.session_state.df_team_stats is not None:
        sheets["forcas_modelo"] = st.session_state.df_team_stats
    if st.session_state.ultima_previsao is not None:
        prev = {k: v for k, v in st.session_state.ultima_previsao.items() if k != "matriz"}
        sheets["ultima_previsao_stat"] = pd.DataFrame([prev])
        sheets["matriz_placares"] = st.session_state.ultima_previsao["matriz"]
    if st.session_state.ultima_previsao_ml is not None:
        sheets["ultima_previsao_ml"] = pd.DataFrame([st.session_state.ultima_previsao_ml])
    if st.session_state.ultima_previsao_ensemble is not None:
        sheets["ultima_previsao_ensemble"] = pd.DataFrame([st.session_state.ultima_previsao_ensemble])
    if st.session_state.df_simulacao_campeao is not None:
        sheets["simulacao_mata_mata"] = st.session_state.df_simulacao_campeao
    if st.session_state.df_simulacao_copa is not None:
        sheets["simulacao_copa_2026"] = st.session_state.df_simulacao_copa
    if st.session_state.df_validacao_resumo is not None:
        sheets["validacao_resumo"] = st.session_state.df_validacao_resumo
    if st.session_state.df_validacao_previsoes is not None:
        sheets["validacao_previsoes"] = st.session_state.df_validacao_previsoes

    for name, df_extra in (st.session_state.internet_extras or {}).items():
        if isinstance(df_extra, pd.DataFrame) and not df_extra.empty:
            sheets[f"extra_{name}"] = df_extra

    if not sheets:
        st.warning("Ainda não há dados para exportar.")
    else:
        st.write(f"Serão exportadas **{len(sheets)}** abas.")
        excel_bytes = dataframe_to_excel_bytes(sheets)
        st.download_button(
            label="Baixar relatório Excel",
            data=excel_bytes,
            file_name="relatorio_fifa2026_analytics.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            type="primary",
            width="stretch",
            icon=":material/download:",
        )

        with st.expander("Abas incluídas"):
            st.write(list(sheets.keys()))
