from __future__ import annotations

import tempfile
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from fifa2026_core import (
    InternetDataExtractor,
    MatchDataLoader,
    DynamicEloModel,
    BasePoissonModel,
    MLMatchOutcomeModel,
    EnsemblePredictionModel,
    MonteCarloChampionSimulator,
    WorldCupFormatSimulator,
    TournamentConfigLoader,
    ConfiguredWorldCupSimulator,
    ModelBacktester,
)


# ============================================================
# CONFIGURAÇÃO VISUAL
# ============================================================

st.set_page_config(
    page_title="FIFA 2026 Analytics",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)


CUSTOM_CSS = """
<style>
    :root {
        --bg-card: rgba(255,255,255,0.06);
        --bg-card-2: rgba(255,255,255,0.09);
        --border: rgba(255,255,255,0.12);
        --text-soft: rgba(255,255,255,0.72);
        --accent: #19C37D;
        --accent-2: #2E8CFF;
        --danger: #FF4B4B;
        --warning: #FFB020;
    }

    .block-container {
        padding-top: 1.6rem;
        padding-bottom: 2.5rem;
        max-width: 1380px;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #07111f 0%, #0b1628 55%, #08101d 100%);
        border-right: 1px solid rgba(255,255,255,0.08);
    }

    .hero {
        padding: 28px 30px;
        border-radius: 26px;
        background:
            radial-gradient(circle at 0% 0%, rgba(25,195,125,0.32), transparent 35%),
            radial-gradient(circle at 100% 20%, rgba(46,140,255,0.28), transparent 35%),
            linear-gradient(135deg, rgba(8,20,37,0.98), rgba(12,26,48,0.92));
        border: 1px solid rgba(255,255,255,0.12);
        box-shadow: 0 18px 55px rgba(0,0,0,0.28);
        margin-bottom: 22px;
    }

    .hero h1 {
        font-size: 2.45rem;
        letter-spacing: -0.04em;
        margin: 0 0 8px 0;
        line-height: 1.05;
    }

    .hero p {
        color: var(--text-soft);
        font-size: 1.02rem;
        max-width: 880px;
        margin: 0;
    }

    .chip-row {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin-top: 18px;
    }

    .chip {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 7px 11px;
        border-radius: 999px;
        background: rgba(255,255,255,0.08);
        border: 1px solid rgba(255,255,255,0.12);
        color: rgba(255,255,255,0.86);
        font-size: 0.84rem;
        white-space: nowrap;
    }

    .glass-card {
        padding: 18px 18px;
        border-radius: 22px;
        background: var(--bg-card);
        border: 1px solid var(--border);
        box-shadow: 0 10px 28px rgba(0,0,0,0.18);
        height: 100%;
    }

    .metric-card {
        padding: 18px 18px;
        border-radius: 20px;
        background: linear-gradient(145deg, rgba(255,255,255,0.08), rgba(255,255,255,0.035));
        border: 1px solid rgba(255,255,255,0.12);
        box-shadow: 0 12px 32px rgba(0,0,0,0.16);
    }

    .metric-icon {
        font-size: 1.35rem;
        width: 36px;
        height: 36px;
        display: flex;
        align-items: center;
        justify-content: center;
        border-radius: 12px;
        background: rgba(25,195,125,0.14);
        margin-bottom: 12px;
    }

    .metric-label {
        color: var(--text-soft);
        font-size: 0.84rem;
        margin-bottom: 5px;
    }

    .metric-value {
        font-size: 1.65rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        margin-bottom: 2px;
    }

    .metric-note {
        color: rgba(255,255,255,0.54);
        font-size: 0.78rem;
    }

    .section-title {
        display: flex;
        align-items: center;
        gap: 10px;
        font-size: 1.34rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        margin: 22px 0 10px 0;
    }

    .soft-text {
        color: var(--text-soft);
        font-size: 0.96rem;
    }

    .status-ok {
        color: #19C37D;
        font-weight: 700;
    }

    .status-warn {
        color: #FFB020;
        font-weight: 700;
    }

    div[data-testid="stMetric"] {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.10);
        padding: 14px 14px;
        border-radius: 18px;
    }

    .stButton > button {
        border-radius: 14px;
        border: 1px solid rgba(255,255,255,0.12);
        font-weight: 700;
        min-height: 42px;
    }

    .stDownloadButton > button {
        border-radius: 14px;
        font-weight: 700;
        min-height: 42px;
    }

    .dataframe {
        border-radius: 14px;
        overflow: hidden;
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
        "simulador_copa_configurada": None,
        "tournament_config": None,
        "ultima_previsao": None,
        "ultima_previsao_ml": None,
        "ultima_previsao_ensemble": None,
        "df_elo": None,
        "df_team_stats": None,
        "df_simulacao_campeao": None,
        "df_simulacao_copa": None,
        "df_simulacao_copa_configurada": None,
        "ultima_copa_simulada": None,
        "ultima_copa_configurada": None,
        "df_validacao_resumo": None,
        "df_validacao_previsoes": None,
        "fonte_base": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def has_data() -> bool:
    return st.session_state.df_matches is not None and not st.session_state.df_matches.empty


def has_models() -> bool:
    return st.session_state.poisson_model is not None and st.session_state.df_team_stats is not None


def metric_card(icon: str, label: str, value: str, note: str = "") -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-icon">{icon}</div>
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def hero() -> None:
    st.markdown(
        """
        <div class="hero">
            <h1>⚽ FIFA 2026 Analytics Engine</h1>
            <p>
                Plataforma profissional para previsão de partidas, Elo dinâmico, xG,
                Poisson bivariada, Dixon-Coles, Machine Learning, Ensemble e simulações Monte Carlo.
            </p>
            <div class="chip-row">
                <span class="chip">🧠 Ensemble estatístico + ML</span>
                <span class="chip">📈 Validação temporal</span>
                <span class="chip">🏆 Simulação de campeão</span>
                <span class="chip">🌐 Extração online</span>
                <span class="chip">📤 Exportação Excel</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section(title: str, icon: str = "") -> None:
    st.markdown(f"<div class='section-title'>{icon} {title}</div>", unsafe_allow_html=True)


def get_teams() -> list[str]:
    if not has_models():
        return []
    return sorted(st.session_state.df_team_stats["Equipe"].astype(str).tolist())


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
    st.session_state.simulador_copa = WorldCupFormatSimulator(
        poisson_model=poisson_model,
        ml_model=ml_model,
        ensemble_model=ensemble_model,
        random_state=42,
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

with st.sidebar:
    st.markdown("### ⚽ FIFA 2026")
    st.caption("Analytics Engine")

    page = st.radio(
        "Navegação",
        [
            "🏠 Dashboard",
            "🌐 Importar dados",
            "🧠 Treinar modelos",
            "🎯 Prever partida",
            "🏆 Simulações",
            "📈 Validação",
            "📤 Exportar",
        ],
        label_visibility="collapsed",
    )

    st.divider()
    if has_data():
        st.success("Base carregada")
        st.caption(st.session_state.fonte_base or "Fonte não informada")
    else:
        st.warning("Sem base")

    if has_models():
        st.success("Modelos treinados")
        if st.session_state.ml_model and getattr(st.session_state.ml_model, "treinado", False):
            st.caption("ML ativo")
        else:
            st.caption("ML indisponível ou base pequena")
    else:
        st.info("Modelos não treinados")


# ============================================================
# DASHBOARD
# ============================================================

if page == "🏠 Dashboard":
    hero()

    if not has_data():
        col1, col2, col3 = st.columns(3)
        with col1:
            metric_card("🌐", "Passo 1", "Importe dados", "Use internet ou planilha local")
        with col2:
            metric_card("🧠", "Passo 2", "Treine", "Elo, Poisson, ML e Ensemble")
        with col3:
            metric_card("🏆", "Passo 3", "Simule", "Partidas e campeão")

        st.info("Vá em **🌐 Importar dados** para começar.")
    else:
        df = st.session_state.df_matches.copy()
        teams = sorted(set(df["home_team"]).union(set(df["away_team"])))

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            metric_card("📅", "Partidas", f"{len(df):,}".replace(",", "."), "jogos válidos")
        with col2:
            metric_card("🌍", "Equipes", str(len(teams)), "seleções/equipes únicas")
        with col3:
            metric_card("🗓️", "Primeiro jogo", str(df["date"].min().date()), "início da série")
        with col4:
            metric_card("⏱️", "Último jogo", str(df["date"].max().date()), "fim da série")

        section("Visão geral da base", "📊")
        df_year = df.assign(ano=df["date"].dt.year).groupby("ano").size().reset_index(name="partidas")
        fig = px.line(
            df_year,
            x="ano",
            y="partidas",
            markers=True,
            title="Partidas por ano",
        )
        fig.update_layout(height=360, margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(fig, width='stretch')

        c1, c2 = st.columns(2)
        with c1:
            top_home = df["home_team"].value_counts().head(15).reset_index()
            top_home.columns = ["Equipe", "Jogos como mandante"]
            fig_home = px.bar(top_home, x="Jogos como mandante", y="Equipe", orientation="h", title="Top mandantes")
            fig_home.update_layout(height=430, yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig_home, width='stretch')
        with c2:
            st.dataframe(df.tail(20), width='stretch', hide_index=True)


# ============================================================
# IMPORTAR DADOS
# ============================================================

elif page == "🌐 Importar dados":
    hero()
    section("Importar base", "🌐")

    tab_net, tab_file = st.tabs(["🌐 Extração da internet", "📁 Planilha local"])

    with tab_net:
        st.markdown(
            "A extração online baixa uma base histórica internacional e tenta coletar tabelas públicas adicionais da FIFA."
        )
        c1, c2, c3 = st.columns([1, 1, 2])
        with c1:
            ano_minimo = st.number_input("Ano mínimo", min_value=1872, max_value=2030, value=2018, step=1)
        with c2:
            incluir_fifa = st.checkbox("Coletar páginas FIFA", value=True)

        if st.button("🌐 Extrair base da internet", type="primary", width='stretch'):
            try:
                with st.spinner("Extraindo e padronizando dados online..."):
                    extractor = InternetDataExtractor(timeout=35)
                    result = extractor.extrair(ano_minimo=int(ano_minimo), incluir_fifa=bool(incluir_fifa))

                st.session_state.df_matches = result.matches.copy()
                st.session_state.internet_extras = result.extras
                st.session_state.internet_log = result.log
                st.session_state.fonte_base = "internet"

                # limpa modelos antigos
                for key in [
                    "elo_model", "poisson_model", "ml_model", "ensemble_model", "simulador_campeao",
                    "simulador_copa", "df_elo", "df_team_stats", "ultima_previsao",
                    "ultima_previsao_ml", "ultima_previsao_ensemble",
                ]:
                    st.session_state[key] = None

                st.success("Base extraída com sucesso.")
                st.dataframe(result.matches.head(30), width='stretch', hide_index=True)
                with st.expander("Ver log da extração"):
                    st.code("\n".join(result.log))
            except Exception as e:
                st.error(f"Erro na extração: {e}")

    with tab_file:
        uploaded = st.file_uploader("Selecione .xlsx, .xls ou .csv", type=["xlsx", "xls", "csv"])
        if uploaded is not None:
            if st.button("📁 Carregar planilha", type="primary", width='stretch'):
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

                    for key in [
                        "elo_model", "poisson_model", "ml_model", "ensemble_model", "simulador_campeao",
                        "simulador_copa", "df_elo", "df_team_stats", "ultima_previsao",
                        "ultima_previsao_ml", "ultima_previsao_ensemble",
                    ]:
                        st.session_state[key] = None

                    st.success("Planilha carregada com sucesso.")
                    st.dataframe(df.head(30), width='stretch', hide_index=True)
                except Exception as e:
                    st.error(f"Erro ao carregar: {e}")


# ============================================================
# TREINAR MODELOS
# ============================================================

elif page == "🧠 Treinar modelos":
    hero()
    section("Treinamento", "🧠")

    if not has_data():
        st.warning("Importe uma base primeiro.")
    else:
        c1, c2, c3 = st.columns([1, 1, 2])
        with c1:
            st.metric("Partidas", len(st.session_state.df_matches))
        with c2:
            st.metric("Equipes", len(set(st.session_state.df_matches["home_team"]).union(set(st.session_state.df_matches["away_team"]))))

        if st.button("🚀 Treinar todos os modelos", type="primary", width='stretch'):
            try:
                with st.spinner("Treinando Elo, xG, Poisson, Dixon-Coles, bivariada, ML e Ensemble..."):
                    treinar_modelos()
                st.success("Modelos treinados com sucesso.")
            except Exception as e:
                st.error(f"Erro no treinamento: {e}")

        if has_models():
            section("Ranking Elo", "🏅")
            df_elo = st.session_state.df_elo.copy()
            c1, c2 = st.columns([1.25, 1])
            with c1:
                st.dataframe(df_elo.head(40), width='stretch', hide_index=True)
            with c2:
                top = df_elo.head(15).sort_values("Elo")
                fig = px.bar(top, x="Elo", y="Equipe", orientation="h", title="Top 15 por Elo")
                fig.update_layout(height=520)
                st.plotly_chart(fig, width='stretch')

            with st.expander("Ver forças ofensivas/defensivas"):
                st.dataframe(st.session_state.df_team_stats, width='stretch', hide_index=True)


# ============================================================
# PREVER PARTIDA
# ============================================================

elif page == "🎯 Prever partida":
    hero()
    section("Previsão de partida", "🎯")

    if not has_models():
        st.warning("Treine os modelos antes de prever.")
    else:
        teams = get_teams()
        c1, c2, c3 = st.columns([1, 1, 1])
        with c1:
            mandante = st.selectbox("Mandante / Time A", teams, index=0)
        with c2:
            default_idx = 1 if len(teams) > 1 else 0
            visitante = st.selectbox("Visitante / Time B", teams, index=default_idx)
        with c3:
            competicao = st.text_input("Competição", value="Não informado")

        if st.button("⚡ Calcular previsão", type="primary", width='stretch'):
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
                metric_card("🏠", f"Vitória {r_ens['mandante']}", f"{r_ens['ensemble_prob_mandante']}%", "Ensemble final")
            with c2:
                metric_card("🤝", "Empate", f"{r_ens['ensemble_prob_empate']}%", "Ensemble final")
            with c3:
                metric_card("🛫", f"Vitória {r_ens['visitante']}", f"{r_ens['ensemble_prob_visitante']}%", "Ensemble final")
            with c4:
                metric_card("🎯", "Placar provável", r_ens["placar_provavel"], f"Confiança: {r_ens['confianca']}")

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
                fig.update_layout(height=390, yaxis_title="%")
                st.plotly_chart(fig, width='stretch')
            with c2:
                fig2 = px.bar(lambda_df, x="Equipe", y="xG esperado", title="xG esperado / intensidade ofensiva", text="xG esperado")
                fig2.update_layout(height=390)
                st.plotly_chart(fig2, width='stretch')

            c1, c2 = st.columns(2)
            with c1:
                st.markdown("#### 📌 Parâmetros do modelo")
                st.dataframe(
                    pd.DataFrame([
                        {"Métrica": "Rho Dixon-Coles", "Valor": r["rho_dixon_coles"]},
                        {"Métrica": "Lambda compartilhado", "Valor": r["lambda_compartilhado"]},
                        {"Métrica": "Peso estatístico", "Valor": r_ens["peso_estatistico"]},
                        {"Métrica": "Peso ML", "Valor": r_ens["peso_ml"]},
                        {"Métrica": "Over 2.5", "Valor": f"{r['prob_over_25']}%"},
                        {"Métrica": "Ambas marcam", "Valor": f"{r['prob_btts']}%"},
                    ]),
                    width='stretch',
                    hide_index=True,
                )
            with c2:
                st.markdown("#### 🔢 Placares mais prováveis")
                st.dataframe(pd.DataFrame(r["top_placares"]), width='stretch', hide_index=True)

            if r_ml:
                with st.expander("Ver previsão separada do Machine Learning"):
                    st.dataframe(pd.DataFrame([r_ml]), width='stretch', hide_index=True)


# ============================================================
# SIMULAÇÕES
# ============================================================

elif page == "🏆 Simulações":
    hero()
    section("Simulações Monte Carlo", "🏆")

    if not has_models():
        st.warning("Treine os modelos antes de simular.")
    else:
        tab_ko, tab_wc = st.tabs(["🏆 Mata-mata aleatório", "🌍 Copa 2026 aproximada"])

        with tab_ko:
            iteracoes = st.number_input("Simulações", min_value=100, max_value=30000, value=3000, step=500, key="it_ko")
            if st.button("🏆 Simular mata-mata", type="primary", width='stretch'):
                try:
                    with st.spinner("Simulando torneios..."):
                        df_sim = st.session_state.simulador_campeao.simular_campeao(iteracoes=int(iteracoes))
                    st.session_state.df_simulacao_campeao = df_sim.copy()
                    st.success("Simulação concluída.")
                except Exception as e:
                    st.error(f"Erro na simulação: {e}")

            if st.session_state.df_simulacao_campeao is not None:
                df_sim = st.session_state.df_simulacao_campeao.head(25).copy()
                fig = px.bar(df_sim.sort_values("Probabilidade_Titulo_%"), x="Probabilidade_Titulo_%", y="Equipe", orientation="h", title="Probabilidade de título — mata-mata")
                fig.update_layout(height=650)
                st.plotly_chart(fig, width='stretch')
                st.dataframe(st.session_state.df_simulacao_campeao, width='stretch', hide_index=True)

        with tab_wc:
            iteracoes_copa = st.number_input("Simulações", min_value=100, max_value=20000, value=1000, step=500, key="it_wc")
            if st.button("🌍 Simular formato Copa 2026", type="primary", width='stretch'):
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
                st.plotly_chart(fig, width='stretch')
                st.dataframe(st.session_state.df_simulacao_copa, width='stretch', hide_index=True)

                if st.session_state.ultima_copa_simulada is not None:
                    with st.expander("Ver última Copa simulada"):
                        st.dataframe(st.session_state.ultima_copa_simulada["classificados"], width='stretch', hide_index=True)
                        st.dataframe(st.session_state.ultima_copa_simulada["mata_mata"], width='stretch', hide_index=True)


# ============================================================
# VALIDAÇÃO
# ============================================================

elif page == "📈 Validação":
    hero()
    section("Validação temporal", "📈")

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

        if st.button("📈 Rodar validação", type="primary", width='stretch'):
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
            st.markdown("#### Métricas")
            st.dataframe(st.session_state.df_validacao_resumo, width='stretch', hide_index=True)

            resumo_long = st.session_state.df_validacao_resumo.melt(
                id_vars=["Modelo", "Partidas_Avaliadas"],
                value_vars=["Acuracia_%", "Log_Loss", "Brier_Score"],
                var_name="Métrica",
                value_name="Valor",
            )
            fig = px.bar(resumo_long, x="Modelo", y="Valor", color="Métrica", barmode="group", title="Comparação de métricas")
            fig.update_layout(height=430)
            st.plotly_chart(fig, width='stretch')

            with st.expander("Ver previsões testadas"):
                st.dataframe(st.session_state.df_validacao_previsoes, width='stretch', hide_index=True)


# ============================================================
# EXPORTAR
# ============================================================

elif page == "📤 Exportar":
    hero()
    section("Exportação", "📤")

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
            label="📥 Baixar relatório Excel",
            data=excel_bytes,
            file_name="relatorio_fifa2026_analytics.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            type="primary",
            width='stretch',
        )

        with st.expander("Abas incluídas"):
            st.write(list(sheets.keys()))
