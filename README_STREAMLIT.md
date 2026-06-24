# FIFA 2026 Analytics Engine — Streamlit

## Arquivos do app

- `streamlit_app.py`: interface profissional em Streamlit.
- `fifa2026_core.py`: motor estatístico sem CustomTkinter.
- `.streamlit/config.toml`: tema visual escuro.
- `requirements_streamlit_fifa2026.txt`: dependências para deploy.

## Rodar localmente

```bash
pip install -r requirements_streamlit_fifa2026.txt
streamlit run streamlit_app.py
```

## Deploy no Streamlit Community Cloud

1. Suba estes arquivos para um repositório GitHub.
2. No Streamlit Community Cloud, selecione o repositório.
3. Main file path: `streamlit_app.py`.
4. Requirements file: `requirements_streamlit_fifa2026.txt`.

## Fluxo de uso

1. Importar dados pela internet ou planilha local.
2. Treinar modelos.
3. Prever partida.
4. Simular campeão.
5. Validar modelo.
6. Exportar Excel.
