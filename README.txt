Pacote com o modelo usando métricas FIFA e tema Streamlit solicitado.

Estrutura correta no GitHub:

fifa2026_mae/
├── streamlit_app.py
├── fifa2026_core.py
├── requirements.txt
└── .streamlit/
    └── config.toml

O config.toml contém:

[theme]
base = "dark"
primaryColor = "#19C37D"
backgroundColor = "#07111F"
secondaryBackgroundColor = "#0E1B2E"
textColor = "#F7FAFC"
font = "sans serif"

[client]
showErrorDetails = "full"
