# 2026 World Cup Score Predictor

A Streamlit dashboard that predicts 2026 World Cup score probabilities with a Poisson model.

No football API key is required. The app uses built-in team ratings and static assumptions so it can run immediately on Streamlit Community Cloud.

## Run locally

```powershell
py -m pip install -r requirements.txt
py -m streamlit run app.py
```

## Deploy to Streamlit Community Cloud

1. Push this folder to a GitHub repository.
2. Go to https://share.streamlit.io/ and create a new app.
3. Select the repository and set `app.py` as the main file.
4. Deploy. No secrets are required.
