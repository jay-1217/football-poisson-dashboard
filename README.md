# World Cup Score Predictor

A Streamlit dashboard that predicts football score probabilities with a Poisson model.

## Run locally

```powershell
py -m pip install -r requirements.txt
py -m streamlit run app.py
```

## Deploy to Streamlit Community Cloud

1. Push this folder to a GitHub repository.
2. Go to https://share.streamlit.io/ and create a new app.
3. Select the repository and set `app.py` as the main file.
4. In Advanced settings, add this secret:

```toml
API_FOOTBALL_KEY = "your_api_football_key"
```

Without the secret, the app still opens with bundled Argentina vs France demo data.
