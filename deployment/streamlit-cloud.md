# Streamlit Cloud Deployment

Use this when you want the dashboard to be reachable from any computer with a public URL.

## Recommended setup

1. Push this repository to GitHub.
2. Open Streamlit Community Cloud.
3. Create a new app and point it at this repository.
4. Set the app entry file to `main.py`.
5. Add any model files or small demo artifacts that the dashboard needs.

## Important note

This project currently contains large datasets and model artifacts for local development. For a public deployment, keep the repo slim and only include the files the dashboard needs to start.

If the dashboard should load trained models in the cloud, place only the exported model bundle files in the repo or store them in external object storage and download them at startup.

## Local test command

```powershell
streamlit run main.py
```
