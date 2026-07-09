# Deploying TrackMyPDB Agent to Streamlit Cloud

This guide covers deploying either branch (sakeer = Claude, sakeer2 = OpenAI) to Streamlit Cloud for free public access.

## Prerequisites
- Your code pushed to GitHub (done)
- A Streamlit Cloud account (free): https://share.streamlit.io
- Your API key (Anthropic for sakeer, OpenAI for sakeer2)

## Step 1: Prepare the requirements file

Streamlit Cloud installs from requirements.txt. Make sure it includes everything.
For the OpenAI branch (sakeer2), it must include:

```
streamlit
requests
pandas
numpy
rdkit
tqdm
matplotlib
openai
```

For the Claude branch (sakeer), replace `openai` with `anthropic`.

## Step 2: Add packages.txt (system libraries)

RDKit's drawing needs some system libraries on Streamlit Cloud.
The packages.txt file in your repo root handles this. It contains:

```
libxrender1
libxext6
```

## Step 3: Deploy on Streamlit Cloud

1. Go to https://share.streamlit.io
2. Click "New app"
3. Select your repository: Standard-Seed-Corporation/TrackMyPDB-Open-Source
4. Choose the branch: sakeer2 (OpenAI) or sakeer (Claude)
5. Set "Main file path" to:
   - For sakeer2: agent/chat_interface_openai.py
   - For sakeer:  agent/chat_interface.py
6. Click "Deploy"

## Step 4: Add your API key as a secret

IMPORTANT: never commit your API key. Use Streamlit secrets instead.

1. In your deployed app, go to Settings -> Secrets
2. Add:

   For OpenAI (sakeer2):
   OPENAI_API_KEY = "sk-your-openai-key"

   For Claude (sakeer):
   ANTHROPIC_API_KEY = "sk-ant-your-key"

3. Save. The app restarts automatically.

## Step 5: Access your public app

Your app will be live at a URL like:
https://trackmypdb-agent.streamlit.app

Share it with your team!

## Notes on the code reading secrets

Streamlit Cloud exposes secrets via st.secrets and also as environment variables.
The chat interface reads os.getenv("OPENAI_API_KEY") / os.getenv("ANTHROPIC_API_KEY"),
which Streamlit Cloud populates from your secrets automatically. If you find it
isn't picked up, add this near the top of the chat interface:

```python
import os, streamlit as st
if "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
```

## Troubleshooting

- Build fails on rdkit: ensure packages.txt is present with the two libraries above.
- "Module not found": add the missing package to requirements.txt.
- App loads but tools fail: check that backend/ is committed and present in the repo.
- API errors: verify the secret name matches exactly and the key has credits.
