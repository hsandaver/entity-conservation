# 🚀 Pushing to Hugging Face - Complete Guide

## Overview
Your Linked Data Explorer can be deployed to Hugging Face in two ways:
1. **Hugging Face Spaces** - Host your Streamlit app directly (recommended)
2. **Model Repository** - Push code as a regular repository

---

## Option 1: Deploy to Hugging Face Spaces (Recommended) ⭐

### What You'll Need
- Hugging Face account (free at https://huggingface.co)
- Hugging Face API token (from https://huggingface.co/settings/tokens)

### Step 1: Create a New Space

1. Go to https://huggingface.co/spaces
2. Click "Create new Space"
3. Fill in:
   - **Space name**: `entity-explorer-2` (or your preferred name)
   - **License**: Apache 2.0 (or your choice)
   - **Space SDK**: Select **"Streamlit"**
   - **Visibility**: Public or Private
4. Click "Create Space"

### Step 2: Clone the Space Repo

```bash
cd ~/Projects  # or your preferred directory
git clone https://huggingface.co/spaces/YOUR_USERNAME/entity-explorer-2
cd entity-explorer-2
```

### Step 3: Copy Your Project Files

```bash
# Copy all your project files to the cloned space
cp -r /Users/huwsandaver/Downloads/entity-explorer-2/* .

# You should now have:
# - app.py
# - src/
# - requirements.txt
# - etc.
```

### Step 4: Update `requirements.txt`

Ensure your `requirements.txt` includes all dependencies:

```txt
streamlit>=1.28.0
streamlit-components-v1>=0.1.0
networkx>=3.0
pyvis>=0.3.0
pandas>=1.5.0
plotly>=5.0.0
pydeck>=0.8.0
rdflib>=6.0.0
requests>=2.28.0
```

### Step 5: Create `README.md` for the Space

```markdown
---
title: Linked Data Explorer
emoji: 🔗
colorFrom: blue
colorTo: red
sdk: streamlit
sdk_version: 1.28.0
app_file: app.py
pinned: false
license: apache-2.0
---

# Linked Data Explorer

A research-grade network studio for RDF, SPARQL, embeddings, and semantic enrichment.

## Features
- Visualize and analyze linked data
- SPARQL query support
- RDF + JSON-LD support
- Embeddings and semantic analysis
- Interactive graph editing
- IIIF manifest support

## Quick Start

1. Upload your RDF/JSON-LD files
2. Configure visualization settings
3. Explore the graph interactively

For more information, see the [documentation](https://github.com/YOUR_GITHUB_USERNAME/entity-explorer-2).
```

### Step 6: Push to Hugging Face

```bash
git add .
git commit -m "Initial commit: Linked Data Explorer with aesthetic upgrades"
git push
```

### Step 7: Hugging Face Auto-Deploy

Hugging Face will automatically:
- Detect the `app.py` file
- Install dependencies from `requirements.txt`
- Launch your Streamlit app
- Provide a public URL

**Your Space URL**: `https://huggingface.co/spaces/YOUR_USERNAME/entity-explorer-2`

---

## Option 2: Push as Regular Repository

### Step 1: Initialize Git Locally

```bash
cd /Users/huwsandaver/Downloads/entity-explorer-2
git init
git add .
git commit -m "Initial commit: Linked Data Explorer with aesthetic upgrades"
```

### Step 2: Create Repository on Hugging Face

1. Go to https://huggingface.co/new
2. Fill in repository name and settings
3. Create the repository

### Step 3: Add Remote and Push

```bash
git remote add origin https://huggingface.co/YOUR_USERNAME/entity-explorer-2
git branch -M main
git push -u origin main
```

---

## Terminal Commands Quick Reference

### For Hugging Face Spaces (Recommended):

```bash
# 1. Clone the space
git clone https://huggingface.co/spaces/YOUR_USERNAME/entity-explorer-2
cd entity-explorer-2

# 2. Copy your files
cp -r /Users/huwsandaver/Downloads/entity-explorer-2/* .

# 3. Add README.md (create with the template above)
# 4. Push to deploy
git add .
git commit -m "Add Linked Data Explorer"
git push
```

### For Regular Repository:

```bash
# 1. Initialize git
cd /Users/huwsandaver/Downloads/entity-explorer-2
git init

# 2. Add all files
git add .

# 3. Make initial commit
git commit -m "Initial commit: Linked Data Explorer"

# 4. Add Hugging Face remote
git remote add origin https://huggingface.co/YOUR_USERNAME/entity-explorer-2

# 5. Push
git push -u origin main
```

---

## Pre-Push Checklist

Before pushing, ensure:

✅ **All files present:**
- `app.py` ✓
- `src/` directory ✓
- `requirements.txt` ✓
- `runtime.txt` (Python version) ✓
- `README.md` ✓
- Documentation files ✓

✅ **Configuration correct:**
- No hardcoded paths ✓
- No sensitive data in code ✓
- `requirements.txt` is up-to-date ✓
- App runs locally with `streamlit run app.py` ✓

✅ **Documentation complete:**
- `AESTHETIC_UPGRADES.md` included ✓
- `QUICK_REFERENCE.md` included ✓
- Upgrade docs in repo ✓

---

## After Deployment

### Monitor Your Space

1. Go to your Space URL
2. Check the "Build" tab for deployment status
3. View logs if there are issues
4. The app will be live once the build completes

### Share Your App

- Direct link: `https://huggingface.co/spaces/YOUR_USERNAME/entity-explorer-2`
- Embed code available on the Space page
- Share on social media

### Troubleshooting

If the app doesn't start:
1. Check logs in the "Build" tab
2. Ensure `app.py` exists in the root
3. Check `requirements.txt` for typos
4. Test locally: `streamlit run app.py`

---

## Environment Variables (if needed)

If your app needs secrets (API keys, etc.):
1. Go to Space Settings
2. Scroll to "Repository secrets"
3. Add your secrets
4. Access in code: `import os; api_key = os.getenv('API_KEY')`

---

## GPU Support (Optional)

For computationally intensive features:
1. Go to Space Settings
2. Upgrade to "GPU medium" or "GPU large"
3. Note: This requires premium Hugging Face account

---

## Next Steps

1. Create Hugging Face account if you don't have one
2. Choose deployment method (Spaces recommended)
3. Follow the steps above
4. Test your deployed app
5. Share the link with others!

---

## Resources

- **Hugging Face Spaces**: https://huggingface.co/spaces
- **Streamlit Documentation**: https://docs.streamlit.io
- **Hugging Face Hub**: https://github.com/huggingface/huggingface_hub
- **Create Secrets**: https://huggingface.co/spaces/FAQ

---

## Support

For issues with:
- **Hugging Face**: Check their documentation or community forum
- **Streamlit**: See https://docs.streamlit.io
- **Your app**: Test locally first, then check logs on Hugging Face

---

**Happy deploying! 🚀**
