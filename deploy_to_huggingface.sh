#!/bin/bash
# Hugging Face Deployment Setup Script
# This script helps you push your Linked Data Explorer to Hugging Face

set -e

echo "🚀 Linked Data Explorer - Hugging Face Deployment"
echo "=================================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if git is installed
if ! command -v git &> /dev/null; then
    echo "❌ Git is not installed. Please install git first."
    exit 1
fi

echo -e "${BLUE}Step 1: Initialize Git Repository${NC}"
if [ -d ".git" ]; then
    echo "✓ Git repository already initialized"
else
    git init
    echo "✓ Git repository initialized"
fi

echo ""
echo -e "${BLUE}Step 2: Configure Git${NC}"
read -p "Enter your name (for git config): " GIT_NAME
read -p "Enter your email (for git config): " GIT_EMAIL

git config user.name "$GIT_NAME"
git config user.email "$GIT_EMAIL"
echo "✓ Git configured"

echo ""
echo -e "${BLUE}Step 3: Stage Files${NC}"
git add .
echo "✓ All files staged"

echo ""
echo -e "${BLUE}Step 4: Create Initial Commit${NC}"
git commit -m "Initial commit: Linked Data Explorer with aesthetic upgrades" || echo "✓ Already committed"
echo "✓ Commit created"

echo ""
echo -e "${BLUE}Step 5: Set Remote Repository${NC}"
read -p "Enter your Hugging Face username: " HF_USERNAME
read -p "Enter repository name (default: entity-explorer-2): " HF_REPO
HF_REPO=${HF_REPO:-entity-explorer-2}

# Determine if pushing to Space or regular repo
echo ""
echo "Choose deployment type:"
echo "1) Hugging Face Space (Recommended for Streamlit apps)"
echo "2) Regular Repository"
read -p "Enter your choice (1 or 2): " DEPLOY_TYPE

if [ "$DEPLOY_TYPE" = "1" ]; then
    REPO_URL="https://huggingface.co/spaces/${HF_USERNAME}/${HF_REPO}"
    echo ""
    echo "⚠️  Before continuing:"
    echo "1. Go to https://huggingface.co/spaces"
    echo "2. Click 'Create new Space'"
    echo "3. Name it: $HF_REPO"
    echo "4. Select 'Streamlit' as SDK"
    echo "5. Create the Space"
    echo ""
    read -p "Press Enter when you've created the Space..."
else
    REPO_URL="https://huggingface.co/${HF_USERNAME}/${HF_REPO}"
    echo ""
    echo "⚠️  Before continuing:"
    echo "1. Go to https://huggingface.co/new"
    echo "2. Create a repository named: $HF_REPO"
    echo "3. Note: Public or Private (your choice)"
    echo ""
    read -p "Press Enter when you've created the Repository..."
fi

echo ""
echo "Setting remote to: $REPO_URL"
git remote remove origin 2>/dev/null || true
git remote add origin "$REPO_URL"
echo "✓ Remote added"

echo ""
echo -e "${BLUE}Step 6: Prepare for Push${NC}"
echo "Verifying files..."
if [ ! -f "app.py" ]; then
    echo "❌ app.py not found in current directory"
    exit 1
fi
if [ ! -f "requirements.txt" ]; then
    echo "❌ requirements.txt not found in current directory"
    exit 1
fi
if [ ! -f "README.md" ]; then
    echo "⚠️  README.md not found (optional but recommended)"
fi
echo "✓ Required files present"

echo ""
echo -e "${BLUE}Step 7: Push to Hugging Face${NC}"
echo "You may be prompted to enter your Hugging Face credentials."
echo "Use your Hugging Face username and access token."
echo ""
echo "To get your access token:"
echo "1. Go to https://huggingface.co/settings/tokens"
echo "2. Create a new token with 'write' access"
echo "3. Copy the token"
echo ""

# Try to push
if git push -u origin main 2>/dev/null || git push -u origin master 2>/dev/null; then
    echo ""
    echo -e "${GREEN}✓ Successfully pushed to Hugging Face!${NC}"
    echo ""
    echo "📍 Your repository is now live at:"
    echo "   $REPO_URL"
    echo ""
    if [ "$DEPLOY_TYPE" = "1" ]; then
        echo "🎉 Hugging Face is automatically deploying your Streamlit app!"
        echo "   Check the 'Build' tab to see deployment progress."
        echo ""
        echo "📊 Your Space will be available at:"
        echo "   https://huggingface.co/spaces/${HF_USERNAME}/${HF_REPO}"
    fi
else
    echo ""
    echo -e "${YELLOW}⚠️  Push encountered an issue.${NC}"
    echo ""
    echo "Try manually:"
    echo "  git push -u origin main"
    echo ""
    echo "Or if using master branch:"
    echo "  git push -u origin master"
    echo ""
    echo "Note: You may need to:"
    echo "1. Enter your Hugging Face username"
    echo "2. Use your access token as password"
fi

echo ""
echo "================================"
echo "Setup complete!"
echo "================================"
