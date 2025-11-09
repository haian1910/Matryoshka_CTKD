#!/bin/bash

# === CONFIGURE GIT USER INFO ===
git config --global user.name "phunggiahuy159"
git config --global user.email "phunggiahuy.15092005@gmail.com"

# === CREATE AND SWITCH TO NEW BRANCH ===
BRANCH_NAME="eval"  # change this to your desired branch name
git checkout -b "$BRANCH_NAME"

# === ADD, COMMIT, AND PUSH ===
git add .
git commit -m "update all eval code"
git push -u origin "$BRANCH_NAME"

echo "✅ Code pushed successfully to branch '$BRANCH_NAME'"
