#!/bin/bash

# Stage all changes
echo "Staging all changes..."
git add .

# Commit changes with the provided message
echo "Committing changes with message: '$commit_msg'..."
git commit -m "pushing"

# Push to GitHub (auto-detects current branch)
echo "Pushing to GitHub..."
git push

echo "Successfully pushed all changes to GitHub!"