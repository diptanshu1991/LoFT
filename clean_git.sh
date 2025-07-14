#!/bin/bash

echo " Cleaning up unwanted files..."

# Remove Mac system files
find . -name ".DS_Store" -print0 | xargs -0 git rm -f

# Remove .egg-info folder
git rm -r --cached loft.egg-info/

# Remove unwanted data files
git rm -f data/finetune_dataset_dolly_1000.json
git rm -f data/finetune_dataset_dolly_300.json
git rm -f data/finetune_dataset_v2.json

# Stage all cleaned files
git add .gitignore
git add Merged_models/README.md

echo " Cleanup done. Now commit the changes:"
echo "    git commit -m 'Clean repo: remove unnecessary files and add .gitignore rules'"


