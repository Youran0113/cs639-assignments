#!/usr/bin/env bash
set -euo pipefail

mkdir -p data
if [ ! -f data/TinyStoriesV2-GPT4-train.txt ]; then
    echo "Downloading TinyStories training data..."
    wget -q -P data https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
fi
if [ ! -f data/TinyStoriesV2-GPT4-valid.txt ]; then
    echo "Downloading TinyStories validation data..."
    wget -q -P data https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt
fi

echo "Training..."
uv run tokenizer_hw.py

echo "Outputs saved to tokenizer_results/"