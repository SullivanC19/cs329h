import os
import json
import numpy as np
import gc
import sys
import torch


# tokens

TOKEN_PATH = os.path.join("..", "tokens.json")
keys = json.load(open(TOKEN_PATH)) if os.path.exists(TOKEN_PATH) else {}

for k, prompt in [
    ("huggingface", "Enter HuggingFace token: "),
    ("openai", "Enter OpenAI API key: ")
]:
    if k not in keys or not keys[k]:
        val = input(prompt).strip()
        if val:
            keys[k] = val

with open(TOKEN_PATH, "w") as f:
    json.dump(keys, f, indent=2)

HUF_TOKEN = keys.get("huggingface", "")
OAI_TOKEN = keys.get("openai", "")


# embeddings

EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_BATCH_SIZE = 512
MAX_CHARS = 4096

EMBEDDING_DIR = os.path.join("..", "data", "embeddings")
if not os.path.exists(EMBEDDING_DIR):
    os.makedirs(EMBEDDING_DIR)

def load_embeddings(emb_dir=EMBEDDING_DIR):
    arrays = []
    for fname in sorted(os.listdir(emb_dir)):
        if fname.endswith(".npy"):
            arrays.append(np.load(os.path.join(emb_dir, fname)))
    return np.vstack(arrays)


# clusterings

ASSIGNMENTS_DIR = os.path.join("..", "data", "assignments")
if not os.path.exists(ASSIGNMENTS_DIR):
    os.makedirs(ASSIGNMENTS_DIR)

ALL_N_CLUSTERS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
ALL_ASSIGNMENTS_PATHS = [os.path.join(ASSIGNMENTS_DIR, f"assignments_{n}.npy") for n in ALL_N_CLUSTERS]
N_CLUSTERS = ALL_N_CLUSTERS[-1]
ASSIGNMENTS_PATH = ALL_ASSIGNMENTS_PATHS[-1]


# topicking

TOPICS_PATH = os.path.join("..", "data", "topics.jsonl")
TOPIC_MODEL = "gpt-4.1-mini"
MAX_CHARS = 100


# filtering

FILTER_PATH = os.path.join("..", "data", "cluster_filter.npy")


# judging

JUDGEMENTS_PATH = os.path.join("..", "data", "judgements.jsonl")
JUDGEMENT_MODEL = "gpt-4.1-mini"
N_JUDGEMENTS = 256

def safe_text(s: str) -> str:
    if len(s) <= 10000:
        return s
    return s[:5000] + "\n\n...[TRUNCATED]...\n\n" + s[-5000:]

def build_judgement_message(prompt: str, a: str, b: str) -> list[dict]:
    return [
        {"role": "system", "content":
            "You are a strict evaluator. "
            "First explain which response to the provided prompt is better (1-2 sentences). "
            "Then output EXACTLY one line: 'PREFERENCE: A' or 'PREFERENCE: B'."},
        {"role": "user", "content":
            f"Prompt:\n{safe_text(prompt)}\n\n"
            f"Response A:\n{safe_text(a)}\n\n"
            f"Response B:\n{safe_text(b)}"}
    ]

def elicit_pref(text: str) -> str:
    pref = "N/A"
    if "PREFERENCE: A" in text and "PREFERENCE: B" not in text:
        pref = "A"
    if "PREFERENCE: B" in text and "PREFERENCE: A" not in text:
        pref = "B"
    return pref
