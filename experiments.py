import os
import subprocess

ALL_EXPERIMENTS = [
    "01-embeddings.ipynb",
    "02-clustering.ipynb",
    "03-topicking.ipynb",
    "04-filtering.ipynb",
    "05-judging.ipynb",
]

for experiment in ALL_EXPERIMENTS:
    path = os.path.join("experiments", experiment)
    print(f"Running {path}...")
    subprocess.run([
        "jupyter", "nbconvert",
        "--to", "notebook",
        "--execute",
        "--inplace",
        "--ExecutePreprocessor.timeout=-1",
        path
    ], check=True)

print("All experiments completed.")
