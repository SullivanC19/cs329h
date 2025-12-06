# Whose score is it anyway?

This repository requires a HuggingFace Hub Token and OpenAI key. You will be prompted to provide both.

The experiments take roughly 5h to run in total.

## [VS Code + Docker]

To build and enter the Docker image in VS Code...

```Command Palette → “Dev Containers: Reopen in Container”```

This might take around 10 minutes to complete. Next...

```Install Microsoft's Jupyter plugin.```

Now you can run all notebooks manually to reproduce experimental results.


## [Apptainer]

Build the apptainer image...

```apptainer pull img.sif docker://colinsullivan203/cs329h:latest```

Then run the experiments.py script...

```apptainer exec img.sif python experiments.py```
