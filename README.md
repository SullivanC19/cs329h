# Whose score is it anyway?

## REQUIRED ENV VARIABLES

Make sure to set your HuggingFace and OpenAI tokens as follows...

```
export HUF_TOKEN=XXXXXXXXX
export OAI_TOKEN=XXXXXXXXX
```

## [VS Code + Docker]

To build and enter the Docker image in VS Code...

```Command Palette → “Dev Containers: Reopen in Container”```

This might take around 10 minutes to complete. Next...

```Install Microsoft's Jupyter plugin.```

Now you can run all notebooks manually to reproduce experimental results.


## [Apptainer]

Build the apptainer image...

```apptainer pull img.sif docker://ghcr.io/colinsullivan/cs329h:latest```

Then run the experiments.py script...

```apptainer exec img.sif python experiments.py```
