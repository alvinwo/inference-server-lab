# Your Lesson 1 workspace

Do not begin by running every test. Start with the complete, notebook-like experiment:

```bash
uv run python work/01-tensors-devices/guided_lab.py
```

Read the full chapter at `lessons/01-tensors-devices/README.md`. It explains the inference-server
picture, walks through each PyTorch idea, and tells you exactly what to implement in seven small
steps.

List those steps with:

```bash
uv run python -m course steps 01
```

Your first implementation checkpoint is:

```bash
uv run python -m course test 01 --step seed
```

Keep your implementation and engineering journal under `work/`. The course reset command archives
this directory instead of deleting it.
