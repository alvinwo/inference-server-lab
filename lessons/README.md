# Lesson snapshot rules

Each published lesson is a self-contained checkpoint with `starter/`, `solution/`, `tests/`,
`checkpoint.toml`, `benchmark.py`, and learner documentation. Tests must run without a network
connection. Starter code must never import or reveal solution code.

Copy `_template` when authoring a lesson. Keep the public interface stable between starter and
solution, name every intended starter failure in `checkpoint.toml`, and run:

```bash
uv run python scripts/verify_checkpoints.py
```

Learner work belongs under the gitignored `work/` directory. Course reset archives it under
`work/.trash/`; lesson tools must not delete learner files.
