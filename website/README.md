# Website (FastAPI)

Two-page site:
- `Info`: project idea + abstract + key plots + selected audio examples.
- `Live Demo`: amen break → tabla with a codebook replacement slider.

## Run

From the repo root:

```bash
python -m uvicorn website.main:app --reload --port 8000
```

Open:
- `http://127.0.0.1:8000/`

## (Optional) Regenerate audio examples

The info page expects example WAVs under `website/static/audio/examples/...`.
If they’re missing, run the generator:

```bash
python website/scripts/generate_examples.py
```

