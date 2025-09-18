# AutoKaggler Titanic Pipeline

AutoKaggler is a self-initialising agent that automates the Kaggle Titanic tutorial
workflow. It downloads (or falls back to a bundled sample of) the dataset, trains a
reproducible machine learning model, evaluates it with cross-validation, and emits a
submission file – all while respecting the JSON in/out contract defined in
[`AGENTS.md`](AGENTS.md).

## Features

* 🔁 **Self-initialisation** – runtime directories and environment defaults are
  created automatically.
* 🧠 **Profile aware** – switch between `fast` (logistic regression) and `power`
  (random forest) profiles.
* 🔐 **Safe fallbacks** – if Kaggle downloads are unavailable the agent uses an
  offline-friendly synthetic dataset.
* 🧾 **Structured I/O** – accepts `TaskInput` JSON and returns `AgentResult`
  JSON with the required `#KGNINJA` tag.
* 🧪 **Reproducible** – deterministic seeds, cached assets, and logged runs.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Usage

Run the agent by piping a JSON payload to `python -m autokaggler`:

```bash
echo '{"profile": "fast", "data_source": "auto"}' | python -m autokaggler
```

The command prints a JSON object describing the outcome and leaves artefacts in
`.agent_tmp/` and logs in `.agent_logs/`.

### Kaggle credentials

To work with the real competition data ensure the Kaggle API credentials are
available via environment variables or `~/.kaggle/kaggle.json`. The agent will
attempt to download the dataset on the first run and reuse the cached copy on
subsequent runs.

## Development

Install the optional test dependencies and run the test suite:

```bash
pip install -e .[test]
pytest
```

The repository also contains a synthetic dataset in `data/sample/` used for tests
and offline development.
