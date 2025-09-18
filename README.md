AutoKaggler Titanic Pipeline

Meet AutoKaggler – a fully automated agent that wakes up, secures the Kaggle Titanic dataset, engineers competition-ready features, trains multiple models, and hands you a validated submission.csv without requiring any manual babysitting. Everything from dataset acquisition to leaderboard-ready predictions is orchestrated via structured JSON I/O, keeping the run reproducible and auditable end-to-end.

Quick demo: hands-off Titanic submissions

Provide a simple task description as JSON.

AutoKaggler resolves the data source (Kaggle → cached copy → bundled sample), fixes random seeds across numpy/LightGBM/XGBoost/sklearn, performs stratified CV, and logs diagnostics plus feature importances.

The agent emits an AgentResult JSON tagged with #KGNINJA and leaves a Kaggle-formatted submission.csv only after validation succeeds.

Example TaskInput
{
  "profile": "boosting",
  "data_source": "auto",
  "use_ensemble": true,
  "notes": "Hands-off leaderboard run"
}

Example AgentResult
{
  "ok": true,
  "meta": {
    "profile": "boosting",
    "tags": ["#KGNINJA"],
    "log_file": ".agent_logs/run-20240101-120000.log"
  },
  "result": {
    "cv_mean_accuracy": 0.84,
    "cv_std": 0.02,
    "model_name": "Voting(LogReg+RF+LightGBM)",
    "submission_path": ".agent_tmp/submissions/submission-20240101-120000.csv",
    "data_source": "kaggle_cached",
    "feature_importances": [
      {"model": "boost", "feature": "num__FarePerPerson", "importance": 0.312},
      {"model": "rf", "feature": "num__Age", "importance": 0.201}
    ]
  }
}

Sample submission preview
PassengerId	Survived
892	0
893	1
894	0

Preview generated from the bundled sample dataset; real Kaggle runs will match the official test.csv row count.

Why it works

🔁 Self-initialisation & logging – runtime directories, structured logs, and profile selection happen automatically, ensuring every run is captured.

🧮 Composed feature engineering – modular steps derive passenger titles, family sizes, cabin presence, Age*Pclass, and fare-per-person interactions that plug straight into a reusable preprocessing stack.

🧠 Profile registry – toggle between fast (logistic regression), power (random forest), and boosting (LightGBM/XGBoost with soft-voting) profiles or inject your own via the registry.

🧪 Deterministic evaluation – Stratified K-Fold CV with fixed seeds reports mean and variance in logs and results, stabilising leaderboard expectations.

📊 Interpretability out of the box – feature importance summaries are logged and returned alongside model metadata for quick inspection.

🔐 Robust data fallbacks – Kaggle API download is attempted first, cached copies are reused on failure, and a bundled synthetic sample keeps offline runs unblocked.

Installation
python -m venv .venv
source .venv/bin/activate
pip install -e .

Usage

Run the agent by piping a JSON payload:

echo '{"profile": "fast", "data_source": "auto"}' | python -m autokaggler


Artifacts land in .agent_tmp/ (datasets & submissions) and .agent_logs/ (structured logs).
Set PROFILE=power or PROFILE=boosting in the environment to change defaults, or override via TaskInput JSON.

Profiles & ensembles
Profile	Description
fast	Logistic regression baseline – quick feedback for iteration.
power	Random forest powered by the engineered feature set.
boosting	LightGBM/XGBoost with optional soft voting ensemble.

Pass {"profile": "boosting", "use_ensemble": true} to enable ensemble mode.

Kaggle credentials

Provide Kaggle API credentials (via environment variables or ~/.kaggle/kaggle.json) to unlock real competition downloads. AutoKaggler reuses cached files on subsequent runs and gracefully degrades to the bundled sample data if the API is unreachable.

Development
pip install -e .[test]
pytest


The data/sample/ directory holds the synthetic Titanic dataset used for tests and offline experimentation.