# AutoKaggler Titanic Pipeline

Meet AutoKaggler – a fully automated agent that wakes up, secures the Kaggle Titanic dataset, engineers competition-ready features, trains multiple models, and hands you a validated `submission.csv` without requiring any manual babysitting. Everything from dataset acquisition to leaderboard-ready predictions is orchestrated via structured JSON I/O, keeping the run reproducible and auditable end-to-end.

## Quick demo: hands-off Titanic submissions

1. Provide a simple task description as JSON.
2. AutoKaggler resolves the data source (Kaggle → cached copy → bundled sample), fixes random seeds across numpy/LightGBM/XGBoost/sklearn, performs stratified CV, and logs diagnostics plus feature importances.
3. The agent emits an `AgentResult` JSON tagged with `#KGNINJA` and leaves a Kaggle-formatted `submission.csv` only after validation succeeds.

### Example `TaskInput`

```json
{
  "profile": "boosting",
  "data_source": "auto",
  "use_ensemble": true,
  "notes": "Hands-off leaderboard run"
}
