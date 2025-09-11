# Extraction Testing Framework

A small, typed framework to evaluate pipeline outputs (entity extraction, multi-feature extraction, and classification) against gold data. It aligns predictions to gold, normalizes values (text/number/date/category), computes macro metrics per feature, aggregates totals, and writes timestamped logs.

## Install

```bash
pip install -e .
# requires: Python ≥ 3.9, pandas, numpy, pydantic
```
If you are using a fresh virtual environment, install dev extras for linting and tests:
```bash
pip install -e ".[dev]"
```

## Quickstart

```python
from extraction_testing.config import TaskType, FeatureRule, RunConfig, MatchingConfig
from extraction_testing.orchestrator import evaluate, build_run_context
from extraction_testing.logger import RunLogger

# Define which features to compare and how
feature_rules = [
    FeatureRule(feature_name="contract_title", feature_type="text", weight_for_matching=2.0),
    FeatureRule(feature_name="contract_amount", feature_type="number", numeric_absolute_tolerance=100.0),
    FeatureRule(feature_name="contract_date", feature_type="date", date_tolerance_days=1),
    FeatureRule(feature_name="currency_code", feature_type="category", alias_map={"US Dollar": "USD"}),
]

# Configure the run
run_config = RunConfig(
    task_type=TaskType.ENTITY_EXTRACTION,
    feature_rules=feature_rules,
    matching_config=MatchingConfig(matching_mode="weighted", minimum_similarity_threshold=0.6),
    log_directory_path="./logs",
)

# Evaluate (predicted_records and gold_records are lists of Pydantic models)
result_bundle = evaluate(predicted_records, gold_records, run_config)

# Write a human-readable log
logger = RunLogger(run_config.log_directory_path)
log_path = logger.write_log(build_run_context(run_config), result_bundle, run_config, "Example run")
print("Per-feature:\n", result_bundle.per_feature_metrics_data_frame)
print("Totals:\n", result_bundle.total_metrics_data_frame)
print("Log:", log_path)
```

## How it works

1. **Adapt:** Convert lists of Pydantic models (predicted/gold) to DataFrames.  
2. **Normalize:** Apply feature-specific normalization (text case/punctuation; numeric rounding & tolerances; date parsing & day window; category aliasing).  
3. **Align:**  
   - **Entity extraction:** Weighted similarity + threshold; one-to-one greedy matching with deterministic tie-breakers.  
   - **Indexed tasks:** Exact join on `index_key_name`.  
4. **Metrics:** Compute per-feature macro precision/recall/F1/specificity, micro accuracy, and row accuracy (all features correct). Entity extraction also reports entity presence precision/recall/F1.  
5. **Aggregate & Log:** Produce tidy DataFrames and a timestamped `.txt` summary.

## Folder structure (what each file contains)

```
src/
└─ extraction_testing/
   ├─ __init__.py              # Public API re-exports
   ├─ config.py                # TaskType, FeatureRule, MatchingConfig, ClassificationConfig, RunConfig
   ├─ models.py                # ResultBundle, RunContext, ConfusionCounts
   ├─ utils.py                 # adapters & predicates: normalize/alias/parse/missing/similarity
   ├─ aligners.py              # IndexAligner, EntityAligner, compute_pair_similarity
   ├─ metrics.py               # confusion counts, macro/micro metrics, row accuracy
   ├─ evaluators.py            # EntityExtractionEvaluator, MultiFeatureExtractionEvaluator, ClassificationEvaluator
   ├─ orchestrator.py          # build_run_context, evaluate
   └─ logger.py                # RunLogger
examples/
└─ test.py                     # runnable demo using synthetic data
tests/                         # pytest unit tests and fixtures
```

## Configuration tips

- **Optional fields:** If both predicted and gold are missing (`None`/unparsable), numbers and dates are treated as **equal**; text/category `None` is normalized to a sentinel for metrics accounting.  
- **Tolerances:** Use `numeric_absolute_tolerance`, `numeric_relative_tolerance`, `numeric_rounding_digits`, and `date_tolerance_days` to control equality.  
- **Alias maps:** Map synonyms (e.g., `"US Dollar" → "USD"`, `"technology" → "tech"`).  
- **Determinism:** Entity matching tie-breakers use a fixed seed (`random_tie_breaker_seed`) for reproducibility.

## Metrics (per-feature by default, macro-averaged over classes)

| Metric | What it is | What it measures | Practical example | High value means | Low value means |
|---|---|---|---|---|---|
| Precision | TP / (TP + FP) | Purity of predicted positives | Of 100 predicted “USD” currency codes, how many truly are “USD”? | Few false positives (you rarely predict a value when it is wrong) | Many false positives (over-predicting) |
| Recall | TP / (TP + FN) | Coverage of actual positives | Of 120 true “USD” cases, how many did you correctly predict as “USD”? | Few false negatives (you rarely miss true values) | Many false negatives (under-predicting) |
| F1 | Harmonic mean of Precision & Recall | Balance of precision and recall | When both precision and recall are decent, F1 is high; if one is low, F1 drops | Balanced performance | Imbalanced or weak precision/recall |
| Specificity | TN / (TN + FP) | True negative rate | For “USD” vs not-“USD”, how often do you correctly say “not-USD”? | Few false positives across the “rest” classes | Many false positives against the rest |
| Micro Accuracy | Correct / Total | Overall correctness at the label level | Across all rows and features, fraction of exact label matches | Overall label correctness is high | Many label mismatches |
| Row Accuracy | Rows with all features correct / Rows | Strict per-row correctness | A contract row counts “correct” only if **every** tested feature matches | Model gets entire rows right | One wrong feature spoils the row |

### Entity extraction: entity presence metrics

For entity detection (matching predicted entities to gold), we also report:

| Metric | What it is | What it measures | Practical example | High value means | Low value means |
|---|---|---|---|---|---|
| Precision (entities) | matched / predicted | Fraction of predicted entities that truly exist in gold | You predicted 10 entities; 9 match gold → 0.90 | Few extra entities | Many extra (spurious) entities |
| Recall (entities) | matched / gold | Fraction of gold entities you found | Gold has 12; you matched 9 → 0.75 | Few missed entities | Many missed gold entities |
| F1 (entities) | Harmonic mean of entity precision & recall | Balance of extra vs missed entities | High when you neither miss nor over-add | Balanced presence performance | Either many extra or many missed |

> Specificity does not apply to entity presence (there is no “true negative entity” set in a standard way), but per-feature specificity is still computed on aligned pairs.

## Visualization

The package includes an optional matplotlib-based visualization helper in `extraction_testing.visualization`. It produces simple, typed figures from a `ResultBundle` (no seaborn, one chart per figure) and can also save a small PNG report.

### Quick start

```python
from extraction_testing.visualization import (
plot_total_metrics_bar,
plot_per_feature_metrics_bar,
plot_entity_presence_summary,
save_all_charts_to_report,
)


# `result` is a ResultBundle from `orchestrator.evaluate(...)`
fig1 = plot_total_metrics_bar(result)
fig2 = plot_per_feature_metrics_bar(result, metric_name="f1")
fig3 = plot_entity_presence_summary(result) # no-op note if summary missing


paths = save_all_charts_to_report(result, "./_viz_out")
print(paths)
```

### Available plotting functions

| Function | Purpose |
|---|---|
| `plot_total_metrics_bar(result_bundle)` | Bar chart of total precision, recall, F1, specificity, micro accuracy, and row accuracy (if present). |
| `plot_per_feature_metrics_bar(result_bundle, metric_name="f1")` | Per-feature bar chart for the selected metric; deterministic sorting. |
| `plot_entity_presence_summary(result_bundle)` | Entity presence precision/recall/F1 (entity extraction only); shows a note if not available. |
| `plot_confusion_matrix_for_classification(gold, pred, class_names)` | Confusion matrix heatmap for classification labels. |
| `plot_metric_by_group(per_feature_df, group_col, metric_name)` | Optional helper to aggregate and visualize a metric by groups after you’ve joined group labels. |
| `save_all_charts_to_report(result_bundle, output_dir)` | Saves a timestamped folder with standard charts and returns a mapping of chart→file path. |

### Example script

Run the example after installing the package:

```python
python examples/visualize_results.py
```

It writes PNGs to ./_viz_out. For an interactive demo, you can also try the Streamlit app:

```python
streamlit run examples/streamlit_app.py
```

### Notes

- All figures use matplotlib defaults (no custom colors) and handle empty inputs by displaying a clear textual note.
- The functions do not mutate global matplotlib state and close figures when saving reports.
- Metric semantics match the testing framework (precision, recall, F1, specificity, micro accuracy); row accuracy is plotted if provided in the ResultBundle.


## Roadmap

- Hungarian assignment as an alternate aligner for entity extraction.  
- JSON structured logs alongside `.txt`.  
- Visualization helpers (cohort plots, confusion heatmaps).  
- Strict Pydantic v2 typing helpers and plugin hooks.

## License

MIT — see `LICENSE`.
