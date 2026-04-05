from pydantic import BaseModel

from extraction_testing.config import FeatureRule, MatchingConfig, RunConfig, TaskType
from extraction_testing.orchestrator import evaluate


class _SingleFeatureRecord(BaseModel):
    row_identifier: int
    topic_label: str


class _SingleEntityRecord(BaseModel):
    row_identifier: int
    headline_text: str
    author_name: str


class _MultiEntityRecord(BaseModel):
    contract_title: str
    contract_amount: float


def test_task_type_values_use_new_public_names():
    assert {task_type.value for task_type in TaskType} == {
        "SINGLE_FEATURE",
        "SINGLE_ENTITY",
        "MULTI_ENTITY",
    }


def test_single_feature_task_runs_end_to_end():
    run_config = RunConfig(
        task_type=TaskType.SINGLE_FEATURE,
        feature_rules=[
            FeatureRule(feature_name="topic_label", feature_type="category"),
        ],
        index_key_name="row_identifier",
    )

    result_bundle = evaluate(
        [_SingleFeatureRecord(row_identifier=1, topic_label="business")],
        [_SingleFeatureRecord(row_identifier=1, topic_label="business")],
        run_config,
    )

    assert result_bundle.row_accuracy_value == 1.0
    assert float(result_bundle.total_metrics_data_frame.iloc[0]["micro_accuracy"]) == 1.0


def test_single_entity_task_runs_end_to_end():
    run_config = RunConfig(
        task_type=TaskType.SINGLE_ENTITY,
        feature_rules=[
            FeatureRule(feature_name="headline_text", feature_type="text"),
            FeatureRule(feature_name="author_name", feature_type="text"),
        ],
        index_key_name="row_identifier",
    )

    result_bundle = evaluate(
        [_SingleEntityRecord(row_identifier=1, headline_text="Market Rally", author_name="Jane Doe")],
        [_SingleEntityRecord(row_identifier=1, headline_text="Market Rally", author_name="Jane Doe")],
        run_config,
    )

    assert result_bundle.row_accuracy_value == 1.0
    assert set(result_bundle.per_feature_metrics_data_frame["feature_name"]) == {"headline_text", "author_name"}


def test_multi_entity_task_runs_end_to_end():
    run_config = RunConfig(
        task_type=TaskType.MULTI_ENTITY,
        feature_rules=[
            FeatureRule(feature_name="contract_title", feature_type="text", weight_for_matching=1.0),
            FeatureRule(feature_name="contract_amount", feature_type="number", weight_for_matching=1.0),
        ],
        matching_config=MatchingConfig(matching_mode="exact", minimum_similarity_threshold=1.0),
    )

    result_bundle = evaluate(
        [
            _MultiEntityRecord(contract_title="Alpha", contract_amount=100.0),
            _MultiEntityRecord(contract_title="Beta", contract_amount=50.0),
        ],
        [
            _MultiEntityRecord(contract_title="Alpha", contract_amount=100.0),
            _MultiEntityRecord(contract_title="Beta", contract_amount=50.0),
        ],
        run_config,
    )

    assert result_bundle.row_accuracy_value == 1.0
    assert result_bundle.entity_detection_summary is not None
    assert result_bundle.entity_detection_summary["matched_count"] == 2
    assert result_bundle.entity_detection_summary["precision_entities"] == 1.0
