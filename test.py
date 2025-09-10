from typing import List, Optional, Dict
from datetime import datetime, timedelta
from pydantic import BaseModel

from src.config import (
    TaskType,
    FeatureRule,
    MatchingConfig,
    ClassificationConfig,
    RunConfig,
)
from src.orchestrator import build_run_context, evaluate
from src.logger import RunLogger


# =========================
# Synthetic Pydantic Models
# =========================

class ContractRecord(BaseModel):
    """Contract entity with multiple features for entity extraction tests."""
    contract_title: str
    contract_amount: float
    contract_date: str  # ISO date string
    currency_code: str
    sector_name: Optional[str] = None  # optional metadata for slicing


class ArticleFeaturesRecord(BaseModel):
    """Article with multiple features for multi-feature extraction tests."""
    row_identifier: int
    headline_text: str
    author_name: str
    view_count: int
    publish_date: str  # ISO date string
    source_name: Optional[str] = None


class ArticleLabelRecord(BaseModel):
    """Article with single classification label."""
    row_identifier: int
    topic_label: str
    source_name: Optional[str] = None


# =========================
# Helpers for synthetic data
# =========================

def make_date_string(days_offset: int) -> str:
    """Return ISO date string with a day offset from fixed base for reproducibility."""
    base_date = datetime(2024, 6, 1)
    return (base_date + timedelta(days=days_offset)).date().isoformat()


def build_entity_extraction_gold() -> List[ContractRecord]:
    """Create a list of gold ContractRecord instances."""
    return [
        ContractRecord(contract_title="Supply Agreement Alpha",    contract_amount=100000.0, contract_date=make_date_string(0), currency_code="USD", sector_name="Energy"),
        ContractRecord(contract_title="Maintenance Contract Beta", contract_amount=50500.0,  contract_date=make_date_string(1), currency_code="EUR", sector_name="Manufacturing"),
        ContractRecord(contract_title="Consulting Deal Gamma",     contract_amount=75000.0,  contract_date=make_date_string(2), currency_code="USD", sector_name="Tech"),
        ContractRecord(contract_title="Distribution Pact Delta",   contract_amount=200000.0, contract_date=make_date_string(3), currency_code="NOK", sector_name="Retail"),
        ContractRecord(contract_title="Licensing Agreement Epsilon", contract_amount=150000.0, contract_date=make_date_string(4), currency_code="USD", sector_name="Energy"),
        
        # Number optional (gold None) — we will test predicted None and predicted value
        ContractRecord(contract_title="Optional Amount Expected Empty - Case Predicted None",  contract_amount=None, contract_date=make_date_string(6), currency_code="USD"),
        ContractRecord(contract_title="Optional Amount Expected Empty - Case Predicted Value", contract_amount=None, contract_date=make_date_string(7), currency_code="USD"),
        
        # Date optional (gold None) — we will test predicted None, predicted value, and predicted ""
        ContractRecord(contract_title="Optional Date Expected Empty - Case Predicted None",   contract_amount=12345.0, contract_date=None, currency_code="USD"),
        ContractRecord(contract_title="Optional Date Expected Empty - Case Predicted Value",  contract_amount=12346.0, contract_date=None, currency_code="USD"),
        ContractRecord(contract_title="Optional Date Expected Empty - Case Predicted EmptyString", contract_amount=12347.0, contract_date=None, currency_code="USD"),
    ]


def build_entity_extraction_predictions() -> List[ContractRecord]:
    """Create predicted ContractRecord instances with edge cases."""
    return [
        # Text normalization differences + currency alias
        ContractRecord(contract_title="supply agreement alpha!", contract_amount=100000.0, contract_date=make_date_string(0), currency_code="US Dollar", sector_name="Energy"),
        # Amount drift within tolerance, date +1 within tolerance
        ContractRecord(contract_title="Maintenance Contract Beta", contract_amount=50520.0, contract_date=make_date_string(2), currency_code="EUR", sector_name="Manufacturing"),
        # Fuzzy text spacing
        ContractRecord(contract_title="consulting deal  gamma", contract_amount=75000.0, contract_date=make_date_string(2), currency_code="USD", sector_name="Tech"),
        # Wrong currency + amount slightly off beyond tolerance
        ContractRecord(contract_title="Distribution Pact Delta", contract_amount=199500.0, contract_date=make_date_string(3), currency_code="USD", sector_name="Retail"),
        # Extra prediction not in gold
        ContractRecord(contract_title="Acquisition Accord Zeta", contract_amount=300000.0, contract_date=make_date_string(5), currency_code="USD", sector_name="Finance"),
        # Duplicate predictions competing for same gold
        ContractRecord(contract_title="Licensing Agreement Epsilon", contract_amount=150000.0, contract_date=make_date_string(4), currency_code="USD", sector_name="Energy"),
        ContractRecord(contract_title="Licensing Agreement Epsilon", contract_amount=149999.0, contract_date=make_date_string(4), currency_code="USD", sector_name="Energy"),
        
        # Number optional (gold None): predicted None → should be equal (if both-missing treated equal)
        ContractRecord(contract_title="Optional Amount Expected Empty - Case Predicted None",  contract_amount=None, contract_date=make_date_string(6), currency_code="USD"),
        # Number optional (gold None): predicted value → should be unequal
        ContractRecord(contract_title="Optional Amount Expected Empty - Case Predicted Value", contract_amount=42.0, contract_date=make_date_string(7), currency_code="USD"),

        # Date optional (gold None): predicted None → should be equal
        ContractRecord(contract_title="Optional Date Expected Empty - Case Predicted None",   contract_amount=12345.0, contract_date=None, currency_code="USD"),
        # Date optional (gold None): predicted value (a real date) → should be unequal
        ContractRecord(contract_title="Optional Date Expected Empty - Case Predicted Value",  contract_amount=12346.0, contract_date=make_date_string(9), currency_code="USD"),
        # Date optional (gold None): predicted empty string "" → parse to None, equal if both-missing treated equal
        ContractRecord(contract_title="Optional Date Expected Empty - Case Predicted EmptyString", contract_amount=12347.0, contract_date="", currency_code="USD"),
    ]


def build_multi_feature_gold() -> List[ArticleFeaturesRecord]:
    """Create indexed gold article feature records."""
    return [
        ArticleFeaturesRecord(row_identifier=1, headline_text="Breaking: Market Rally", author_name="John Smith", view_count=1000, publish_date=make_date_string(0), source_name="NewsNet"),
        ArticleFeaturesRecord(row_identifier=2, headline_text="Tech Giants Merge", author_name="Jane Doe", view_count=2500, publish_date=make_date_string(1), source_name="NewsNet"),
        ArticleFeaturesRecord(row_identifier=3, headline_text="Local Sports Win", author_name="A. Coach", view_count=400, publish_date=make_date_string(2), source_name="LocalDaily"),
        ArticleFeaturesRecord(row_identifier=4, headline_text="Economic Outlook", author_name="John Smith", view_count=800, publish_date=make_date_string(3), source_name="BizTimes"),
        ArticleFeaturesRecord(row_identifier=5, headline_text="Optional View Count None - Case Predicted None", author_name="John Smith", view_count=None, publish_date=make_date_string(10), source_name="NewsNet"),
        ArticleFeaturesRecord(row_identifier=6, headline_text="Optional View Count None - Case Predicted Value", author_name="Jane Doe",  view_count=None, publish_date=make_date_string(11), source_name="NewsNet"),
        ArticleFeaturesRecord(row_identifier=7, headline_text="Optional Publish Date None - Case Predicted None",  author_name="A. Coach", view_count=100, publish_date=None, source_name="LocalDaily"),
        ArticleFeaturesRecord(row_identifier=8, headline_text="Optional Publish Date None - Case Predicted Value", author_name="A. Coach", view_count=101, publish_date=None, source_name="LocalDaily"),
        ArticleFeaturesRecord(row_identifier=9, headline_text="Optional Publish Date None - Case Predicted EmptyString", author_name="A. Coach", view_count=102, publish_date=None, source_name="LocalDaily"),
    ]


def build_multi_feature_predictions() -> List[ArticleFeaturesRecord]:
    """Create predicted article feature records with common errors."""
    return [
        # Text punctuation/case differences, minor numeric error inside tolerance, date exact
        ArticleFeaturesRecord(row_identifier=1, headline_text="breaking market rally", author_name="J. Smith", view_count=1002, publish_date=make_date_string(0), source_name="NewsNet"),
        # Exact headline, author alias, date shifted by 1 day inside tolerance, numeric exact
        ArticleFeaturesRecord(row_identifier=2, headline_text="Tech Giants Merge", author_name="Jane D.", view_count=2500, publish_date=make_date_string(2), source_name="NewsNet"),
        # Wrong headline/author, numeric off beyond tolerance, date exact
        ArticleFeaturesRecord(row_identifier=3, headline_text="Regional Sports Loss", author_name="Coach A.", view_count=460, publish_date=make_date_string(2), source_name="LocalDaily"),
        # Exact headline/author/numeric, date off by 2 days (will fail if tolerance=1)
        ArticleFeaturesRecord(row_identifier=4, headline_text="Economic Outlook", author_name="John Smith", view_count=800, publish_date=make_date_string(5), source_name="BizTimes"),
        
        # view_count gold None: predicted None → equal
        ArticleFeaturesRecord(row_identifier=5, headline_text="Optional View Count None - Case Predicted None", author_name="John Smith", view_count=None, publish_date=make_date_string(10), source_name="NewsNet"),
        # view_count gold None: predicted value → unequal
        ArticleFeaturesRecord(row_identifier=6, headline_text="Optional View Count None - Case Predicted Value", author_name="Jane Doe",  view_count=777,  publish_date=make_date_string(11), source_name="NewsNet"),

        # publish_date gold None: predicted None → equal
        ArticleFeaturesRecord(row_identifier=7, headline_text="Optional Publish Date None - Case Predicted None",  author_name="A. Coach", view_count=100, publish_date=None, source_name="LocalDaily"),
        # publish_date gold None: predicted concrete date → unequal
        ArticleFeaturesRecord(row_identifier=8, headline_text="Optional Publish Date None - Case Predicted Value", author_name="A. Coach", view_count=101, publish_date=make_date_string(12), source_name="LocalDaily"),
        # publish_date gold None: predicted "" → parse to None, equal if both-missing treated equal
        ArticleFeaturesRecord(row_identifier=9, headline_text="Optional Publish Date None - Case Predicted EmptyString", author_name="A. Coach", view_count=102, publish_date="", source_name="LocalDaily"),
    ]


def build_classification_gold() -> List[ArticleLabelRecord]:
    """Create gold classification labels with class imbalance."""
    return [
        ArticleLabelRecord(row_identifier=1, topic_label="business", source_name="NewsNet"),
        ArticleLabelRecord(row_identifier=2, topic_label="tech",     source_name="NewsNet"),
        ArticleLabelRecord(row_identifier=3, topic_label="sports",   source_name="LocalDaily"),
        ArticleLabelRecord(row_identifier=4, topic_label="business", source_name="BizTimes"),
        ArticleLabelRecord(row_identifier=5, topic_label="business", source_name="BizTimes"),
        ArticleLabelRecord(row_identifier=6, topic_label="politics", source_name="NewsNet"),
    ]


def build_classification_predictions() -> List[ArticleLabelRecord]:
    """Create predicted classification labels with plausible confusions."""
    return [
        ArticleLabelRecord(row_identifier=1, topic_label="business",   source_name="NewsNet"),
        ArticleLabelRecord(row_identifier=2, topic_label="technology", source_name="NewsNet"),  # alias of 'tech'
        ArticleLabelRecord(row_identifier=3, topic_label="sports",     source_name="LocalDaily"),
        ArticleLabelRecord(row_identifier=4, topic_label="tech",       source_name="BizTimes"), # confusion
        ArticleLabelRecord(row_identifier=5, topic_label="business",   source_name="BizTimes"),
        ArticleLabelRecord(row_identifier=6, topic_label="politics",   source_name="NewsNet"),
    ]


# =========================
# Build configurations
# =========================

def build_entity_run_config() -> RunConfig:
    """Build RunConfig for entity extraction."""
    contract_feature_rules = [
        FeatureRule(feature_name="contract_title",  feature_type="text",    weight_for_matching=2.0),
        FeatureRule(feature_name="contract_amount", feature_type="number",  numeric_absolute_tolerance=100.0, numeric_rounding_digits=0, weight_for_matching=1.5),
        FeatureRule(feature_name="contract_date",   feature_type="date",    date_tolerance_days=1, weight_for_matching=1.0),
        FeatureRule(feature_name="currency_code",   feature_type="category", alias_map={"US Dollar": "USD"}, weight_for_matching=1.0),
    ]
    matching_config = MatchingConfig(matching_mode="weighted", minimum_similarity_threshold=0.6)
    return RunConfig(
        task_type=TaskType.ENTITY_EXTRACTION,
        feature_rules=contract_feature_rules,
        index_key_name=None,
        matching_config=matching_config,
        log_directory_path="./logs",
    )


def build_multi_run_config() -> RunConfig:
    """Build RunConfig for multi-feature extraction."""
    article_feature_rules = [
        FeatureRule(feature_name="headline_text", feature_type="text"),
        FeatureRule(feature_name="author_name",   feature_type="text", alias_map={"J. Smith": "John Smith", "Jane D.": "Jane Doe", "Coach A.": "A. Coach"}),
        FeatureRule(feature_name="view_count",    feature_type="number", numeric_absolute_tolerance=2, numeric_rounding_digits=0),
        FeatureRule(feature_name="publish_date",  feature_type="date", date_tolerance_days=1),
    ]
    return RunConfig(
        task_type=TaskType.MULTI_FEATURE,
        feature_rules=article_feature_rules,
        index_key_name="row_identifier",
        log_directory_path="./logs",
    )


def build_classification_run_config() -> RunConfig:
    """Build RunConfig for classification."""
    classification_feature_rules = [
        FeatureRule(feature_name="topic_label", feature_type="category", alias_map={"technology": "tech"})
    ]
    classification_config = ClassificationConfig(positive_label=None, average_strategy="macro")
    return RunConfig(
        task_type=TaskType.CLASSIFICATION,
        feature_rules=classification_feature_rules,
        index_key_name="row_identifier",
        classification_config=classification_config,
        log_directory_path="./logs",
    )


# =========================
# Main runner
# =========================

def main() -> None:
    """Run all three synthetic evaluations against the split testing framework."""
    # Build datasets
    entity_gold_records = build_entity_extraction_gold()
    entity_predicted_records = build_entity_extraction_predictions()

    multi_gold_records = build_multi_feature_gold()
    multi_predicted_records = build_multi_feature_predictions()

    class_gold_records = build_classification_gold()
    class_predicted_records = build_classification_predictions()

    # Build configurations
    entity_run_config = build_entity_run_config()
    multi_run_config = build_multi_run_config()
    classification_run_config = build_classification_run_config()

    # Run Entity Extraction
    entity_run_context = build_run_context(entity_run_config)
    entity_result_bundle = evaluate(entity_predicted_records, entity_gold_records, entity_run_config)
    entity_log_path = RunLogger(entity_run_config.log_directory_path).write_log(
        entity_run_context,
        entity_result_bundle,
        entity_run_config,
        note_message="Entity extraction synthetic run."
    )

    print("\n=== ENTITY EXTRACTION RESULTS ===")
    print("Total Metrics:")
    print(entity_result_bundle.total_metrics_data_frame.to_string(index=False))
    print("\nPer-Feature Metrics:")
    print(entity_result_bundle.per_feature_metrics_data_frame.to_string(index=False))
    print(f"\nRow Accuracy: {entity_result_bundle.row_accuracy_value:.4f}")
    if entity_result_bundle.entity_detection_summary:
        print("\nEntity Detection Summary:")
        for key, value in entity_result_bundle.entity_detection_summary.items():
            print(f"{key}: {value}")
    if entity_result_bundle.matched_pairs_data_frame is not None:
        print("\nMatched Pairs (first 10):")
        print(entity_result_bundle.matched_pairs_data_frame.head(10).to_string(index=False))
    print(f"\nLog written to: {entity_log_path}")

    # Run Multi-Feature Extraction
    multi_run_context = build_run_context(multi_run_config)
    multi_result_bundle = evaluate(multi_predicted_records, multi_gold_records, multi_run_config)
    multi_log_path = RunLogger(multi_run_config.log_directory_path).write_log(
        multi_run_context,
        multi_result_bundle,
        multi_run_config,
        note_message="Multi-feature synthetic run."
    )

    print("\n=== MULTI-FEATURE EXTRACTION RESULTS ===")
    print("Total Metrics:")
    print(multi_result_bundle.total_metrics_data_frame.to_string(index=False))
    print("\nPer-Feature Metrics:")
    print(multi_result_bundle.per_feature_metrics_data_frame.to_string(index=False))
    print(f"\nRow Accuracy: {multi_result_bundle.row_accuracy_value:.4f}")
    print(f"\nLog written to: {multi_log_path}")

    # Run Classification
    classification_run_context = build_run_context(classification_run_config)
    classification_result_bundle = evaluate(class_predicted_records, class_gold_records, classification_run_config)
    classification_log_path = RunLogger(classification_run_config.log_directory_path).write_log(
        classification_run_context,
        classification_result_bundle,
        classification_run_config,
        note_message="Classification synthetic run."
    )

    print("\n=== CLASSIFICATION RESULTS ===")
    print("Total Metrics:")
    print(classification_result_bundle.total_metrics_data_frame.to_string(index=False))
    print("\nPer-Feature Metrics:")
    print(classification_result_bundle.per_feature_metrics_data_frame.to_string(index=False))
    print(f"\nRow Accuracy: {classification_result_bundle.row_accuracy_value:.4f}")
    print(f"\nLog written to: {classification_log_path}")

    print("\nAll logs are available in ./logs")
    print("Note: For date '', ensure your framework treats unparsable dates as None;")
    print("      For numbers, Pydantic prevents '', so we test only None/value for numeric optional fields.")

if __name__ == "__main__":
    main()
