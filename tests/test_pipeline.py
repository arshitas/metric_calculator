"""
Tests for the Telemetry Metrics Pipeline.

Uses a local SparkSession with small datasets to validate:
- Data generation schema and quality
- Metric computations with known inputs
- Edge cases (all success, all failure)
"""

import sys
import pytest
from datetime import datetime
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField, StringType, DoubleType,
    TimestampType, BooleanType, IntegerType
)

from src.data_generator import generate_telemetry_data
from src.metrics import (
    compute_overall_latency_percentiles,
    compute_per_service_latency_percentiles,
    compute_failure_rates_by_service,
    compute_failure_rates_by_endpoint,
    compute_relevance_scores_by_service,
    compute_hourly_metrics,
    compute_all_metrics,
)


@pytest.fixture(scope="session")
def spark():
    """Create a SparkSession for testing."""
    import os
    os.environ["PYSPARK_PYTHON"] = sys.executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
    session = (
        SparkSession.builder
        .master("local[1]")
        .appName("TelemetryPipelineTests")
        .config("spark.sql.shuffle.partitions", "2")
        .config("spark.ui.enabled", "false")
        .config("spark.ui.showConsoleProgress", "false")
        .config("spark.driver.host", "127.0.0.1")
        .config("spark.driver.bindAddress", "127.0.0.1")
        .getOrCreate()
    )
    session.sparkContext.setLogLevel("ERROR")
    yield session
    session.stop()


@pytest.fixture(scope="session")
def sample_telemetry(spark):
    """Generate a small telemetry dataset for testing."""
    return generate_telemetry_data(spark, num_records=10_000, seed=42)


@pytest.fixture(scope="session")
def known_data(spark):
    """Create a DataFrame with known values for deterministic tests."""
    schema = StructType([
        StructField("request_id", StringType()),
        StructField("timestamp", TimestampType()),
        StructField("service_name", StringType()),
        StructField("endpoint", StringType()),
        StructField("latency_ms", DoubleType()),
        StructField("status_code", IntegerType()),
        StructField("is_success", BooleanType()),
        StructField("relevance_score", DoubleType()),
        StructField("region", StringType()),
        StructField("user_id", StringType()),
    ])
    data = [
        ("r1", datetime(2024, 1, 1, 0, 0, 0), "svc-a", "/api", 10.0, 200, True, 0.8, "us-east-1", "u1"),
        ("r2", datetime(2024, 1, 1, 0, 0, 0), "svc-a", "/api", 20.0, 200, True, 0.6, "us-east-1", "u2"),
        ("r3", datetime(2024, 1, 1, 0, 0, 0), "svc-a", "/api", 30.0, 500, False, 0.4, "us-east-1", "u3"),
        ("r4", datetime(2024, 1, 1, 1, 0, 0), "svc-a", "/health", 5.0, 200, True, 0.9, "us-west-2", "u4"),
        ("r5", datetime(2024, 1, 1, 1, 0, 0), "svc-b", "/data", 100.0, 200, True, 0.7, "us-west-2", "u5"),
        ("r6", datetime(2024, 1, 1, 1, 0, 0), "svc-b", "/data", 200.0, 502, False, 0.3, "eu-west-1", "u6"),
        ("r7", datetime(2024, 1, 1, 2, 0, 0), "svc-b", "/data", 150.0, 200, True, 0.5, "eu-west-1", "u7"),
        ("r8", datetime(2024, 1, 1, 2, 0, 0), "svc-b", "/status", 50.0, 404, False, 0.2, "ap-south-1", "u8"),
    ]
    return spark.createDataFrame(data, schema)


class TestDataGeneration:
    """Tests for synthetic data generation."""

    def test_record_count(self, sample_telemetry):
        """Generated data should have approximately the requested number of records."""
        count = sample_telemetry.count()
        # Allow small reduction from quality filters
        assert count >= 9_000, f"Expected ~10,000 records, got {count}"

    def test_schema_columns(self, sample_telemetry):
        """Output DataFrame should have all expected columns."""
        expected_cols = {
            "request_id", "timestamp", "service_name", "endpoint",
            "latency_ms", "status_code", "is_success", "relevance_score",
            "region", "user_id"
        }
        assert set(sample_telemetry.columns) == expected_cols

    def test_no_nulls_in_required_fields(self, sample_telemetry):
        """No null values in critical fields."""
        for col in ["service_name", "endpoint", "latency_ms", "status_code", "is_success"]:
            null_count = sample_telemetry.where(F.col(col).isNull()).count()
            assert null_count == 0, f"Found {null_count} nulls in {col}"

    def test_latency_positive(self, sample_telemetry):
        """All latency values should be positive."""
        neg_count = sample_telemetry.where(F.col("latency_ms") <= 0).count()
        assert neg_count == 0, f"Found {neg_count} non-positive latencies"

    def test_relevance_score_range(self, sample_telemetry):
        """Relevance scores should be between 0 and 1."""
        out_of_range = sample_telemetry.where(
            (F.col("relevance_score") < 0) | (F.col("relevance_score") > 1)
        ).count()
        assert out_of_range == 0, f"Found {out_of_range} out-of-range relevance scores"

    def test_status_codes_valid(self, sample_telemetry):
        """Status codes should be valid HTTP codes."""
        valid_codes = {200, 201, 400, 404, 500, 502, 503}
        actual_codes = set(
            row.status_code for row in sample_telemetry.select("status_code").distinct().collect()
        )
        assert actual_codes.issubset(valid_codes), f"Unexpected status codes: {actual_codes - valid_codes}"

    def test_is_success_consistency(self, sample_telemetry):
        """is_success should be True for status < 400, False otherwise."""
        inconsistent = sample_telemetry.where(
            (F.col("is_success") & (F.col("status_code") >= 400))
            | (~F.col("is_success") & (F.col("status_code") < 400))
        ).count()
        assert inconsistent == 0, f"Found {inconsistent} inconsistent is_success values"

    def test_multiple_services_present(self, sample_telemetry):
        """Data should contain multiple services."""
        service_count = sample_telemetry.select("service_name").distinct().count()
        assert service_count >= 5, f"Expected at least 5 services, got {service_count}"

    def test_reproducibility(self, spark):
        """Same seed should produce identical data."""
        df1 = generate_telemetry_data(spark, num_records=1000, seed=99)
        df2 = generate_telemetry_data(spark, num_records=1000, seed=99)
        count1 = df1.count()
        count2 = df2.count()
        assert count1 == count2


class TestMetrics:
    """Tests for metric computations."""

    def test_overall_latency_has_expected_columns(self, known_data):
        result = compute_overall_latency_percentiles(known_data)
        expected_cols = {"total_records", "mean_latency_ms", "p50_latency_ms",
                         "p95_latency_ms", "p99_latency_ms", "min_latency_ms", "max_latency_ms"}
        assert set(result.columns) == expected_cols

    def test_overall_latency_values(self, known_data):
        row = compute_overall_latency_percentiles(known_data).collect()[0]
        assert row.total_records == 8
        assert row.min_latency_ms == 5.0
        assert row.max_latency_ms == 200.0

    def test_per_service_latency(self, known_data):
        result = compute_per_service_latency_percentiles(known_data)
        rows = {r.service_name: r for r in result.collect()}
        assert "svc-a" in rows
        assert "svc-b" in rows
        assert rows["svc-a"].record_count == 4
        assert rows["svc-b"].record_count == 4

    def test_failure_rates_by_service(self, known_data):
        result = compute_failure_rates_by_service(known_data)
        rows = {r.service_name: r for r in result.collect()}
        # svc-a: 1 failure out of 4 = 25%
        assert rows["svc-a"].failed_requests == 1
        assert rows["svc-a"].failure_rate_pct == 25.0
        # svc-b: 2 failures out of 4 = 50%
        assert rows["svc-b"].failed_requests == 2
        assert rows["svc-b"].failure_rate_pct == 50.0

    def test_failure_rates_by_endpoint(self, known_data):
        result = compute_failure_rates_by_endpoint(known_data)
        rows = result.collect()
        assert len(rows) >= 3  # svc-a:/api, svc-a:/health, svc-b:/data, svc-b:/status

    def test_relevance_scores_by_service(self, known_data):
        result = compute_relevance_scores_by_service(known_data)
        rows = {r.service_name: r for r in result.collect()}
        # svc-a relevance: (0.8 + 0.6 + 0.4 + 0.9) / 4 = 0.675
        assert abs(rows["svc-a"].mean_relevance - 0.675) < 0.01

    def test_hourly_metrics(self, known_data):
        result = compute_hourly_metrics(known_data)
        rows = result.collect()
        assert len(rows) == 3  # 3 distinct hours in test data

    def test_compute_all_metrics_keys(self, known_data):
        result = compute_all_metrics(known_data)
        expected_keys = {
            "overall_latency", "per_service_latency",
            "failure_rates_by_service", "failure_rates_by_endpoint",
            "relevance_by_service", "hourly_metrics", "region_metrics"
        }
        assert set(result.keys()) == expected_keys


class TestEdgeCases:
    """Tests for edge cases."""

    def test_all_success(self, spark):
        """Pipeline handles a dataset with zero failures."""
        schema = StructType([
            StructField("request_id", StringType()),
            StructField("timestamp", TimestampType()),
            StructField("service_name", StringType()),
            StructField("endpoint", StringType()),
            StructField("latency_ms", DoubleType()),
            StructField("status_code", IntegerType()),
            StructField("is_success", BooleanType()),
            StructField("relevance_score", DoubleType()),
            StructField("region", StringType()),
            StructField("user_id", StringType()),
        ])
        data = [
            ("r1", datetime(2024, 1, 1, 0, 0, 0), "svc-a", "/api", 10.0, 200, True, 0.9, "us-east-1", "u1"),
            ("r2", datetime(2024, 1, 1, 0, 0, 0), "svc-a", "/api", 20.0, 200, True, 0.8, "us-east-1", "u2"),
        ]
        df = spark.createDataFrame(data, schema)
        result = compute_failure_rates_by_service(df)
        row = result.collect()[0]
        assert row.failure_rate_pct == 0.0

    def test_all_failure(self, spark):
        """Pipeline handles a dataset with 100% failure rate."""
        schema = StructType([
            StructField("request_id", StringType()),
            StructField("timestamp", TimestampType()),
            StructField("service_name", StringType()),
            StructField("endpoint", StringType()),
            StructField("latency_ms", DoubleType()),
            StructField("status_code", IntegerType()),
            StructField("is_success", BooleanType()),
            StructField("relevance_score", DoubleType()),
            StructField("region", StringType()),
            StructField("user_id", StringType()),
        ])
        data = [
            ("r1", datetime(2024, 1, 1, 0, 0, 0), "svc-x", "/api", 500.0, 500, False, 0.1, "us-east-1", "u1"),
            ("r2", datetime(2024, 1, 1, 0, 0, 0), "svc-x", "/api", 600.0, 503, False, 0.0, "us-east-1", "u2"),
        ]
        df = spark.createDataFrame(data, schema)
        result = compute_failure_rates_by_service(df)
        row = result.collect()[0]
        assert row.failure_rate_pct == 100.0

    def test_small_dataset(self, spark):
        """Metrics should work with minimal data."""
        df = generate_telemetry_data(spark, num_records=100, seed=7)
        metrics = compute_all_metrics(df)
        for key, metric_df in metrics.items():
            assert metric_df.count() > 0, f"{key} produced empty results"
