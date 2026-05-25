"""
Telemetry metrics computation module.

Computes key metrics from telemetry data:
- Latency percentiles (P50, P95, P99) overall and per-service
- Failure rates per service and per endpoint
- Relevance score aggregations per service
- Time-windowed metrics (hourly)
"""

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window

PERCENTILE_ACCURACY = 10000


def compute_overall_latency_percentiles(df: DataFrame) -> DataFrame:
    """Compute P50, P95, P99 latency across all records."""
    return df.agg(
        F.count("*").alias("total_records"),
        F.round(F.mean("latency_ms"), 2).alias("mean_latency_ms"),
        F.round(F.percentile_approx("latency_ms", 0.50, PERCENTILE_ACCURACY), 2).alias("p50_latency_ms"),
        F.round(F.percentile_approx("latency_ms", 0.95, PERCENTILE_ACCURACY), 2).alias("p95_latency_ms"),
        F.round(F.percentile_approx("latency_ms", 0.99, PERCENTILE_ACCURACY), 2).alias("p99_latency_ms"),
        F.round(F.min("latency_ms"), 2).alias("min_latency_ms"),
        F.round(F.max("latency_ms"), 2).alias("max_latency_ms"),
    )


def compute_per_service_latency_percentiles(df: DataFrame) -> DataFrame:
    """Compute P50, P95, P99 latency grouped by service."""
    return (
        df.groupBy("service_name")
        .agg(
            F.count("*").alias("record_count"),
            F.round(F.mean("latency_ms"), 2).alias("mean_latency_ms"),
            F.round(F.percentile_approx("latency_ms", 0.50, PERCENTILE_ACCURACY), 2).alias("p50_latency_ms"),
            F.round(F.percentile_approx("latency_ms", 0.95, PERCENTILE_ACCURACY), 2).alias("p95_latency_ms"),
            F.round(F.percentile_approx("latency_ms", 0.99, PERCENTILE_ACCURACY), 2).alias("p99_latency_ms"),
        )
        .orderBy("service_name")
    )


def compute_failure_rates_by_service(df: DataFrame) -> DataFrame:
    """Compute failure rates per service."""
    return (
        df.groupBy("service_name")
        .agg(
            F.count("*").alias("total_requests"),
            F.sum(F.when(~F.col("is_success"), 1).otherwise(0)).alias("failed_requests"),
            F.round(
                F.sum(F.when(~F.col("is_success"), 1).otherwise(0)) / F.count("*") * 100, 2
            ).alias("failure_rate_pct"),
        )
        .orderBy(F.desc("failure_rate_pct"))
    )


def compute_failure_rates_by_endpoint(df: DataFrame) -> DataFrame:
    """Compute failure rates per service-endpoint combination."""
    return (
        df.groupBy("service_name", "endpoint")
        .agg(
            F.count("*").alias("total_requests"),
            F.sum(F.when(~F.col("is_success"), 1).otherwise(0)).alias("failed_requests"),
            F.round(
                F.sum(F.when(~F.col("is_success"), 1).otherwise(0)) / F.count("*") * 100, 2
            ).alias("failure_rate_pct"),
        )
        .orderBy("service_name", "endpoint")
    )


def compute_relevance_scores_by_service(df: DataFrame) -> DataFrame:
    """Compute relevance score statistics per service."""
    return (
        df.groupBy("service_name")
        .agg(
            F.round(F.mean("relevance_score"), 4).alias("mean_relevance"),
            F.round(F.percentile_approx("relevance_score", 0.50, PERCENTILE_ACCURACY), 4).alias(
                "median_relevance"
            ),
            F.round(F.stddev("relevance_score"), 4).alias("stddev_relevance"),
            F.round(F.min("relevance_score"), 4).alias("min_relevance"),
            F.round(F.max("relevance_score"), 4).alias("max_relevance"),
        )
        .orderBy("service_name")
    )


def compute_hourly_metrics(df: DataFrame) -> DataFrame:
    """Compute time-windowed (hourly) metrics."""
    return (
        df.withColumn("hour", F.date_trunc("hour", "timestamp"))
        .groupBy("hour")
        .agg(
            F.count("*").alias("request_count"),
            F.round(F.mean("latency_ms"), 2).alias("mean_latency_ms"),
            F.round(F.percentile_approx("latency_ms", 0.95, PERCENTILE_ACCURACY), 2).alias("p95_latency_ms"),
            F.round(
                F.sum(F.when(~F.col("is_success"), 1).otherwise(0)) / F.count("*") * 100, 2
            ).alias("failure_rate_pct"),
            F.round(F.mean("relevance_score"), 4).alias("mean_relevance"),
        )
        .orderBy("hour")
    )


def compute_region_metrics(df: DataFrame) -> DataFrame:
    """Compute metrics per deployment region."""
    return (
        df.groupBy("region")
        .agg(
            F.count("*").alias("request_count"),
            F.round(F.mean("latency_ms"), 2).alias("mean_latency_ms"),
            F.round(F.percentile_approx("latency_ms", 0.95, PERCENTILE_ACCURACY), 2).alias("p95_latency_ms"),
            F.round(F.percentile_approx("latency_ms", 0.99, PERCENTILE_ACCURACY), 2).alias("p99_latency_ms"),
            F.round(
                F.sum(F.when(~F.col("is_success"), 1).otherwise(0)) / F.count("*") * 100, 2
            ).alias("failure_rate_pct"),
        )
        .orderBy("region")
    )


def compute_all_metrics(df: DataFrame) -> dict:
    """
    Compute all metrics and return as a dictionary of DataFrames.

    Returns:
        Dictionary mapping metric names to DataFrames
    """
    # Cache the input DataFrame since it's used multiple times
    df.cache()

    metrics = {
        "overall_latency": compute_overall_latency_percentiles(df),
        "per_service_latency": compute_per_service_latency_percentiles(df),
        "failure_rates_by_service": compute_failure_rates_by_service(df),
        "failure_rates_by_endpoint": compute_failure_rates_by_endpoint(df),
        "relevance_by_service": compute_relevance_scores_by_service(df),
        "hourly_metrics": compute_hourly_metrics(df),
        "region_metrics": compute_region_metrics(df),
    }

    return metrics
