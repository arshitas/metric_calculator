"""
Telemetry Metrics Pipeline orchestration.

Manages the end-to-end pipeline:
1. SparkSession creation
2. Data generation
3. Metrics computation
4. Results display
"""

from pyspark.sql import SparkSession

from src.data_generator import generate_telemetry_data
from src.metrics import compute_all_metrics


def create_spark_session(app_name: str = "TelemetryMetricsPipeline") -> SparkSession:
    """Create and configure a SparkSession for the pipeline."""
    return (
        SparkSession.builder.appName(app_name)
        .config("spark.sql.shuffle.partitions", "200")
        .config("spark.driver.memory", "4g")
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
        .config("spark.ui.showConsoleProgress", "false")
        .getOrCreate()
    )


def display_metrics(metrics: dict) -> None:
    """Display all computed metrics to console."""
    separator = "=" * 80

    print(f"\n{separator}")
    print("  TELEMETRY METRICS PIPELINE - RESULTS DASHBOARD")
    print(separator)

    print(f"\n{'─' * 60}")
    print("  OVERALL LATENCY PERCENTILES")
    print(f"{'─' * 60}")
    metrics["overall_latency"].show(truncate=False)

    print(f"\n{'─' * 60}")
    print("  LATENCY PERCENTILES BY SERVICE")
    print(f"{'─' * 60}")
    metrics["per_service_latency"].show(20, truncate=False)

    print(f"\n{'─' * 60}")
    print("  FAILURE RATES BY SERVICE")
    print(f"{'─' * 60}")
    metrics["failure_rates_by_service"].show(20, truncate=False)

    print(f"\n{'─' * 60}")
    print("  FAILURE RATES BY ENDPOINT")
    print(f"{'─' * 60}")
    metrics["failure_rates_by_endpoint"].show(50, truncate=False)

    print(f"\n{'─' * 60}")
    print("  RELEVANCE SCORES BY SERVICE")
    print(f"{'─' * 60}")
    metrics["relevance_by_service"].show(20, truncate=False)

    print(f"\n{'─' * 60}")
    print("  METRICS BY REGION")
    print(f"{'─' * 60}")
    metrics["region_metrics"].show(20, truncate=False)

    print(f"\n{'─' * 60}")
    print("  HOURLY METRICS (sample: first 24 hours)")
    print(f"{'─' * 60}")
    metrics["hourly_metrics"].show(24, truncate=False)

    print(f"\n{separator}")
    print("  PIPELINE COMPLETE")
    print(f"{separator}\n")


def run_pipeline(num_records: int = 1_000_000, seed: int = 42) -> dict:
    """
    Execute the full telemetry metrics pipeline.

    Args:
        num_records: Number of telemetry records to generate
        seed: Random seed for reproducible data generation

    Returns:
        Dictionary of computed metric DataFrames
    """
    print(f"\n[Pipeline] Initializing Spark session...")
    spark = create_spark_session()
    spark.sparkContext.setLogLevel("WARN")

    print(f"[Pipeline] Generating {num_records:,} telemetry records...")
    telemetry_df = generate_telemetry_data(spark, num_records=num_records, seed=seed)

    # Repartition for parallel processing
    num_partitions = max(8, num_records // 100_000)
    telemetry_df = telemetry_df.repartition(num_partitions, "service_name", "region")

    print(f"[Pipeline] Data schema:")
    telemetry_df.printSchema()

    record_count = telemetry_df.count()
    print(f"[Pipeline] Generated {record_count:,} valid records")

    print(f"[Pipeline] Computing metrics...")
    metrics = compute_all_metrics(telemetry_df)

    display_metrics(metrics)

    return metrics
