"""
Synthetic telemetry data generator using Spark-native expressions.

Generates realistic high-volume telemetry records with:
- Log-normal latency distribution (realistic long-tail)
- Weighted HTTP status codes (mostly successful)
- Beta-approximated relevance scores
- Multi-service, multi-region, time-distributed data
"""

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField, StringType, DoubleType,
    TimestampType, BooleanType, IntegerType
)

# Service definitions with relative traffic weights
SERVICES = [
    ("auth-service", 0.15),
    ("user-service", 0.18),
    ("search-service", 0.20),
    ("recommendation-engine", 0.12),
    ("payment-service", 0.08),
    ("notification-service", 0.07),
    ("inventory-service", 0.06),
    ("analytics-service", 0.05),
    ("media-service", 0.05),
    ("gateway-service", 0.04),
]

ENDPOINTS = {
    "auth-service": ["/login", "/logout", "/refresh-token", "/verify"],
    "user-service": ["/users", "/users/profile", "/users/settings", "/users/search"],
    "search-service": ["/search", "/search/suggest", "/search/autocomplete"],
    "recommendation-engine": ["/recommend", "/recommend/similar", "/recommend/trending"],
    "payment-service": ["/pay", "/refund", "/status", "/webhook"],
    "notification-service": ["/notify", "/subscribe", "/unsubscribe"],
    "inventory-service": ["/stock", "/reserve", "/release"],
    "analytics-service": ["/track", "/report", "/aggregate"],
    "media-service": ["/upload", "/download", "/transcode"],
    "gateway-service": ["/route", "/healthcheck"],
}

REGIONS = ["us-east-1", "us-west-2", "eu-west-1", "ap-south-1", "ap-northeast-1"]

# Latency profiles per service (log-normal mu, sigma)
LATENCY_PROFILES = {
    "auth-service": (3.5, 0.8),
    "user-service": (3.0, 0.6),
    "search-service": (4.0, 1.0),
    "recommendation-engine": (4.5, 1.2),
    "payment-service": (4.2, 0.9),
    "notification-service": (2.5, 0.5),
    "inventory-service": (3.2, 0.7),
    "analytics-service": (3.8, 0.8),
    "media-service": (5.0, 1.5),
    "gateway-service": (2.0, 0.4),
}


def generate_telemetry_data(
    spark: SparkSession,
    num_records: int = 1_000_000,
    seed: int = 42,
    time_span_hours: int = 168,
) -> DataFrame:
    """
    Generate synthetic telemetry data using Spark-native expressions.

    Args:
        spark: Active SparkSession
        num_records: Number of records to generate (default: 1M)
        seed: Random seed for reproducibility
        time_span_hours: Hours of data to generate (default: 168 = 7 days)

    Returns:
        DataFrame with telemetry records
    """
    # Base DataFrame with row IDs and random columns
    df = spark.range(0, num_records).withColumn("_rand", F.rand(seed))

    # Assign service based on weighted distribution using cumulative thresholds
    cumulative = 0.0
    service_expr = None
    for svc_name, weight in SERVICES:
        cumulative += weight
        condition = F.col("_rand") < F.lit(cumulative)
        if service_expr is None:
            service_expr = F.when(condition, F.lit(svc_name))
        else:
            service_expr = service_expr.when(condition, F.lit(svc_name))
    service_expr = service_expr.otherwise(F.lit(SERVICES[-1][0]))

    df = df.withColumn("service_name", service_expr)

    # Assign endpoints per service
    endpoint_expr = None
    for svc_name, endpoints in ENDPOINTS.items():
        num_eps = len(endpoints)
        ep_rand = (F.col("_rand") * 1000).cast("int") % num_eps
        ep_case = None
        for i, ep in enumerate(endpoints):
            if ep_case is None:
                ep_case = F.when(ep_rand == i, F.lit(ep))
            else:
                ep_case = ep_case.when(ep_rand == i, F.lit(ep))
        ep_case = ep_case.otherwise(F.lit(endpoints[0]))
        condition = F.col("service_name") == F.lit(svc_name)
        if endpoint_expr is None:
            endpoint_expr = F.when(condition, ep_case)
        else:
            endpoint_expr = endpoint_expr.when(condition, ep_case)
    endpoint_expr = endpoint_expr.otherwise(F.lit("/unknown"))

    df = df.withColumn("endpoint", endpoint_expr)

    # Generate latency using log-normal distribution per service
    rand_normal = F.randn(seed + 1)
    latency_expr = None
    for svc_name, (mu, sigma) in LATENCY_PROFILES.items():
        lat = F.exp(rand_normal * F.lit(sigma) + F.lit(mu))
        condition = F.col("service_name") == F.lit(svc_name)
        if latency_expr is None:
            latency_expr = F.when(condition, lat)
        else:
            latency_expr = latency_expr.when(condition, lat)
    latency_expr = latency_expr.otherwise(F.exp(rand_normal * 0.8 + 3.5))

    df = df.withColumn("latency_ms", F.round(latency_expr, 2))

    # Generate HTTP status codes with realistic distribution
    status_rand = F.rand(seed + 2)
    df = df.withColumn(
        "status_code",
        F.when(status_rand < 0.90, F.lit(200))
        .when(status_rand < 0.93, F.lit(201))
        .when(status_rand < 0.95, F.lit(400))
        .when(status_rand < 0.965, F.lit(404))
        .when(status_rand < 0.98, F.lit(500))
        .when(status_rand < 0.99, F.lit(502))
        .otherwise(F.lit(503)),
    )

    df = df.withColumn(
        "is_success", (F.col("status_code") < 400).cast(BooleanType())
    )

    # Relevance score: approximate beta distribution using Spark-native ops
    # Beta(2,5) approximation: mean ~0.28, right-skewed
    r1 = -F.log(F.lit(1.0) - F.rand(seed + 3))
    r2 = -F.log(F.lit(1.0) - F.rand(seed + 4))
    # Gamma(2) ~ sum of 2 exponentials
    g1 = r1 + (-F.log(F.lit(1.0) - F.rand(seed + 5)))
    # Gamma(5) ~ sum of 5 exponentials
    g2 = (
        r2
        + (-F.log(F.lit(1.0) - F.rand(seed + 6)))
        + (-F.log(F.lit(1.0) - F.rand(seed + 7)))
        + (-F.log(F.lit(1.0) - F.rand(seed + 8)))
        + (-F.log(F.lit(1.0) - F.rand(seed + 9)))
    )
    relevance_raw = g1 / (g1 + g2)
    df = df.withColumn(
        "relevance_score", F.round(F.greatest(F.lit(0.0), F.least(F.lit(1.0), relevance_raw)), 4)
    )

    # Timestamps: distributed over the time span
    epoch_base = F.unix_timestamp(F.current_timestamp()) - F.lit(time_span_hours * 3600)
    epoch_offset = (F.rand(seed + 10) * F.lit(time_span_hours * 3600)).cast("long")
    df = df.withColumn(
        "timestamp", F.from_unixtime(epoch_base + epoch_offset).cast(TimestampType())
    )

    # Region assignment
    region_rand = (F.rand(seed + 11) * len(REGIONS)).cast("int")
    region_expr = None
    for i, region in enumerate(REGIONS):
        if region_expr is None:
            region_expr = F.when(region_rand == i, F.lit(region))
        else:
            region_expr = region_expr.when(region_rand == i, F.lit(region))
    region_expr = region_expr.otherwise(F.lit(REGIONS[0]))
    df = df.withColumn("region", region_expr)

    # User ID: hash-based synthetic user IDs
    df = df.withColumn(
        "user_id",
        F.concat(F.lit("user_"), F.abs(F.hash(F.col("id"), F.rand(seed + 12))).cast("string")),
    )

    # Request ID
    df = df.withColumn("request_id", F.expr("uuid()"))

    # Select final columns and validate
    result = df.select(
        "request_id",
        "timestamp",
        "service_name",
        "endpoint",
        "latency_ms",
        "status_code",
        "is_success",
        "relevance_score",
        "region",
        "user_id",
    )

    # Data quality validation: ensure no nulls and valid ranges
    result = (
        result.where(F.col("latency_ms") > 0)
        .where(F.col("relevance_score").between(0, 1))
        .where(F.col("status_code").isNotNull())
    )

    return result
