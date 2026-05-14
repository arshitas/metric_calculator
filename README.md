# Telemetry Metrics Pipeline

A scalable telemetry processing pipeline built with **PySpark** that generates and analyzes large-scale log data, computing key performance metrics for distributed microservice architectures.

## Features

- **High-Volume Data Generation**: Generates 1M+ synthetic telemetry records using Spark-native expressions (no Python UDFs) for maximum throughput
- **Latency Percentiles**: Computes P50, P95, and P99 latency metrics overall, per-service, and per-region
- **Failure Rate Analysis**: Calculates failure rates by service and by individual endpoint
- **Relevance Scoring**: Aggregates relevance scores (mean, median, std deviation) per service
- **Time-Windowed Metrics**: Hourly breakdown of latency, failure rates, and relevance
- **Reproducible Results**: Configurable random seed for deterministic data generation

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Data Generator  │───▶│  PySpark Engine   │───▶│ Metrics Output  │
│  (1M+ records)   │    │  (distributed)    │    │ (dashboard)     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

### Simulated Services
The pipeline simulates traffic for 10 microservices with realistic characteristics:
1. **auth-service**
2. **user-service**
3. **search-service**
4. **recommendation-engine**
5. **payment-service**
6. **notification-service**
7. **inventory-service**
8. **analytics-service**
9. **media-service**
10. **gateway-service**

### Data Model
Each telemetry record contains:
| Field | Type | Description |
|-------|------|-------------|
| `request_id` | UUID | Unique request identifier |
| `timestamp` | Timestamp | Request time (distributed over 7 days) |
| `service_name` | String | Originating microservice |
| `endpoint` | String | API endpoint path |
| `latency_ms` | Double | Response latency (log-normal distribution) |
| `status_code` | Integer | HTTP status code (weighted realistic distribution) |
| `is_success` | Boolean | Whether status < 400 |
| `relevance_score` | Double | Response relevance 0-1 (beta distribution) |
| `region` | String | Deployment region |
| `user_id` | String | Hashed user identifier |

## Prerequisites

- **Python** 3.9+
- **Java** 17+ (required by PySpark — install via `winget install Microsoft.OpenJDK.17`)
- **pip** for dependency management

## Setup

```bash
# Install dependencies
pip install -r requirements.txt
```

## Usage

```bash
# Run with default 1M records
python main.py

# Custom record count
python main.py --records 5000000

# Custom random seed for reproducibility
python main.py --seed 123
```

## Running Tests

```bash
pytest tests/ -v
```

## Project Structure

```
├── main.py                  # CLI entry point
├── src/
│   ├── data_generator.py    # Spark-native synthetic data generation
│   ├── metrics.py           # Metric computation functions
│   └── pipeline.py          # Pipeline orchestration
├── tests/
│   └── test_pipeline.py     # Unit tests with known-value validation
├── requirements.txt         # Python dependencies
└── README.md
```

## Metrics Computed

### Latency Percentiles
- **P50** (median): Typical response time
- **P95**: 95th percentile — captures most users' experience
- **P99**: 99th percentile — tail latency for SLA monitoring

### Failure Rates
- Per-service error percentage
- Per-endpoint error percentage
- HTTP status code breakdown (4xx vs 5xx)

### Relevance Scores
- Mean, median, and standard deviation per service
- Useful for search/recommendation quality tracking

### Time-Windowed Analysis
- Hourly aggregated metrics for trend detection
- Regional performance comparison
