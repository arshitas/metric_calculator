"""
Telemetry Metrics Pipeline - Entry Point

A scalable telemetry processing pipeline using PySpark to process
large-scale log data and compute key performance metrics.
"""

import argparse
import sys
import time


def main():
    parser = argparse.ArgumentParser(
        description="Telemetry Metrics Pipeline - Process large-scale telemetry data with PySpark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                     # Run with 1M records (default)
  python main.py --records 5000000   # Run with 5M records
  python main.py --seed 123          # Custom random seed
        """,
    )
    parser.add_argument(
        "--records",
        type=int,
        default=1_000_000,
        help="Number of telemetry records to generate (default: 1,000,000)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible data generation (default: 42)",
    )

    args = parser.parse_args()

    if args.records < 1000:
        print("Error: Minimum 1,000 records required for meaningful metrics.")
        sys.exit(1)

    print("╔══════════════════════════════════════════════════════════╗")
    print("║        Telemetry Metrics Pipeline                       ║")
    print("║        Scalable PySpark Telemetry Processing            ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print(f"\n  Records to generate: {args.records:,}")
    print(f"  Random seed:         {args.seed}")

    start_time = time.time()

    from src.pipeline import run_pipeline

    run_pipeline(num_records=args.records, seed=args.seed)

    elapsed = time.time() - start_time
    print(f"\n  Total execution time: {elapsed:.2f} seconds")


if __name__ == "__main__":
    main()
