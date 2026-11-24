"""Creates a Spark session tuned for a single Windows/Linux/macOS laptop.

Spark runs in *local mode*: the driver and the executors share one JVM and
use every CPU core as a parallel worker (``local[*]``). No Hadoop cluster,
YARN, Kafka or cloud service is involved.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

from pyspark.sql import SparkSession


def create_spark(cfg: dict[str, Any]) -> SparkSession:
    # Make sure Spark's Python workers use the *same* interpreter as the driver.
    # On Windows a mismatch causes the classic "Python worker failed to connect back".
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)

    run = cfg["run"]
    spark_cfg = cfg["spark"]
    tmp_dir = Path(cfg["paths"]["spark_tmp_dir"])
    tmp_dir.mkdir(parents=True, exist_ok=True)

    builder = (
        SparkSession.builder.master(spark_cfg.get("master", "local[*]"))
        .appName(f"{spark_cfg.get('app_name', 'TasteTrends')}-{cfg['mode']}")
        # Memory: in local mode the driver JVM also hosts the executors.
        .config("spark.driver.memory", run["driver_memory"])
        .config("spark.driver.maxResultSize", "2g")
        # Partitioning: number of partitions created by joins / aggregations.
        .config("spark.sql.shuffle.partitions", str(run["shuffle_partitions"]))
        .config("spark.default.parallelism", str(run["shuffle_partitions"]))
        # Adaptive Query Execution coalesces small shuffle partitions at runtime.
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
        # Arrow makes Spark -> pandas transfers of small results much faster.
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .config("spark.sql.execution.arrow.pyspark.fallback.enabled", "true")
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.local.dir", str(tmp_dir))
        .config("spark.ui.showConsoleProgress", "false")
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
    )
    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel(spark_cfg.get("log_level", "WARN"))
    return spark
