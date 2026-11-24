"""STEP 1 - inspect the computer before running the project.

    python scripts/check_environment.py            # checks only
    python scripts/check_environment.py --spark    # also starts a tiny Spark job

Reports the Python version, Java version/JAVA_HOME, RAM, CPU cores, free disk,
and whether each required Python package is installed.
"""
from __future__ import annotations

import argparse
import importlib
import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path

import _bootstrap  # noqa: F401

PACKAGES = ["pyspark", "pandas", "numpy", "pyarrow", "plotly", "streamlit", "yaml", "pytest"]
OK, WARN, FAIL = "[ OK ]", "[WARN]", "[FAIL]"


def total_ram_gb() -> float | None:
    try:
        if platform.system() == "Windows":
            import ctypes

            class MemStatus(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("sullAvailExtendedVirtual", ctypes.c_ulonglong)]

            stat = MemStatus()
            stat.dwLength = ctypes.sizeof(MemStatus)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
            return stat.ullTotalPhys / 1024 ** 3
        if hasattr(os, "sysconf"):
            return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1024 ** 3
    except Exception:
        return None
    return None


def java_version() -> tuple[str | None, int | None]:
    java = "java"
    if os.environ.get("JAVA_HOME"):
        candidate = Path(os.environ["JAVA_HOME"]) / "bin" / ("java.exe" if os.name == "nt" else "java")
        if candidate.exists():
            java = str(candidate)
    if java == "java" and not shutil.which("java"):
        return None, None
    try:
        out = subprocess.run([java, "-version"], capture_output=True, text=True, timeout=30)
        text = (out.stderr or out.stdout).strip()
        match = re.search(r'version "(\d+)(?:\.(\d+))?', text)
        major = None
        if match:
            major = int(match.group(1))
            if major == 1 and match.group(2):  # Java 8 reports "1.8"
                major = int(match.group(2))
        version_lines = [line for line in text.splitlines() if "version" in line and "Picked up" not in line]
        return (version_lines[0] if version_lines else text.splitlines()[-1]), major
    except Exception as exc:
        return f"error: {exc}", None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spark", action="store_true", help="start Spark and run a tiny job")
    args = parser.parse_args()
    problems = 0

    print("=" * 64)
    print(" Environment check - Taste, Trends & Algorithms")
    print("=" * 64)
    print(f"OS        : {platform.system()} {platform.release()} ({platform.machine()})")
    py = sys.version_info
    py_ok = (3, 9) <= (py.major, py.minor) <= (3, 12)
    print(f"{OK if py_ok else WARN} Python {platform.python_version()} at {sys.executable}")
    if not py_ok:
        print("       Recommended: Python 3.10 or 3.11 for PySpark 3.5.")

    jtext, jmajor = java_version()
    if jtext is None:
        problems += 1
        print(f"{FAIL} Java not found. Install Temurin JDK 17 and set JAVA_HOME.")
    else:
        j_ok = jmajor in (8, 11, 17)
        print(f"{OK if j_ok else WARN} Java: {jtext}")
        if not j_ok:
            print("       PySpark 3.5 officially supports Java 8/11/17. Java 17 is recommended.")
    print(f"       JAVA_HOME = {os.environ.get('JAVA_HOME', '(not set)')}")
    if os.name == "nt":
        print(f"       HADOOP_HOME = {os.environ.get('HADOOP_HOME', '(not set - fine, this project does not need winutils)')}")

    if " " in str(_bootstrap.ROOT) or " " in sys.executable:
        print(f"{WARN} The project or Python path contains a space: {_bootstrap.ROOT}")
        print("       Spark's Windows launch scripts can fail on such paths. Move the project to e.g. C:\\BDA\\")
    ram = total_ram_gb()
    if ram is not None:
        status = OK if ram >= 8 else WARN
        print(f"{status} RAM: {ram:.1f} GB" + ("" if ram >= 8 else "  (use --mode dev or driver_memory 3g)"))
    print(f"{OK} CPU cores: {os.cpu_count()}")
    free_gb = shutil.disk_usage(_bootstrap.ROOT).free / 1024 ** 3
    print(f"{OK if free_gb > 5 else WARN} Free disk: {free_gb:.1f} GB (MovieLens 25M needs ~1.5 GB)")

    print("-" * 64)
    for name in PACKAGES:
        try:
            mod = importlib.import_module(name)
            print(f"{OK} {name:<10} {getattr(mod, '__version__', '')}")
        except ImportError:
            problems += 1
            print(f"{FAIL} {name:<10} not installed  ->  pip install -r requirements.txt")

    if args.spark:
        print("-" * 64)
        try:
            from src.config import load_config
            from src.spark_session import create_spark

            spark = create_spark(load_config("dev"))
            n = spark.range(0, 1_000_000).selectExpr("sum(id) AS s").collect()[0]["s"]
            print(f"{OK} Spark {spark.version} started; test job result = {n}")
            spark.stop()
        except Exception as exc:
            problems += 1
            print(f"{FAIL} Spark could not start: {exc}")

    print("=" * 64)
    print("All checks passed." if problems == 0 else f"{problems} problem(s) found - see above.")
    return 0 if problems == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
