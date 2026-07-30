# src/utils/logging_utils.py
import atexit
import sys
import time
from functools import wraps
from pathlib import Path
from typing import Callable, Optional

from loguru import logger
from src.config.schemas import Config

def _csv_safe_patcher(record) -> None:
    """
    Escape newline characters and double quotes in log messages for CSV safety.

    Replaces `"` with `""` (CSV standard escaping) and `\n` with `\\n`
    to keep log entries on a single line.
    """
    msg = record["message"]
    msg = msg.replace('"', '""')
    if "\n" in msg:
        msg = msg.replace("\n", "\\n")
    record["extra"]["csv_message"] = msg


def _on_exit() -> None:
    """Log a message when the process exits."""
    if sys.exc_info()[0]:
        logger.error(f"Process terminated abnormally: {sys.exc_info()[1]}")
    else:
        logger.info("Process completed successfully.")


def log_prepare(
    cfg: Config,
    model_name: str = "resnet50",
    dataset_name: str = "TID2013",
    log_level: str = "DEBUG",
) -> str:
    """
    Configure global logging with console, file, and CSV outputs.

    Args:
        model_name: Name of the model (used in log filename)
        dataset_name: Name of the dataset (used in log filename)
        log_level: Minimum log level for file output (default: DEBUG)

    Returns:
        base_filename: The base filename used for log files
    """
    log_dir = cfg.paths.train_logs_dir(dataset_name)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Clear any existing Loguru handlers
    logger.remove()

    # Attach CSV patcher for safe CSV output
    logger.configure(patcher=_csv_safe_patcher)

    # Console: INFO and above (clean, user-friendly)
    console_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <5}</level> | "
        "<cyan>{name}:{function}:{line}</cyan> - "
        "<level>{message}</level>"
    )
    logger.add(
        sys.stderr,
        level="INFO",
        format=console_format,
    )

    # Generate timestamped filename
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    base_filename = f"{timestamp}_{model_name.lower()}_{dataset_name.lower()}"

    # File log: DEBUG and above (full detail)
    file_format = (
        "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
        "{level: <5} | "
        "{name}:{function}:{line} - "
        "{message}"
    )
    logger.add(
        log_dir / f"{base_filename}.log",
        rotation="500 MB",
        level=log_level,
        format=file_format,
        encoding="utf-8",
    )

    # CSV log: structured, escapable, with header
    csv_format = (
        '{time:YYYY-MM-DD HH:mm:ss.SSS},'
        '{level},'
        '{name},{function},{line},'
        '"{extra[csv_message]}"'
    )
    csv_path = log_dir / f"{base_filename}.csv"
    csv_header = "timestamp,level,module,function,line,message\n"

    def csv_rotation_callback(message, file_object):
        file_object.write(csv_header)

    if not csv_path.exists():
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write(csv_header)

    logger.add(
        csv_path,
        rotation=csv_rotation_callback,
        level=log_level,
        format=csv_format,
        encoding="utf-8",
    )

    logger.info(f"Logging initialized. Logs: {log_dir / base_filename}")
    logger.debug(f"TXT log: {base_filename}.log")
    logger.debug(f"CSV log: {base_filename}.csv")

    atexit.register(_on_exit)
    return base_filename


def time_it(func: Callable) -> Callable:
    """
    Decorator to measure and log execution time of a function.

    Usage:
        @time_it
        def train_model():
            ...
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start

        module = func.__module__.split(".")[-1]
        logger.debug(f"{module}.{func.__name__}: {elapsed:.4f}s")
        return result

    return wrapper