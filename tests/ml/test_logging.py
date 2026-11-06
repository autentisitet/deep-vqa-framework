"""CSV headers survive append and rotation without becoming data rows."""

import csv
import subprocess
import sys


def test_csv_logging_writes_one_header_per_file_including_rotation(tmp_path):
    # Isolate global Loguru handlers and atexit registration from the test runner.
    program = """
import sys
from pathlib import Path
from loguru import logger
from src.config.loader import load_config
from src.utils.logging_utils import log_prepare

root = Path(sys.argv[1])
config = load_config()
config.paths.project_root = root
config.paths.results_dir = root / "results"
original_add = logger.add

def add(sink, **kwargs):
    if str(sink).endswith(".csv"):
        # Exercise real rotation using a small threshold and the configured opener.
        kwargs["rotation"] = 300
    return original_add(sink, **kwargs)

logger.add = add
log_prepare(config)
for index in range(10):
    logger.info('message {} with "quotes" and newline\\ncontinued', index)
logger.remove()
"""
    result = subprocess.run([sys.executable, "-c", program, str(tmp_path)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    files = list(tmp_path.rglob("*.csv"))
    assert len(files) > 1
    messages = []
    header = ["timestamp", "level", "module", "function", "line", "message"]
    for path in files:
        with path.open(newline="") as source:
            rows = list(csv.reader(source))
        assert rows[0] == header
        assert all(len(row) == len(header) and row != header for row in rows[1:])
        messages.extend(row[-1] for row in rows[1:])
    assert sum(message.startswith("message ") for message in messages) == 10
    assert 'message 0 with "quotes" and newline\\ncontinued' in messages
