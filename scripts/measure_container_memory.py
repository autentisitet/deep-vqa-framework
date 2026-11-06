#!/usr/bin/env python3
"""Sample container and host memory across idle and scoring workloads."""

from __future__ import annotations

import argparse
import http.cookiejar
import json
import math
import os
import shutil
import subprocess
import threading
import time
import tomllib
import urllib.error
import urllib.request
import uuid
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, TypedDict

CONTAINERS = ("vqa-infer", "vqa-nginx")
UNIT_BYTES = {
    "b": 1,
    "kb": 1000,
    "kib": 1024,
    "mb": 1000**2,
    "mib": 1024**2,
    "gb": 1000**3,
    "gib": 1024**3,
}


class MemoryPhase(TypedDict):
    name: str
    elapsed_seconds: float
    sample_interval_seconds: float
    sample_count: int
    sampled_peak_bytes: int
    sampled_peak_containers: dict[str, int]
    settled_bytes: int
    settled_containers: dict[str, int]
    host_total_bytes: int | None
    host_available_bytes: int | None
    swap_used_bytes: int | None
    peak_is_sampling_lower_bound: bool
    response: dict[str, object]


class MemoryReport(TypedDict):
    recorded_at: str
    runtime: str
    base_url: str
    containers: list[str]
    phases: list[MemoryPhase]


@dataclass(frozen=True)
class MemorySnapshot:
    timestamp: float
    containers: dict[str, int]
    host_total_bytes: int | None = None
    host_available_bytes: int | None = None
    swap_used_bytes: int | None = None

    @property
    def total_bytes(self) -> int:
        """Sum of all measured container memory usages."""
        return sum(self.containers.values())


def _load_env(path: Path) -> None:
    """Load a local .env without overriding exported variables."""
    if not path.is_file():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def _parse_bytes(value: str) -> int:
    """Parse a container runtime memory string like ``"123.4MiB / 1GiB"``."""
    token = value.strip().split("/", 1)[0].strip()
    number = "".join(character for character in token if character.isdigit() or character in ".-")
    unit = token[len(number) :].strip().lower()
    if not number or unit not in UNIT_BYTES:
        raise ValueError(f"Unsupported memory value: {value}")
    return round(float(number) * UNIT_BYTES[unit])


def _runtime(requested: str) -> str:
    """Resolve the requested container runtime to an available executable."""
    if requested != "auto":
        if shutil.which(requested) is None:
            raise RuntimeError(f"Container runtime is unavailable: {requested}")
        return requested
    for candidate in ("podman", "docker"):
        if shutil.which(candidate):
            return candidate
    raise RuntimeError("Neither podman nor docker is available")


def _host_memory() -> tuple[int | None, int | None, int | None]:
    """Read Linux host memory using MemTotal, MemAvailable, and swap counters."""
    try:
        values: dict[str, int] = {}
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            name, raw = line.split(":", 1)
            if name in {"MemTotal", "MemAvailable", "SwapTotal", "SwapFree"}:
                values[name] = int(raw.strip().split()[0]) * 1024
        total = values["MemTotal"]
        available = values["MemAvailable"]
        swap_used = values.get("SwapTotal", 0) - values.get("SwapFree", 0)
        return total, available, swap_used
    except (KeyError, OSError, ValueError, IndexError):
        return None, None, None


def _snapshot(runtime: str) -> MemorySnapshot:
    """Read one memory snapshot for every measured container plus the host."""
    command = [runtime, "stats", "--no-stream", "--format", "{{.Name}}|{{.MemUsage}}", *CONTAINERS]
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    containers: dict[str, int] = {}
    for line in completed.stdout.splitlines():
        if "|" not in line:
            continue
        name, usage = line.split("|", 1)
        containers[name.strip()] = _parse_bytes(usage)
    missing = sorted(set(CONTAINERS) - containers.keys())
    if missing:
        raise RuntimeError(f"Container stats are missing: {', '.join(missing)}")
    host_total, host_available, swap_used = _host_memory()
    return MemorySnapshot(time.time(), containers, host_total, host_available, swap_used)


@dataclass(frozen=True)
class MultipartBody:
    """Replayable upload iterable; each request reads at most 1 MiB at a time.

    Chunks own their bytes so consumers may retain them safely. The input file
    must remain unchanged until the HTTP request completes.
    """

    path: Path
    header: bytes
    trailer: bytes
    file_size: int

    @property
    def content_length(self) -> int:
        """Length of the multipart body, including framing."""
        return len(self.header) + self.file_size + len(self.trailer)

    def __iter__(self) -> Iterator[bytes]:
        yield self.header
        with self.path.open("rb") as source:
            yield from iter(lambda: source.read(1024 * 1024), b"")
        yield self.trailer


def _multipart(path: Path) -> tuple[MultipartBody, str]:
    """Prepare multipart framing without materializing the media file."""
    boundary = f"----deep-vqa-{uuid.uuid4().hex}"
    header = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{path.name}"\r\n'
        "Content-Type: application/octet-stream\r\n\r\n"
    ).encode()
    trailer = f"\r\n--{boundary}--\r\n".encode()
    body = MultipartBody(path, header, trailer, path.stat().st_size)
    return body, f"multipart/form-data; boundary={boundary}"


def _opener(
    base_url: str,
    api_key: str,
    email: str,
    password: str,
) -> urllib.request.OpenerDirector:
    """Build an opener, signing in with whichever credential shape is configured."""
    cookies = http.cookiejar.CookieJar()
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cookies), urllib.request.ProxyHandler({}))
    credentials: dict[str, str] = {}
    if email and password:
        credentials = {"email": email, "password": password}
    elif api_key:
        credentials = {"api_key": api_key}
    if credentials:
        request = urllib.request.Request(
            f"{base_url}/auth/login",
            data=json.dumps(credentials).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with opener.open(request, timeout=30):
            pass
    return opener


def _post_media(
    opener: urllib.request.OpenerDirector,
    url: str,
    media_path: Path,
    timeout: float,
) -> dict[str, object]:
    """POST one media file and return the parsed JSON response."""
    body, content_type = _multipart(media_path)
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": content_type, "Content-Length": str(body.content_length)},
        method="POST",
    )
    try:
        with opener.open(request, timeout=timeout) as response:
            payload: object = json.loads(response.read().decode("utf-8"))
            if not isinstance(payload, dict) or any(not isinstance(key, str) for key in payload):
                raise RuntimeError(f"Expected a JSON object from {url}")
            return {key: value for key, value in payload.items()}
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from {url}: {detail}") from exc


def _measure_phase(
    name: str,
    operation: Callable[[], dict[str, object]],
    runtime: str,
    interval: float,
    settle_seconds: float,
) -> MemoryPhase:
    """Run one workload while sampling memory, returning the phase summary."""
    samples: list[MemorySnapshot] = [_snapshot(runtime)]
    stop = threading.Event()

    def sample() -> None:
        """Continuously append memory snapshots until the phase ends."""
        while not stop.wait(interval):
            try:
                samples.append(_snapshot(runtime))
            except (RuntimeError, subprocess.SubprocessError, ValueError):
                continue

    sampler = threading.Thread(target=sample, name=f"memory-sampler-{name}", daemon=True)
    sampler.start()
    started = time.perf_counter()
    try:
        response = operation()
    finally:
        stop.set()
        sampler.join(timeout=max(1.0, interval * 2))
    elapsed = time.perf_counter() - started
    deadline = time.monotonic() + settle_seconds
    while time.monotonic() < deadline:
        time.sleep(min(interval, max(0.0, deadline - time.monotonic())))
        samples.append(_snapshot(runtime))
    sampled_peak = max(samples, key=lambda item: item.total_bytes)
    per_container_peaks = {
        container: max(sample.containers[container] for sample in samples) for container in CONTAINERS
    }
    settled = samples[-1]
    return {
        "name": name,
        "elapsed_seconds": round(elapsed, 3),
        "sample_interval_seconds": interval,
        "sample_count": len(samples),
        "sampled_peak_bytes": sampled_peak.total_bytes,
        "sampled_peak_containers": per_container_peaks,
        "settled_bytes": settled.total_bytes,
        "settled_containers": settled.containers,
        "host_total_bytes": settled.host_total_bytes,
        "host_available_bytes": settled.host_available_bytes,
        "swap_used_bytes": settled.swap_used_bytes,
        "peak_is_sampling_lower_bound": True,
        "response": response,
    }


def _to_mib(value: int) -> float:
    """Convert bytes to mebibytes, rounded to two decimals."""
    return round(value / 1024**2, 2)


def _format_mib(value: int | None) -> str:
    """Format a byte count as MiB, or ``"n/a"`` when unavailable."""
    return f"{_to_mib(value):.2f} MiB" if value is not None else "n/a"


def _write_report(path: Path, report: MemoryReport) -> None:
    """Write the measurement JSON and a Markdown summary."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    markdown = path.with_suffix(".md")
    rows = [
        "# Deep-VQA container memory measurement",
        "",
        f"Recorded: {report['recorded_at']}",
        f"Runtime: `{report['runtime']}`",
        "",
        "The reported peak is the largest sampled value and may be lower than",
        "the true instantaneous peak between samples.",
        "",
        "| Phase | Elapsed | Samples | Interval | Sampled peak | Settled | Host available | Swap used |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for phase in report["phases"]:
        rows.append(
            f"| {phase['name']} | {phase['elapsed_seconds']:.3f}s | {phase['sample_count']} | "
            f"{phase['sample_interval_seconds']:.3f}s | {_to_mib(phase['sampled_peak_bytes']):.2f} MiB | "
            f"{_to_mib(phase['settled_bytes']):.2f} MiB | "
            f"{_format_mib(phase['host_available_bytes'])} | "
            f"{_format_mib(phase['swap_used_bytes'])} |"
        )
    rows.extend(["", "## Per-container measurements", ""])
    for phase in report["phases"]:
        rows.append(f"### {phase['name']}")
        rows.append("")
        rows.append("| Container | Peak | Settled |")
        rows.append("| --- | ---: | ---: |")
        for container in CONTAINERS:
            rows.append(
                f"| `{container}` | {_to_mib(phase['sampled_peak_containers'][container]):.2f} MiB | "
                f"{_to_mib(phase['settled_containers'][container]):.2f} MiB |"
            )
        rows.append("")
    markdown.write_text("\n".join(rows) + "\n", encoding="utf-8")


@dataclass(frozen=True)
class MemoryOptions:
    """Explicit execution inputs, independent of the CLI parser."""

    runtime: str = "auto"
    base_url: str = "http://127.0.0.1:8000/api/v1"
    image: Path = Path("examples/images/psc (2).jpeg")
    video: Path = Path("examples/videos/first_pentakill.mp4")
    sample_interval: float = 0.05
    settle_seconds: float = 5.0
    timeout: float = 300.0
    skip_video: bool = False
    email: str = ""
    password: str = field(default="", repr=False)
    output: Path = Path("reports/memory/latest.json")


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line contract without executing the command."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Examples:\n  python scripts/measure_container_memory.py --runtime podman\n  python scripts/measure_container_memory.py --skip-video --sample-interval 0.1",
    )
    parser.add_argument("--runtime", choices=("auto", "podman", "docker"), default="auto")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/api/v1")
    parser.add_argument("--image", type=Path, default=Path("examples/images/psc (2).jpeg"))
    parser.add_argument("--video", type=Path, default=Path("examples/videos/first_pentakill.mp4"))
    parser.add_argument("--sample-interval", type=float, default=0.05)
    parser.add_argument("--settle-seconds", type=float, default=5.0)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--skip-video", action="store_true")
    parser.add_argument("--email", default=os.getenv("DEEP_VQA_EMAIL", ""))
    parser.add_argument("--password", default=os.getenv("DEEP_VQA_PASSWORD", ""))
    parser.add_argument("--output", type=Path, default=Path("reports/memory/latest.json"))
    parser.add_argument(
        "--version",
        action="version",
        version=f"Deep-VQA {tomllib.loads((Path(__file__).resolve().parents[1] / 'pyproject.toml').read_text(encoding='utf-8'))['project']['version']}",
    )
    return parser


def parse_args(argv: list[str] | None = None) -> MemoryOptions:
    """Parse and validate CLI input, returning project-owned execution options."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.sample_interval <= 0:
        parser.error("--sample-interval must be positive")

    if not math.isfinite(args.sample_interval) or not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("--sample-interval and --timeout must be finite and positive")
    if not math.isfinite(args.settle_seconds) or args.settle_seconds < 0:
        parser.error("--settle-seconds must be finite and non-negative")
    return MemoryOptions(
        runtime=args.runtime,
        base_url=args.base_url,
        image=args.image,
        video=args.video,
        sample_interval=args.sample_interval,
        settle_seconds=args.settle_seconds,
        timeout=args.timeout,
        skip_video=args.skip_video,
        email=args.email,
        password=args.password,
        output=args.output,
    )


def run(options: MemoryOptions) -> int:
    """Consume parsed options and execute the command."""
    _load_env(Path(".env"))
    runtime = _runtime(options.runtime)
    opener = _opener(
        options.base_url.rstrip("/"),
        os.getenv("DEEP_VQA_API_KEY", ""),
        options.email,
        options.password,
    )
    phases: list[MemoryPhase] = []
    idle = _snapshot(runtime)
    phases.append(
        {
            "name": "idle",
            "elapsed_seconds": 0.0,
            "sample_interval_seconds": options.sample_interval,
            "sample_count": 1,
            "sampled_peak_bytes": idle.total_bytes,
            "sampled_peak_containers": idle.containers,
            "settled_bytes": idle.total_bytes,
            "settled_containers": idle.containers,
            "host_total_bytes": idle.host_total_bytes,
            "host_available_bytes": idle.host_available_bytes,
            "swap_used_bytes": idle.swap_used_bytes,
            "peak_is_sampling_lower_bound": True,
            "response": {},
        }
    )
    phases.append(
        _measure_phase(
            "image-evaluation",
            lambda: _post_media(opener, f"{options.base_url}/evaluations?model_id=iqa", options.image, options.timeout),
            runtime,
            options.sample_interval,
            options.settle_seconds,
        )
    )
    if not options.skip_video:
        phases.append(
            _measure_phase(
                "video-evaluation",
                lambda: _post_media(
                    opener, f"{options.base_url}/evaluations?model_id=vqa", options.video, options.timeout
                ),
                runtime,
                options.sample_interval,
                options.settle_seconds,
            )
        )
    report: MemoryReport = {
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "runtime": runtime,
        "base_url": options.base_url,
        "containers": list(CONTAINERS),
        "phases": phases,
    }
    _write_report(options.output, report)
    print(f"Memory report: {options.output}")
    print(f"Markdown report: {options.output.with_suffix('.md')}")
    for phase in phases:
        print(
            f"{phase['name']}: sampled_peak={_to_mib(phase['sampled_peak_bytes']):.2f} MiB "
            f"settled={_to_mib(phase['settled_bytes']):.2f} MiB "
            f"host_available={_format_mib(phase['host_available_bytes'])} "
            f"swap_used={_format_mib(phase['swap_used_bytes'])}"
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    """Parse command-line options and execute the command."""
    return run(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
