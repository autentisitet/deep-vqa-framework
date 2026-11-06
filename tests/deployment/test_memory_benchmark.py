from scripts import measure_container_memory as benchmark


def test_parse_container_memory_units() -> None:
    assert benchmark._parse_bytes("512MiB / 4GiB") == 512 * 1024**2
    assert benchmark._parse_bytes("2.5GB / 16GB") == 2_500_000_000
    assert benchmark._parse_bytes("64kB / 1GB") == 64_000


def test_memory_snapshot_sums_containers() -> None:
    snapshot = benchmark.MemorySnapshot(0.0, {"a": 10, "b": 20}, 100, 40, 5)

    assert snapshot.total_bytes == 30
    assert snapshot.host_available_bytes == 40
    assert snapshot.swap_used_bytes == 5


def test_format_mib_handles_unavailable_host_metrics() -> None:
    assert benchmark._format_mib(None) == "n/a"
    assert benchmark._format_mib(1024**2) == "1.00 MiB"


def test_host_memory_reads_linux_meminfo(monkeypatch, tmp_path) -> None:
    meminfo = tmp_path / "meminfo"
    meminfo.write_text(
        "MemTotal:       4096000 kB\n"
        "MemAvailable:   1024000 kB\n"
        "SwapTotal:      2048000 kB\n"
        "SwapFree:       1536000 kB\n",
        encoding="utf-8",
    )
    original_read_text = benchmark.Path.read_text

    def fake_read_text(path, *args, **kwargs):
        if str(path) == "/proc/meminfo":
            return meminfo.read_text(encoding="utf-8")
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(benchmark.Path, "read_text", fake_read_text)

    assert benchmark._host_memory() == (4096000 * 1024, 1024000 * 1024, 512000 * 1024)


def test_multipart_is_bounded_replayable_and_preserves_framing(tmp_path, monkeypatch):
    from pathlib import Path

    path = tmp_path / "video.mp4"
    payload = b"x" * (2 * 1024 * 1024 + 17)
    path.write_bytes(payload)

    def forbid_read_bytes(_path):
        raise AssertionError("upload must not materialize the whole file")

    monkeypatch.setattr(Path, "read_bytes", forbid_read_bytes)
    body, content_type = benchmark._multipart(path)
    boundary = content_type.split("boundary=", 1)[1]
    for _ in range(2):
        chunks = list(body)
        assert max(map(len, chunks)) <= 1024 * 1024
        assert b"".join(chunks[1:-1]) == payload
        assert chunks[0].startswith(f"--{boundary}\r\n".encode())
        assert chunks[-1] == f"\r\n--{boundary}--\r\n".encode()
        assert sum(map(len, chunks)) == body.content_length


def test_post_media_streams_through_http_transport(tmp_path, monkeypatch):
    import http.client
    import io
    import urllib.request

    path = tmp_path / "video.mp4"
    path.write_bytes(b"video" * 300000)
    requests = []

    class Socket:
        def __init__(self):
            self.sent = bytearray()

        def sendall(self, data):
            self.sent.extend(data)

        def makefile(self, *_args, **_kwargs):
            return io.BytesIO(b'HTTP/1.1 200 OK\r\nContent-Length: 11\r\n\r\n{"ok":true}')

        def close(self):
            pass

    def connect(connection):
        connection.sock = Socket()
        requests.append(connection.sock)

    monkeypatch.setattr(http.client.HTTPConnection, "connect", connect)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    assert benchmark._post_media(opener, "http://test.invalid/evaluations", path, 5) == {"ok": True}
    headers, body = bytes(requests[0].sent).split(b"\r\n\r\n", 1)
    assert f"Content-Length: {len(body)}".encode() in headers
    assert b"Transfer-Encoding" not in headers
    assert path.read_bytes() in body
