import json

import pytest

from deploy.core import subjective_quality


def test_parse_ollama_response_accepts_json_and_bounds_score() -> None:
    description, score = subjective_quality._parse_ollama_response(
        '{"description":"画面清晰，压缩伪影较少","quality_score":82}'
    )

    assert description == "画面清晰，压缩伪影较少"
    assert score == 82


def test_parse_ollama_response_extracts_json_from_model_prefix() -> None:
    description, score = subjective_quality._parse_ollama_response(
        'Here is the result:\n{"description":"轻微模糊","quality_score":60}'
    )

    assert description == "轻微模糊"
    assert score == 60


@pytest.mark.parametrize(
    ("raw_score", "expected"),
    [("78", 78), (50.9, 50), (-4, 0), (104, 100)],
)
def test_parse_ollama_response_normalizes_model_score(raw_score, expected) -> None:
    _, score = subjective_quality._parse_ollama_response(
        json.dumps({"description": "质量描述", "quality_score": raw_score})
    )
    assert score == expected


@pytest.mark.parametrize(
    "response",
    [
        "not json",
        json.dumps({"description": "缺少分数"}),
        json.dumps({"description": "" , "quality_score": 50}),
        json.dumps({"description": "额外字段", "quality_score": 50, "extra": True}),
    ],
)
def test_parse_ollama_response_rejects_invalid_payload(response: str) -> None:
    with pytest.raises(subjective_quality.SubjectiveQualityError):
        subjective_quality._parse_ollama_response(response)


def test_bayesian_score_is_monotonic_and_interval_is_bounded() -> None:
    low, low_interval = subjective_quality._bayesian_quality_score(20)
    high, high_interval = subjective_quality._bayesian_quality_score(80)

    assert low < high
    assert 0 <= low_interval[0] <= low <= low_interval[1] <= 100
    assert 0 <= high_interval[0] <= high <= high_interval[1] <= 100


def test_assessment_uses_ollama_response_without_persisting_results(monkeypatch, tmp_path) -> None:
    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return json.dumps(
                {"response": '{"description":"清晰度良好","quality_score":80}'}
            ).encode("utf-8")

    captured = {}

    def fake_urlopen(request, timeout):
        captured["request"] = request
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr(subjective_quality, "_image_payload", lambda _path: ["encoded-frame"])
    monkeypatch.setattr(subjective_quality.urllib.request, "urlopen", fake_urlopen)
    media_path = tmp_path / "sample.jpg"
    media_path.write_bytes(b"temporary media")

    result = subjective_quality.assess_subjective_quality(
        media_path, "image", "http://ollama:11434/", "deep-vqa-subjective", 7, 4
    )

    assert result.description == "清晰度良好"
    assert result.llm_score == 80
    assert result.bayesian_score == 65
    assert result.frames_sampled == 1
    assert result.model_used == "deep-vqa-subjective"
    assert captured["request"].full_url == "http://ollama:11434/api/generate"
    assert json.loads(captured["request"].data)["images"] == ["encoded-frame"]
    assert captured["timeout"] == 7
    assert media_path.exists()
