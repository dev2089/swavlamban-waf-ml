from __future__ import annotations

from waf.core.models import RequestEnvelope
from waf.features.http_v2 import ProductionHTTPFeatureExtractor, normalize_target


def req(**kwargs) -> RequestEnvelope:
    data = dict(
        request_id="p3",
        method="POST",
        scheme="https",
        host="Example.TEST",
        path="/search",
        query="q=%2525253Cscript%253E",
        headers={
            "Content-Type": "application/json",
            "Cookie": "a=1; b=2",
            "Content-Length": "15",
        },
        body=b'{"x":"hello"}',
    )
    data.update(kwargs)
    return RequestEnvelope(**data)


def test_version_and_numeric_bounds() -> None:
    fv = ProductionHTTPFeatureExtractor().extract(req())
    assert fv.schema_version == "http-v2"
    assert len(fv.values) >= 40
    assert all(isinstance(k, str) and isinstance(v, float) for k, v in fv.values.items())
    assert all(0.0 <= v <= 1.0 for v in fv.values.values())


def test_multi_pass_decoding_and_unicode_normalization() -> None:
    assert "<script>" in normalize_target("/", "q=%25253Cscript%25253E")
    assert normalize_target("/\uff0fadmin", "").endswith("/admin")
    assert "/a+b" in normalize_target("/a+b", "")


def test_query_duplicates_cookie_content_type() -> None:
    fv = ProductionHTTPFeatureExtractor().extract(
        req(
            query="a=1&a=2&b=",
            headers={"cookie": "a=1; b=2", "content-type": "application/json"},
        )
    )
    assert fv.values["duplicate_query_key_count"] > 0.0
    assert fv.values["cookie_count"] > 0.0
    assert fv.values["has_json_body"] == 1.0


def test_malformed_percent_double_encoding_and_cl() -> None:
    fv = ProductionHTTPFeatureExtractor().extract(
        req(query="x=%GG&y=%2525253Cscript%2525253E", headers={"Content-Length": "1"})
    )
    assert fv.values["malformed_percent_flag"] == 1.0
    assert fv.values["double_encoded_flag"] == 1.0
    assert fv.values["has_xss_token"] == 1.0
    assert fv.values["content_length_mismatch"] == 1.0


def test_invalid_utf8_and_control_bytes_are_safe() -> None:
    fv = ProductionHTTPFeatureExtractor().extract(
        req(body=(b"abc\xff\xfe\x00\x01" * 100))
    )
    assert fv.values["body_utf8_replacement_ratio"] > 0.0
    assert fv.values["null_byte_flag"] == 1.0
    assert fv.values["control_char_ratio"] > 0.0


def test_query_overflow_is_explicitly_flagged() -> None:
    query = "&".join(f"k{i}=1" for i in range(300))
    fv = ProductionHTTPFeatureExtractor().extract(req(query=query))
    assert fv.values["query_parse_overflow"] == 1.0


def test_header_and_body_processing_are_bounded() -> None:
    headers = {f"X-{i}": "v" * 100 for i in range(1000)}
    fv = ProductionHTTPFeatureExtractor().extract(
        req(headers=headers, body=b"C" * 400_000)
    )
    assert fv.values["header_count"] == 1.0
    assert fv.values["body_length"] > 0.0


def test_feature_vector_contains_no_raw_request_material() -> None:
    fv = ProductionHTTPFeatureExtractor().extract(
        req(query="token=super-secret-value", body=b"api-key=secret")
    )
    rendered = str(fv.values)
    assert "super-secret-value" not in rendered
    assert "api-key=secret" not in rendered


def test_determinism() -> None:
    extractor = ProductionHTTPFeatureExtractor()
    assert extractor.extract(req()) == extractor.extract(req())


def test_attack_indicators_cover_rules() -> None:
    extractor = ProductionHTTPFeatureExtractor()
    payload = req(
        query="q=union+select+1+from+users",
        body=b"<script>alert(1)</script>; id && wget x",
    )
    values = extractor.extract(payload).values
    assert values["has_sql_keyword"] == 1.0
    assert values["has_xss_token"] == 1.0
    assert values["has_command_token"] == 1.0


def test_unknown_method_is_safe() -> None:
    fv = ProductionHTTPFeatureExtractor().extract(req(method="UNKNOWN"))
    assert fv.values["method_known"] == 0.0
    assert fv.values["method_code"] == 0.0
