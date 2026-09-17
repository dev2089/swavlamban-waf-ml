from __future__ import annotations

from waf.core.models import RequestEnvelope
from waf.features.http_v2 import ProductionHTTPFeatureExtractor, normalize_target


def req(**kwargs):
    data = dict(
        request_id="p3",
        method="POST",
        scheme="https",
        host="Example.TEST",
        path="/search",
        query="q=%2525253Cscript%253E",
        headers={"Content-Type": "application/json", "Cookie": "a=1; b=2", "Content-Length": "15"},
        body=b'{"x":"hello"}',
    )
    data.update(kwargs)
    return RequestEnvelope(**data)


def test_version_ranges_and_numeric_only():
    fv = ProductionHTTPFeatureExtractor().extract(req())
    assert fv.schema_version == "http-v2"
    assert len(fv.values) >= 30
    assert all(isinstance(k, str) and isinstance(v, float) for k, v in fv.values.items())
    assert all(0.0 <= v <= 1.0 for v in fv.values.values())


def test_multi_pass_decoding_and_unicode_normalization():
    assert "<script>" in normalize_target("/", "q=%25253Cscript%25253E")
    assert normalize_target("/\uff0fadmin", "").endswith("/admin")
    assert "/a+b" in normalize_target("/a+b", "")


def test_query_duplicates_and_cookie_features():
    fv = ProductionHTTPFeatureExtractor().extract(req(query="a=1&a=2&b="))
    assert fv.values["query_param_count"] > 0
    assert fv.values["unique_query_key_count"] > 0
    assert fv.values["duplicate_query_key_count"] > 0
    assert fv.values["cookie_count"] > 0


def test_content_length_mismatch_and_malformed_percent():
    fv = ProductionHTTPFeatureExtractor().extract(req(query="x=%GG", headers={"Content-Length": "1"}))
    assert fv.values["content_length_mismatch"] == 1.0
    assert fv.values["malformed_percent_flag"] == 1.0


def test_content_type_flags_and_https():
    fv = ProductionHTTPFeatureExtractor().extract(req(headers={"content-type": "application/json; charset=utf-8"}))
    assert fv.values["has_json_body"] == 1.0
    assert fv.values["scheme_https"] == 1.0


def test_invalid_utf8_and_control_bytes_are_bounded():
    body = b"abc\xff\xfe\x00\x01" * 100
    fv = ProductionHTTPFeatureExtractor().extract(req(body=body))
    assert fv.values["body_utf8_replacement_ratio"] > 0.0
    assert fv.values["null_byte_flag"] == 1.0
    assert fv.values["control_char_ratio"] > 0.0


def test_feature_output_does_not_store_raw_payload():
    fv = ProductionHTTPFeatureExtractor().extract(req(query="token=super-secret-value", body=b"api-key=secret"))
    assert "super-secret-value" not in str(fv.values)
    assert "api-key=secret" not in str(fv.values)


def test_determinism():
    extractor = ProductionHTTPFeatureExtractor()
    assert extractor.extract(req()) == extractor.extract(req())


def test_large_input_is_bounded_without_error():
    path = "/" + "A" * 100_000
    query = "q=" + "B" * 100_000
    body = b"C" * 400_000
    fv = ProductionHTTPFeatureExtractor().extract(req(path=path, query=query, body=body))
    assert all(0.0 <= v <= 1.0 for v in fv.values.values())


def test_header_processing_is_bounded():
    headers = {f"X-{i}": "v" for i in range(1000)}
    fv = ProductionHTTPFeatureExtractor().extract(req(headers=headers))
    assert fv.values["header_count"] == 1.0


def test_unknown_method_is_safe():
    fv = ProductionHTTPFeatureExtractor().extract(req(method="UNKNOWN"))
    assert fv.values["method_known"] == 0.0
    assert fv.values["method_code"] == 0.0
