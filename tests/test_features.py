from __future__ import annotations

import unittest

from waf.core.models import RequestEnvelope
from waf.features.http import HTTPFeatureExtractor


class FeatureTests(unittest.TestCase):
    def setUp(self):
        self.extractor = HTTPFeatureExtractor()

    def req(self, **kwargs):
        data = dict(
            request_id="f-1",
            method="GET",
            scheme="https",
            host="example.test",
            path="/",
        )
        data.update(kwargs)
        return RequestEnvelope(**data)

    def test_schema_and_normalized_ranges(self):
        fv = self.extractor.extract(self.req())
        self.assertEqual(fv.schema_version, "http-v2")
        self.assertGreaterEqual(len(fv.values), 40)
        for value in fv.values.values():
            self.assertGreaterEqual(value, 0.0)
            self.assertLessEqual(value, 1.0)

    def test_known_payload_indicators(self):
        fv = self.extractor.extract(
            self.req(
                query="q=1 union select x from y",
                body=b"<script>alert(1)</script>",
            )
        )
        self.assertEqual(fv.values["has_sql_keyword"], 1.0)
        self.assertEqual(fv.values["has_xss_token"], 1.0)

    def test_unicode_and_invalid_utf8_are_safe(self):
        fv = self.extractor.extract(self.req(body=b"hello\xff\xfe\xe2\x82\xac"))
        self.assertIn("body_length", fv.values)
        self.assertGreater(fv.values["body_utf8_replacement_ratio"], 0.0)

    def test_path_plus_is_not_decoded_as_space(self):
        fv = self.extractor.extract(self.req(path="/a+b"))
        self.assertGreater(fv.values["path_length"], 0.0)

    def test_content_type_flags(self):
        fv = self.extractor.extract(
            self.req(headers={"content-type": "application/json; charset=utf-8"})
        )
        self.assertEqual(fv.values["has_json_body"], 1.0)

    def test_duplicate_query_and_cookie_features(self):
        fv = self.extractor.extract(
            self.req(query="a=1&a=2&b=", headers={"cookie": "a=1; b=2"})
        )
        self.assertGreater(fv.values["duplicate_query_key_count"], 0.0)
        self.assertGreater(fv.values["cookie_count"], 0.0)

    def test_malformed_and_double_encoded_flags(self):
        fv = self.extractor.extract(
            self.req(query="x=%GG&y=%2525253Cscript%2525253E")
        )
        self.assertEqual(fv.values["malformed_percent_flag"], 1.0)
        self.assertEqual(fv.values["double_encoded_flag"], 1.0)
        self.assertEqual(fv.values["has_xss_token"], 1.0)

    def test_query_field_overflow_is_flagged(self):
        query = "&".join(f"k{i}=1" for i in range(300))
        fv = self.extractor.extract(self.req(query=query))
        self.assertEqual(fv.values["query_parse_overflow"], 1.0)

    def test_body_and_header_bounds(self):
        headers = {f"X-{i}": "v" * 100 for i in range(1000)}
        body = b"C" * 400_000
        fv = self.extractor.extract(self.req(headers=headers, body=body))
        self.assertEqual(fv.values["header_count"], 1.0)
        self.assertGreater(fv.values["body_length"], 0.0)

    def test_no_raw_payload_in_feature_vector(self):
        fv = self.extractor.extract(
            self.req(query="token=super-secret-value", body=b"api-key=secret")
        )
        rendered = str(fv.values)
        self.assertNotIn("super-secret-value", rendered)
        self.assertNotIn("api-key=secret", rendered)

    def test_determinism(self):
        assert self.extractor.extract(self.req()) == self.extractor.extract(self.req())

    def test_unknown_method_is_safe(self):
        fv = self.extractor.extract(self.req(method="UNKNOWN"))
        self.assertEqual(fv.values["method_known"], 0.0)
        self.assertEqual(fv.values["method_code"], 0.0)


if __name__ == "__main__":
    unittest.main()
