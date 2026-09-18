# HTTP Feature Schema v2

The active Phase 3 WAF feature pipeline emits a deterministic numeric-only http-v2 vector. Every feature is clamped to [0,1].

## Feature contract

| Feature | Meaning |
|---|---|
| method_code | Normalized HTTP method ordinal |
| method_known | Whether the method is in the supported method set |
| scheme_https | HTTPS transport indicator |
| host_length | Bounded host length |
| path_length | Bounded raw path length |
| normalized_path_length | Bounded multi-decoded path length |
| query_length | Bounded raw query length |
| body_length | Bounded original request-body length |
| header_count | Bounded count of normalized headers |
| header_bytes | Bounded normalized header key/value size |
| header_value_bytes | Bounded aggregate normalized header-value size |
| query_param_count | Parsed query parameter count |
| unique_query_key_count | Unique query key count |
| duplicate_query_key_count | Repeated query key count |
| query_parse_overflow | Whether the query exceeded safe parser field limits |
| cookie_count | Cookie pair count |
| has_json_body | JSON content type indicator |
| has_form_body | Form URL-encoded content type indicator |
| has_xml_body | XML content type indicator |
| has_multipart_body | Multipart content type indicator |
| has_content_length | Content-Length presence |
| content_length_mismatch | Declared vs observed body length mismatch |
| percent_encoded_ratio | Percent-encoded token density in path/query |
| malformed_percent_flag | Malformed percent-encoding indicator |
| double_encoded_flag | Double-encoding marker |
| null_byte_flag | NUL-byte indicator |
| control_char_ratio | Control-character density |
| path_entropy | Normalized path entropy |
| query_entropy | Decoded query entropy |
| body_entropy | Bounded scanned body entropy |
| target_entropy | Combined normalized target/body entropy |
| special_char_ratio | Special-character density |
| digit_ratio | Digit density |
| alpha_ratio | Alphabetic density |
| has_sql_keyword | SQL-like indicator |
| has_xss_token | XSS-like indicator |
| has_traversal | Traversal indicator |
| has_command_token | Command-like indicator |
| query_key_entropy | Query-key entropy |
| body_utf8_replacement_ratio | Invalid-UTF8 replacement density |

## Processing guarantees

- Path uses URL decoding that preserves literal plus.
- Query uses form-style plus decoding.
- Path/query decoding is capped at three passes.
- Unicode is normalized with NFKC.
- Query parsing is capped at 256 fields and overflow is explicitly represented.
- Header processing is capped at 128 normalized headers and 4 KiB per header value.
- Feature header-size accounting is capped.
- Body inspection is capped at 256 KiB while total body length remains represented.
- No raw request payload is stored in the FeatureVector.
- Values are numeric, deterministic and clamped to [0,1].
- Feature values are characteristics, not probabilities and not calibrated confidence.
- Feature extraction performs no database or network I/O.

## Versioning

http-v2 is an explicit schema boundary. Changing feature semantics, names, ranges or normalization behavior requires a new schema version and corresponding model/evaluation compatibility checks.