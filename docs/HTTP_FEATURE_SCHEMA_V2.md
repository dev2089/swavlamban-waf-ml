# HTTP Feature Schema v2

The active WAF feature pipeline emits a numeric-only `http-v2` vector. Every value is normalized to `[0,1]`.

## Features

`method_code`, `method_known`, `scheme_https`, `host_length`, `path_length`, `normalized_path_length`, `query_length`, `body_length`, `header_count`, `header_bytes`, `query_param_count`, `unique_query_key_count`, `duplicate_query_key_count`, `cookie_count`, `has_json_body`, `has_form_body`, `has_xml_body`, `has_multipart_body`, `has_content_length`, `content_length_mismatch`, `percent_encoded_ratio`, `malformed_percent_flag`, `double_encoded_flag`, `null_byte_flag`, `control_char_ratio`, `path_entropy`, `query_entropy`, `body_entropy`, `target_entropy`, `special_char_ratio`, `digit_ratio`, `alpha_ratio`, `has_sql_keyword`, `has_xss_token`, `has_traversal`, `has_command_token`, `query_key_entropy`, `body_utf8_replacement_ratio`.

## Processing guarantees

- Path and query use different URL decoding semantics so literal `+` in a path is not converted to a space.
- URL decoding is capped at three passes.
- Unicode is normalized with NFKC.
- Query parsing is capped at 256 fields.
- Header names are normalized and header values are capped at 4 KiB each.
- Body inspection is capped at 256 KiB while request size remains represented by the bounded size feature.
- No raw request payload is persisted in the feature vector.
- All outputs are numeric, deterministic and clamped to `[0,1]`.
- Feature characteristics are not probabilities or calibrated confidence values.
