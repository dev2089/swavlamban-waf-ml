from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
import random
from threading import Lock
import time

import numpy as np
from sklearn.linear_model import LogisticRegression

from waf.core.models import DetectionSignal, FeatureVector, RequestEnvelope


@dataclass(slots=True)
class _SourceWindow:
    events: deque[tuple[float, str]] = field(default_factory=deque)


@dataclass(slots=True)
class BehaviouralDetector:
    """Stateful learned detector for request-burst and endpoint-churn behaviour."""

    model: LogisticRegression | None = None
    window_seconds: float = 10.0
    max_events_per_source: int = 128
    name: str = "behaviour-v1"
    _sources: dict[str, _SourceWindow] = field(init=False, repr=False)
    _lock: Lock = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.model is None:
            self.model = self.train_default().model
        self._sources = defaultdict(_SourceWindow)
        self._lock = Lock()

    @classmethod
    def train_default(cls, seed: int = 42) -> "BehaviouralDetector":
        rng = random.Random(seed)
        X: list[list[float]] = []
        y: list[int] = []
        for _ in range(1500):
            count = rng.randint(1, 12)
            unique = rng.randint(1, min(count, 3))
            span = rng.uniform(8.0, 12.0)
            mean_interval = rng.uniform(0.65, 1.8)
            interval_std = rng.uniform(0.0, 0.45)
            X.append(cls._summary_features(count, unique, span, mean_interval, interval_std))
            y.append(0)
        for _ in range(1500):
            count = rng.randint(28, 80)
            unique = rng.randint(6, min(count, 40))
            span = rng.uniform(0.2, 4.0)
            mean_interval = rng.uniform(0.01, 0.12)
            interval_std = rng.uniform(0.0, 0.18)
            X.append(cls._summary_features(count, unique, span, mean_interval, interval_std))
            y.append(1)
        model = LogisticRegression(C=3.0, max_iter=500, random_state=seed)
        model.fit(np.asarray(X, dtype=float), np.asarray(y, dtype=int))
        return cls(model)

    @staticmethod
    def _summary_features(
        count: int,
        unique_paths: int,
        span_seconds: float,
        mean_interval: float,
        interval_std: float,
    ) -> list[float]:
        rate = count / 10.0
        unique_ratio = unique_paths / max(1, count)
        interval_cv = interval_std / max(0.001, mean_interval)
        return [
            min(1.0, count / 128.0),
            min(1.0, unique_paths / 64.0),
            min(1.0, rate / 12.8),
            min(1.0, unique_ratio),
            min(1.0, span_seconds / 10.0),
            min(1.0, mean_interval / 2.0),
            min(1.0, interval_cv),
        ]

    def _window_features(self, events: deque[tuple[float, str]]) -> list[float]:
        count = len(events)
        if count <= 1:
            span = self.window_seconds
            mean_interval = self.window_seconds
            interval_std = 0.0
        else:
            times = [timestamp for timestamp, _ in events]
            intervals = np.diff(np.asarray(times, dtype=float))
            span = max(0.001, times[-1] - times[0])
            mean_interval = float(np.mean(intervals))
            interval_std = float(np.std(intervals))
        unique_paths = len({path for _, path in events})
        return self._summary_features(count, unique_paths, span, mean_interval, interval_std)

    def __getstate__(self):
        return {
            "model": self.model,
            "window_seconds": self.window_seconds,
            "max_events_per_source": self.max_events_per_source,
            "name": self.name,
        }

    def __setstate__(self, state):
        self.model = state.get("model")
        self.window_seconds = state.get("window_seconds", 10.0)
        self.max_events_per_source = state.get("max_events_per_source", 128)
        self.name = state.get("name", "behaviour-v1")
        self._sources = defaultdict(_SourceWindow)
        self._lock = Lock()

    def detect(self, request: RequestEnvelope, features: FeatureVector) -> DetectionSignal:
        now = float(request.timestamp or time.time())
        source = request.source_ip or "unknown"
        with self._lock:
            window = self._sources[source]
            cutoff = now - self.window_seconds
            while window.events and window.events[0][0] < cutoff:
                window.events.popleft()
            window.events.append((now, request.path))
            while len(window.events) > self.max_events_per_source:
                window.events.popleft()
            count = len(window.events)
            unique_paths = len({path for _, path in window.events})
            summary = self._window_features(window.events)
        assert self.model is not None
        score = float(self.model.predict_proba(np.asarray([summary], dtype=float))[0, 1])
        confidence = min(1.0, abs(score - 0.5) * 2.0)
        if score >= 0.5:
            reasons = (f"learned behavioural anomaly: {count} requests/{self.window_seconds:g}s, {unique_paths} unique paths",)
        else:
            reasons = ("learned behaviour within current baseline window",)
        return DetectionSignal(
            detector=self.name,
            score=round(max(0.0, min(1.0, score)), 6),
            confidence=round(confidence, 6),
            reasons=reasons,
            metadata={
                "window_requests": count,
                "unique_paths": unique_paths,
                "window_seconds": self.window_seconds,
                "model_type": "logistic_regression",
            },
        )
