"""Lightweight RSS / tracemalloc probes for score-path investigation."""

from __future__ import annotations

import gc
import os
import threading
import time
import tracemalloc
from dataclasses import dataclass, field


def rss_mb() -> float:
    """Current process RSS in MiB."""
    try:
        import psutil

        return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    except Exception:  # noqa: BLE001
        # Fallback: macOS/Linux ps
        import subprocess

        out = subprocess.check_output(["ps", "-o", "rss=", "-p", str(os.getpid())], text=True)
        return float(out.strip() or "0") / 1024.0


@dataclass
class MemSample:
    label: str
    rss_mb: float
    delta_from_prev_mb: float
    delta_from_baseline_mb: float
    tracemalloc_current_mb: float | None = None
    tracemalloc_peak_mb: float | None = None
    t_s: float = 0.0


@dataclass
class MemTrace:
    """Ordered stage snapshots + background peak sampler."""

    baseline_rss_mb: float = 0.0
    samples: list[MemSample] = field(default_factory=list)
    peak_rss_mb: float = 0.0
    peak_label: str = "baseline"
    _prev_rss: float = 0.0
    _stop: bool = False
    _thread: threading.Thread | None = None
    _use_tracemalloc: bool = False

    def start(self, use_tracemalloc: bool = True) -> None:
        gc.collect()
        self._use_tracemalloc = use_tracemalloc
        if use_tracemalloc and not tracemalloc.is_tracing():
            tracemalloc.start(25)
        self.baseline_rss_mb = rss_mb()
        self._prev_rss = self.baseline_rss_mb
        self.peak_rss_mb = self.baseline_rss_mb
        self.peak_label = "baseline"
        self._stop = False
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()
        self.mark("baseline_after_warm")

    def _sample_loop(self) -> None:
        while not self._stop:
            r = rss_mb()
            if r > self.peak_rss_mb:
                self.peak_rss_mb = r
                self.peak_label = "background_peak"
            time.sleep(0.02)

    def mark(self, label: str) -> MemSample:
        gc.collect()
        r = rss_mb()
        tm_cur = tm_peak = None
        if self._use_tracemalloc and tracemalloc.is_tracing():
            cur, peak = tracemalloc.get_traced_memory()
            tm_cur = cur / (1024 * 1024)
            tm_peak = peak / (1024 * 1024)
        sample = MemSample(
            label=label,
            rss_mb=r,
            delta_from_prev_mb=r - self._prev_rss,
            delta_from_baseline_mb=r - self.baseline_rss_mb,
            tracemalloc_current_mb=tm_cur,
            tracemalloc_peak_mb=tm_peak,
            t_s=time.perf_counter(),
        )
        self.samples.append(sample)
        self._prev_rss = r
        if r >= self.peak_rss_mb:
            self.peak_rss_mb = r
            self.peak_label = label
        return sample

    def stop(self) -> None:
        self._stop = True
        if self._thread:
            self._thread.join(timeout=2)
        self.mark("final")
        if self._use_tracemalloc and tracemalloc.is_tracing():
            tracemalloc.stop()

    def report(self) -> str:
        lines = [
            f"baseline_rss_mb={self.baseline_rss_mb:.1f}",
            f"peak_rss_mb={self.peak_rss_mb:.1f}  (at: {self.peak_label})",
            "",
            f"{'stage':<36} {'rss':>8} {'d_prev':>8} {'d_base':>8} {'tm_cur':>8} {'tm_peak':>8}",
        ]
        t0 = self.samples[0].t_s if self.samples else 0.0
        for s in self.samples:
            tm_c = f"{s.tracemalloc_current_mb:.1f}" if s.tracemalloc_current_mb is not None else "-"
            tm_p = f"{s.tracemalloc_peak_mb:.1f}" if s.tracemalloc_peak_mb is not None else "-"
            lines.append(
                f"{s.label:<36} {s.rss_mb:8.1f} {s.delta_from_prev_mb:8.1f} "
                f"{s.delta_from_baseline_mb:8.1f} {tm_c:>8} {tm_p:>8}  (+{s.t_s - t0:.2f}s)"
            )
        # Top deltas
        ranked = sorted(self.samples[1:], key=lambda x: x.delta_from_prev_mb, reverse=True)
        lines.append("")
        lines.append("largest stage deltas (RSS):")
        for s in ranked[:8]:
            lines.append(f"  {s.delta_from_prev_mb:+7.1f} MB  {s.label}")
        return "\n".join(lines)


class RequestMemTrace:
    """Per-request RSS snapshots for production /score debugging.

    Logs via print(flush=True) so Render captures lines even when app log level
    is unset; also emits logger.info for structured log drains.
    """

    def __init__(self, baseline_mb: float | None = None, request_id: str = "") -> None:
        self.request_id = request_id
        self.baseline_mb = baseline_mb if baseline_mb is not None else rss_mb()
        self._prev_mb = self.baseline_mb
        self.peak_mb = self.baseline_mb
        self.peak_label = "request_start"
        self.samples: list[tuple[str, float, float, float]] = []

    def mark(self, label: str, logger=None, extra: str = "") -> float:
        gc.collect()
        r = rss_mb()
        d_prev = r - self._prev_mb
        d_base = r - self.baseline_mb
        self.samples.append((label, r, d_prev, d_base))
        suffix = f" {extra}" if extra else ""
        msg = (
            f"score_mem {label} rss_mb={r:.1f} d_prev={d_prev:+.1f} "
            f"d_base={d_base:+.1f}{suffix}"
        )
        if self.request_id:
            msg = f"[{self.request_id}] {msg}"
        print(msg, flush=True)
        if logger is not None:
            logger.info(msg)
        self._prev_mb = r
        if r >= self.peak_mb:
            self.peak_mb = r
            self.peak_label = label
        return r

    def summary(self, logger=None) -> str:
        ranked = sorted(self.samples[1:], key=lambda x: x[2], reverse=True)
        lines = [
            f"score_mem_summary peak_rss_mb={self.peak_mb:.1f} at={self.peak_label} "
            f"baseline={self.baseline_mb:.1f}",
            "score_mem_summary largest_d_prev:",
        ]
        for label, _r, d_prev, _d_base in ranked[:8]:
            lines.append(f"  {d_prev:+7.1f} MB  {label}")
        text = "\n".join(lines)
        print(text, flush=True)
        if logger is not None:
            logger.info(text)
        return text
