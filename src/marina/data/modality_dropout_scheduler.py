from __future__ import annotations

from collections import Counter
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Set
import math
import random

import matplotlib.pyplot as plt
import numpy as np


# ---------------------- Core bit-pattern utilities ----------------------

def _pattern_tuple_from_entry(entry: Mapping[str, bool], keys: Sequence[str]) -> Tuple[int, ...]:
    """Convert {'has_x':bool,...} → tuple of 0/1 in given key order."""
    return tuple(1 if bool(entry[k]) else 0 for k in keys)

def _all_patterns(n: int) -> List[Tuple[int, ...]]:
    return [tuple((mask >> i) & 1 for i in range(n)) for mask in range(1 << n)]

def _popcount(tup: Tuple[int, ...]) -> int:
    return sum(tup)

def _all_subsets_of(a_tuple: Tuple[int, ...], exclude_empty: bool = True) -> Iterable[Tuple[int, ...]]:
    """Yield all t ⊆ a (same length), optionally excluding the empty set."""
    idxs = [i for i, v in enumerate(a_tuple) if v]
    m = len(idxs)
    start = 1 if exclude_empty else 0
    for mask in range(start, 1 << m):
        t = [0] * len(a_tuple)
        for j in range(m):
            if (mask >> j) & 1:
                t[idxs[j]] = 1
        yield tuple(t)

def _labels_from_tuple(tup: Tuple[int, ...], keys: Sequence[str]) -> List[str]:
    """('has_hsqc','has_mass_spec',...) + (1,0,1,...) → ['hsqc','c_nmr', ...]."""
    return [keys[i].replace("has_", "") for i, b in enumerate(tup) if b]


# ---------------------- Matrix balancing (IPF/RAS) ----------------------

def _ipf_with_mask(
    row_sums: Dict[Tuple[int, ...], float],
    col_sums: Dict[Tuple[int, ...], float],
    allowed: Set[Tuple[Tuple[int, ...], Tuple[int, ...]]],
    max_iter: int = 5000,
    tol: float = 1e-9,
) -> Dict[Tuple[Tuple[int, ...], Tuple[int, ...]], float]:
    rows = list(row_sums.keys())
    cols = list(col_sums.keys())
    X: Dict[Tuple[Tuple[int, ...], Tuple[int, ...]], float] = {}

    # init: spread row mass uniformly over allowed columns
    for a in rows:
        allowed_t = [t for t in cols if (a, t) in allowed]
        if not allowed_t or row_sums[a] == 0:
            for t in cols:
                X[(a, t)] = 0.0
            continue
        init = row_sums[a] / len(allowed_t)
        for t in cols:
            X[(a, t)] = init if (a, t) in allowed else 0.0

    for _ in range(max_iter):
        # Row scaling
        for a in rows:
            current = sum(X[(a, t)] for t in cols)
            target = row_sums[a]
            if current > 0:
                scale = target / current
                for t in cols:
                    if (a, t) in allowed:
                        X[(a, t)] *= scale

        # Column scaling + convergence check
        max_diff = 0.0
        for t in cols:
            current = sum(X[(a, t)] for a in rows)
            target = col_sums[t]
            if current > 0:
                scale = target / current
                for a in rows:
                    if (a, t) in allowed:
                        X[(a, t)] *= scale
                max_diff = max(max_diff, abs(target - sum(X[(a, t)] for a in rows)))
            else:
                # infeasible column; leave at 0
                max_diff = max(max_diff, target)

        if max_diff < tol:
            break

    return X


# ---------------------- Column target builders ----------------------

def _row_sums_from_dataset(dataset_entries: Sequence, keys: Sequence[str]) -> Tuple[Dict[Tuple[int, ...], float], List[Tuple[int, ...]]]:
    """dataset_entries is either [entry, ...] or [(idx, entry), ...]."""
    def _entry(x):
        return x[1] if (isinstance(x, (tuple, list)) and len(x) == 2 and isinstance(x[1], dict)) else x

    counts = Counter(_pattern_tuple_from_entry(_entry(x), keys) for x in dataset_entries)
    patterns = _all_patterns(len(keys))
    total = sum(counts.values())
    row_sums = {a: counts.get(a, 0) / total for a in patterns}
    return row_sums, patterns

def _allowed_and_reachable(row_sums: Dict[Tuple[int, ...], float], keys: Sequence[str], enforce_constraints: bool) -> Tuple[Set, Set]:
    """Build structural mask and set of reachable target columns."""
    patterns = list(row_sums.keys())
    allowed: Set[Tuple[Tuple[int, ...], Tuple[int, ...]]] = set()
    reachable_cols: Set[Tuple[int, ...]] = set()

    for a in patterns:
        if row_sums[a] == 0:
            continue

        if enforce_constraints and _popcount(a) == 1:
            # singleton availability → must keep itself
            allowed.add((a, a))
            reachable_cols.add(a)
            continue

        # any non-empty subset allowed
        for t in _all_subsets_of(a, exclude_empty=True):
            allowed.add((a, t))
            reachable_cols.add(t)

    return allowed, reachable_cols

def _identity_col_sums(row_sums: Dict[Tuple[int, ...], float], patterns: List[Tuple[int, ...]]):
    """Column target that keeps 'max info' (no extra dropping)."""
    col_sums = {t: 0.0 for t in patterns}
    for a in patterns:
        col_sums[a] += row_sums[a]
    return col_sums

def _uniform_reachable_col_sums(reachable_cols: Set[Tuple[int, ...]], patterns: List[Tuple[int, ...]]):
    each = 1.0 / len(reachable_cols)
    return {t: (each if t in reachable_cols else 0.0) for t in patterns}

def _size_weighted_col_sums(reachable_cols: Set[Tuple[int, ...]], patterns: List[Tuple[int, ...]], beta: float):
    """Prefer smaller kept sets: weight(t) = exp(-beta * |t|). beta>=0."""
    weights = {}
    for t in patterns:
        if t in reachable_cols:
            k = _popcount(t)
            weights[t] = math.exp(-beta * k)
        else:
            weights[t] = 0.0
    Z = sum(weights.values())
    if Z <= 0:
        return _uniform_reachable_col_sums(reachable_cols, patterns)
    return {t: (weights[t] / Z) for t in patterns}

def _blend(a: Dict, b: Dict, alpha: float, patterns: List[Tuple[int, ...]]) -> Dict:
    """(1-alpha)*a + alpha*b"""
    return {t: (1.0 - alpha) * a.get(t, 0.0) + alpha * b.get(t, 0.0) for t in patterns}


# ---------------------- Solve for a given column target ----------------------

def _compute_probs_lookup_for_col_target(
    dataset_entries: Sequence,
    keys: Sequence[str],
    col_sums: Dict[Tuple[int, ...], float],
    enforce_constraints: bool,
) -> Tuple[Dict[Tuple[int, ...], Dict[Tuple[int, ...], float]], Dict]:
    row_sums, patterns = _row_sums_from_dataset(dataset_entries, keys)
    allowed, reachable_cols = _allowed_and_reachable(row_sums, keys, enforce_constraints=enforce_constraints)

    # Trim/renormalize col_sums to reachable columns
    trimmed = {t: (col_sums.get(t, 0.0) if t in reachable_cols else 0.0) for t in patterns}
    Z = sum(trimmed.values())
    if Z <= 0:
        trimmed = _uniform_reachable_col_sums(reachable_cols, patterns)
    else:
        trimmed = {t: v / Z for t, v in trimmed.items()}

    X = _ipf_with_mask(row_sums, trimmed, allowed)

    probs: Dict[Tuple[int, ...], Dict[Tuple[int, ...], float]] = {}
    for a in patterns:
        ra = row_sums[a]
        if ra <= 0:
            probs[a] = {a: 1.0}
            continue
        inner = {t: X[(a, t)] / ra for t in patterns if (a, t) in allowed}
        s = sum(inner.values())
        probs[a] = {t: (v / s) for t, v in inner.items()} if s > 0 else {a: 1.0}

    info = {
        "reachable_targets_count": len(reachable_cols),
        "keys_order": list(keys),
    }
    return probs, info


# ---------------------- Public Scheduler Class ----------------------

class ModalityDropoutScheduler:
    """
    Chooses kept-modality combinations for each example, with either:
      - mode="constant":   fixed uniform-over-reachable target (balanced)
      - mode="scheduled":  smoothly morph identity → uniform → sparse (no-empty, preserve singletons)

    Call `sample_kept_modalities(entry, requires=set(...))` to get a list like ['hsqc','h_nmr'].
    """

    def __init__(
        self,
        dataset_entries: Sequence,
        keys: Sequence[str] = ('has_hsqc', 'has_mass_spec', 'has_c_nmr', 'has_h_nmr'),
        mode: str = "constant",
        total_phases: float = 2.0,   # used for "scheduled": 0..1 (identity->uniform), 1..2 (uniform->sparser)
        beta_end: float = 1.5,       # target sparsity at phase==2
        enforce_constraints: bool = True,
        rng: Optional[random.Random] = None,
    ):
        self.keys = tuple(keys)
        self.n = len(keys)
        self.patterns = _all_patterns(self.n)
        self.mode = mode
        self.total_phases = float(total_phases)
        self.beta_end = float(beta_end)
        self.enforce_constraints = enforce_constraints
        self._rng = rng or random.Random()

        # Precompute distributions we might need across the schedule
        self._row_sums, _ = _row_sums_from_dataset(dataset_entries, self.keys)
        self._allowed, self._reachable_cols = _allowed_and_reachable(self._row_sums, self.keys, enforce_constraints=enforce_constraints)
        self._col_identity = _identity_col_sums(self._row_sums, self.patterns)
        self._col_uniform = _uniform_reachable_col_sums(self._reachable_cols, self.patterns)

        # Precompute constant lookup if needed
        if self.mode == "constant":
            self._probs_lookup, _ = _compute_probs_lookup_for_col_target(
                dataset_entries, self.keys, self._col_uniform, enforce_constraints=self.enforce_constraints
            )
        else:
            self._probs_lookup = None  # generated on demand per phase

        # Cache for scheduled lookups keyed by rounded phase
        self._phase_cache: dict[float, dict] = {}
        self._current_phase: float = 0.0
        self._current_probs = None  # lazily computed for scheduled mode

        # Keep a reference to the dataset to support plotting expected distributions
        self._dataset_entries = dataset_entries

    # ---------- Phase handling for scheduled mode ----------

    def _col_target_at_phase(self, phase: float) -> Dict[Tuple[int, ...], float]:
        """phase ∈ [0, total_phases]"""
        phase = max(0.0, min(self.total_phases, float(phase)))
        if phase <= 1.0:
            # linear: identity -> uniform
            alpha = phase  # 0..1
            return _blend(self._col_identity, self._col_uniform, alpha, self.patterns)
        else:
            # uniform -> size-weighted sparse
            seg = min(1.0, phase - 1.0)  # 0..1
            beta = seg * self.beta_end
            col_sparse = _size_weighted_col_sums(self._reachable_cols, self.patterns, beta=beta)
            return _blend(self._col_uniform, col_sparse, seg, self.patterns)

    def _compute_and_cache_phase(self, phase: float, key_precision: int = 3):
        """Compute probs for a given phase and cache it. Returns probs_lookup."""
        key = round(float(phase), key_precision)
        if key in self._phase_cache:
            return self._phase_cache[key]
        col = self._col_target_at_phase(key)
        probs, _ = _compute_probs_lookup_for_col_target(
            self._dataset_entries, self.keys, col, enforce_constraints=self.enforce_constraints
        )
        self._phase_cache[key] = probs
        return probs

    def set_phase(self, phase: float):
        """
        Set the active phase (e.g., once per epoch). Recomputes (or fetches cached)
        probs_lookup for scheduled mode; no-op for constant mode.
        """
        self._current_phase = max(0.0, min(self.total_phases, float(phase)))
        if self.mode == "constant":
            return
        self._current_probs = self._compute_and_cache_phase(self._current_phase)

    def probs_lookup(self, phase: Optional[float] = None) -> Dict[Tuple[int, ...], Dict[Tuple[int, ...], float]]:
        """
        If 'phase' is provided (scheduled mode), returns/creates a cached lookup for that phase.
        If 'phase' is None, returns the scheduler's current cached lookup (set via set_phase()).
        In constant mode, always returns the fixed lookup.
        """
        if self.mode == "constant":
            return self._probs_lookup
        if phase is None:
            if self._current_probs is None:
                self._current_probs = self._compute_and_cache_phase(self._current_phase)
            return self._current_probs
        return self._compute_and_cache_phase(phase)

    # ---------- Sampling for a single entry ----------

    def _sample_target_from_availability(self, a: Tuple[int, ...], probs_for_a: Mapping[Tuple[int, ...], float]) -> Tuple[int, ...]:
        r, acc = self._rng.random(), 0.0
        items = list(probs_for_a.items())
        for t, p in items:
            acc += p
            if r <= acc:
                return t
        return items[-1][0] if items else a

    def sample_kept_modalities(
        self,
        entry: Mapping[str, bool],
        requires: Optional[Set[str]] = None,
        phase: Optional[float] = None,
    ) -> List[str]:
        """
        Given one dataset entry with 'has_*' flags, choose a kept-combo (list of names without 'has_').
        'requires' modalities are always kept if available.
        """
        requires = set(requires or set())

        a = _pattern_tuple_from_entry(entry, self.keys)
        probs = self.probs_lookup(phase)
        pa = probs.get(a)
        if not pa:
            kept = _labels_from_tuple(a, self.keys)
            return kept

        t = self._sample_target_from_availability(a, pa)
        kept = set(_labels_from_tuple(t, self.keys))

        # Always keep required modalities if they're available
        for name in list(requires):
            has_key = f"has_{name}"
            if entry.get(has_key, False):
                kept.add(name)

        # Defensive constraints (should already hold):
        if not kept:
            kept = set(_labels_from_tuple(a, self.keys))
        if _popcount(a) == 1:
            kept = set(_labels_from_tuple(a, self.keys))

        return sorted(kept)

    # ---------- Distribution plotting (save to file) ----------

    def _observed_distribution(self) -> Dict[Tuple[int, ...], float]:
        counts = Counter(
            _pattern_tuple_from_entry(x[1] if (isinstance(x, (tuple, list)) and len(x) == 2 and isinstance(x[1], dict)) else x, self.keys)
            for x in self._dataset_entries
        )
        total = sum(counts.values()) or 1
        return {a: counts.get(a, 0) / total for a in self.patterns}

    def _expected_distribution(self, probs_lookup: Mapping[Tuple[int, ...], Mapping[Tuple[int, ...], float]]) -> Dict[Tuple[int, ...], float]:
        observed = self._observed_distribution()
        m = {t: 0.0 for t in self.patterns}
        for a in self.patterns:
            pa = probs_lookup.get(a, {})
            for t, p in pa.items():
                m[t] += observed.get(a, 0.0) * p
        return m

    def save_combo_distributions_plot(self, filename: str, phase: Optional[float] = None, only_reachable: bool = True):
        """Save a bar chart comparing observed availability vs expected post-drop distribution."""
        probs = self.probs_lookup(phase)
        observed = self._observed_distribution()
        expected = self._expected_distribution(probs)

        patterns = list(self.patterns)
        if only_reachable:
            reachable = set()
            for a, inner in probs.items():
                reachable.update(inner.keys())
            patterns = [p for p in patterns if p in reachable]

        labels = ["+".join(_labels_from_tuple(p, self.keys)) if _popcount(p) > 0 else "∅" for p in patterns]
        x = np.arange(len(patterns))
        width = 0.45

        plt.figure(figsize=(max(10, len(patterns) * 0.5), 5))
        plt.bar(x - width / 2, [observed.get(p, 0.0) for p in patterns], width, label="Observed availability")
        plt.bar(x + width / 2, [expected.get(p, 0.0) for p in patterns], width, label="Expected after dropping")
        plt.xticks(x, labels, rotation=90)
        plt.ylabel("Fraction")
        plt.title("Distribution of modality combinations (observed vs. expected)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()
        print(f"[ModalityDropoutScheduler] Saved plot to {filename}")

    # ---------- Log-friendly helpers ----------

    def _label_for(self, tup: Tuple[int, ...]) -> str:
        names = [k.replace("has_", "") for k, v in zip(self.keys, tup) if v]
        return "+".join(names) if names else "∅"

    def expected_target_marginal(self, phase: Optional[float] = None, labeled: bool = False) -> Dict:
        """
        Return expected target (kept-combo) marginal p[t] for the given phase:
            p[t] = sum_a row_sums[a] * p(t | a, phase)
        Uses the scheduler's cached current phase if phase=None.
        """
        probs = self.probs_lookup(phase)  # p(t|a)
        row = self._row_sums              # p(a)
        out: Dict[Tuple[int, ...], float] = {t: 0.0 for t in self.patterns}
        for a, pa in probs.items():
            ra = row.get(a, 0.0)
            if ra <= 0:
                continue
            for t, p in pa.items():
                out[t] += ra * p

        # drop strictly-zero entries (usually unreachable)
        out = {t: v for t, v in out.items() if v > 0}

        if labeled:
            return {self._label_for(t): v for t, v in out.items()}
        return out

    def observed_availability_marginal(self, labeled: bool = False) -> Dict:
        """
        Return observed availability p(a) computed from the dataset entries the scheduler was built on.
        """
        # reuse existing helper and reformat
        obs = self._observed_distribution()  # dict[a_tuple] -> fraction
        # drop zeros
        obs = {a: v for a, v in obs.items() if v > 0}
        if labeled:
            return {self._label_for(a): v for a, v in obs.items()}
        return obs