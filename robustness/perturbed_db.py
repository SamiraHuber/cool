"""Foreground (in-memory) perturbation of perception query results.

This module intercepts the DB reads between the chat-agent tools and
PostgreSQL, so robustness experiments never modify the ground-truth
database. It wraps ``psycopg2`` connections/cursors and rewrites the
*result rows* on the fly, according to a :class:`PerturbationConfig`.

Usage (before importing/running the agent)::

    from agent import tools as agent_tools          # bordsupr/frontend/agent
    from robustness.perturbed_db import PerturbationConfig, install

    install(PerturbationConfig(kind="dropout", rate=0.2, seed=1), agent_tools)

Every call to ``agent_tools._get_conn()`` then returns a wrapped
connection whose query results are perturbed. The database itself is
never written to.

Supported perturbation kinds (mirror the robustness-study table):

- ``none``        : pass-through baseline.
- ``dropout``     : randomly drop ``rate`` of rows from perception tables
                    (object_observations, interactions, face_observations).
- ``fp_inject``   : duplicate ``rate`` of rows with the identity column
                    (person_id / subject_id / object_id) swapped to another
                    id seen during the run -> false-positive sightings.
- ``emb_noise``   : add Gaussian noise with std ``sigma`` to every column
                    whose name contains "embedding".
- ``track_frag``  : for ``rate`` of object identities, split the track:
                    later observations get mapped to a synthetic object_id.
- ``face_occl``   : for ``rate`` of face_observations rows, blank the image
                    byte columns (face occluded); rows without image bytes
                    are dropped instead.

Limitations (by design, to stay foreground-only):
- Aggregate queries (COUNT/SUM/...) are not rewritten, so counts reflect
  the clean DB while row listings are perturbed.
- Rows fetched via ``fetchone()`` are never *dropped* (callers often
  dereference them unconditionally); value-level perturbations still apply.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from typing import Any

TABLE_RE = re.compile(r'\b(?:from|join)\s+"?([a-zA-Z_][a-zA-Z0-9_]*)"?', re.IGNORECASE)
AGG_RE = re.compile(r"\b(count|sum|avg|min|max)\s*\(", re.IGNORECASE)

#: Tables that carry perception output and are affected by row perturbations.
OBS_TABLES = {"object_observations", "interactions", "face_observations"}

#: Base for synthetic ids created by track fragmentation / fp injection.
SYNTH_ID_BASE = 10_000_000

KINDS = (
    "none",
    "dropout",
    "fp_inject",
    "emb_noise",
    "track_frag",
    "face_occl",
    "temporal",
    "action_noise",
    "class_confuse",
    "identity_swap",
)


def _is_object_identity_col(col: str) -> bool:
    """True for result columns that carry a *perception object/person identity*.

    The agent tools expose identity under several aliases depending on the join:
    ``object_id`` (objects/observations), ``subject_id``/``object_id`` (raw
    interaction FKs), ``person_id`` (face/body observations) and the interaction
    aliases ``subject_object_id`` / ``object_object_id``. Matching by suffix keeps
    track-fragmentation and false-positive injection working across all of these
    instead of only the exact ``object_id`` name (which made them no-ops on the
    interaction/ownership query path).
    """
    c = col.lower()
    return (
        c == "object_id"
        or c == "subject_id"
        or c == "person_id"
        or c.endswith("_object_id")   # subject_object_id, object_object_id
        or c.endswith("_person_id")
    )


def _is_timestamp_col(col: str) -> bool:
    """True for result columns carrying a timestamp the agent reasons over."""
    c = col.lower()
    return c.endswith("_at") or c in ("last_seen", "first_seen", "created_at")


def _is_action_col(col: str) -> bool:
    return col.lower() == "action"


def _is_class_id_col(col: str) -> bool:
    """True for class_id columns (subject/object aliases included).

    The tools derive the human-readable class_name in Python from class_id
    (``_class_name``), so perturbing the int class_id keeps id/name consistent
    without us having to touch class_name strings.
    """
    c = col.lower()
    return c == "class_id" or c.endswith("_class_id")


def _paired_name_col(id_col: str) -> str | None:
    """Return the display-name column paired with an identity column.

    The agent answers from *names*, not raw ids: the tools return e.g.
    ``subject_object_id`` next to a clean-JOINed ``subject_name``. If an
    identity perturbation only rewrites the id number, the name stays correct
    and the ablation cannot change the answer (observed: identity_swap /
    fp_inject / track_frag all scored exactly baseline). A real mislabeled
    tracker shows the wrong *name*, so whenever an id cell is rewritten the
    paired name cell must follow it (see ``_apply_name_fixup``).
    """
    c = id_col.lower()
    if c == "object_id":
        return "name"
    if c.endswith("_object_id"):
        return c[: -len("_object_id")] + "_name"
    if c.endswith("_person_id"):
        return c[: -len("_person_id")] + "_name"
    return None


#: Visually/semantically confusable YOLO class pairs (bidirectional). Person (0)
#: is deliberately excluded — confusing a person with an object would break the
#: ownership semantics the study is trying to measure rather than simulate a
#: realistic detector confusion.
CLASS_CONFUSION = {
    63: 66, 66: 63,   # laptop <-> keyboard
    57: 56, 56: 57,   # couch <-> chair
    57: 59,           # couch -> bed (extra; first match wins via rng below)
    26: 24, 24: 26,   # handbag <-> backpack
    28: 24,           # suitcase -> backpack
    27: 25, 25: 27,   # tie <-> umbrella
    45: 41, 41: 45,   # bowl <-> cup
    39: 41,           # bottle -> cup
    62: 63,           # tv -> laptop
    64: 66,           # mouse -> keyboard
}

#: Ownership-relevant actions seen in interaction captions; used as the swap pool
#: for action_noise so a perturbed action stays a plausible (but wrong) verb.
ACTION_SWAP_POOL = [
    "holding", "carrying", "using", "sitting on", "standing next to",
    "next to", "looking at", "touching", "placing", "picking up",
]


@dataclass
class PerturbationConfig:
    kind: str = "none"          # one of KINDS
    rate: float = 0.0           # fraction for dropout/fp_inject/track_frag/face_occl/action_noise/class_confuse/identity_swap
    sigma: float = 0.0          # Gaussian noise std for emb_noise
    seed: int = 0
    seconds: float = 0.0        # max absolute jitter (seconds) for temporal

    def severity_label(self) -> str:
        if self.kind == "none":
            return "---"
        if self.kind == "emb_noise":
            return f"sigma={self.sigma}"
        if self.kind == "temporal":
            return f"{self.seconds:.0f}s"
        return f"{self.rate:.0%}"


@dataclass
class _RunState:
    """Mutable state shared by all wrapped connections of one study run."""

    rng: random.Random
    id_pools: dict[str, set[int]] = field(default_factory=dict)
    id_names: dict[int, str | None] = field(default_factory=dict)  # objects.id -> name
    frag_map: dict[int, int | None] = field(default_factory=dict)
    frag_seen: dict[int, int] = field(default_factory=dict)
    swap_map: dict[int, int] = field(default_factory=dict)  # identity_swap bijection
    synth_counter: int = 0
    # Diagnostics: how many rows this run actually perturbed, per kind. Surfaced
    # in the study output so a silent no-op condition (e.g. emb_noise on a query
    # path that never selects embeddings) is visible instead of looking like a
    # real (but flat) result.
    perturbed_rows: int = 0
    queries_touched: int = 0

    def next_synth_id(self) -> int:
        self.synth_counter += 1
        return SYNTH_ID_BASE + self.synth_counter


# ---------------------------------------------------------------------------
# vector helpers (pgvector values arrive as str like '[0.1,0.2]' or sequences)
# ---------------------------------------------------------------------------

def _parse_vector(value: Any) -> list[float] | None:
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("[") and text.endswith("]"):
            try:
                return [float(x) for x in text[1:-1].split(",")]
            except ValueError:
                return None
        return None
    if isinstance(value, (list, tuple)):
        try:
            return [float(x) for x in value]
        except (TypeError, ValueError):
            return None
    return None


def _format_like(original: Any, vec: list[float]) -> Any:
    if isinstance(original, str):
        return "[" + ",".join(repr(v) for v in vec) + "]"
    if isinstance(original, tuple):
        return tuple(vec)
    return vec


def _add_noise(original: Any, sigma: float, rng: random.Random) -> Any:
    vec = _parse_vector(original)
    if vec is None:
        return original
    noisy = [v + rng.gauss(0.0, sigma) for v in vec]
    return _format_like(original, noisy)


# ---------------------------------------------------------------------------
# cursor / connection wrappers
# ---------------------------------------------------------------------------

class PerturbedCursor:
    def __init__(self, real, config: PerturbationConfig, state: _RunState):
        self._real = real
        self._config = config
        self._state = state
        self._sql: str = ""
        self._tables: set[str] = set()

    # -- passthrough protocol ----------------------------------------------
    def __enter__(self):
        self._real.__enter__()
        return self

    def __exit__(self, *exc):
        return self._real.__exit__(*exc)

    def __iter__(self):
        return iter(self.fetchall())

    def __getattr__(self, name):
        return getattr(self._real, name)

    @property
    def description(self):
        return self._real.description

    # -- execution -----------------------------------------------------------
    def execute(self, sql, params=None):
        self._sql = sql if isinstance(sql, str) else str(sql)
        self._tables = {t.lower() for t in TABLE_RE.findall(self._sql)}
        return self._real.execute(sql, params)

    def executemany(self, sql, seq):
        return self._real.executemany(sql, seq)

    # -- fetching --------------------------------------------------------------
    def fetchall(self):
        return self._perturb_rows(list(self._real.fetchall()), allow_drop=True)

    def fetchmany(self, size=None):
        rows = self._real.fetchmany(size) if size is not None else self._real.fetchmany()
        return self._perturb_rows(list(rows), allow_drop=True)

    def fetchone(self):
        row = self._real.fetchone()
        if row is None:
            return None
        perturbed = self._perturb_rows([row], allow_drop=False)
        return perturbed[0] if perturbed else row

    # -- perturbation core -----------------------------------------------------
    def _columns(self) -> list[str]:
        desc = self._real.description or []
        return [str(d[0]).lower() for d in desc]

    def _perturb_rows(self, rows: list, allow_drop: bool) -> list:
        cfg = self._config
        if cfg.kind == "none" or not rows:
            return rows
        if AGG_RE.search(self._sql):
            return rows
        cols = self._columns()
        if not cols:
            return rows

        before = len(rows)
        changed_cells = 0

        if cfg.kind == "emb_noise":
            emb_idx = [i for i, c in enumerate(cols) if "embedding" in c]
            if not emb_idx:
                return rows
            out = [
                tuple(
                    _add_noise(v, cfg.sigma, self._state.rng) if i in emb_idx else v
                    for i, v in enumerate(row)
                )
                for row in rows
            ]
            changed_cells = sum(1 for r in rows for i in emb_idx if _parse_vector(r[i]) is not None)
            self._record_effect(before, len(out), changed_cells)
            return out

        if cfg.kind == "track_frag":
            # Fire on any query that surfaces a perception object/person identity,
            # not only bare `object_observations.object_id`. The ownership path
            # reads interactions via the aliases subject_object_id/object_object_id,
            # so requiring the exact "object_id" column made this a no-op there.
            if not (self._tables & {"object_observations", "interactions"}):
                return rows
            id_idx = [i for i, c in enumerate(cols) if _is_object_identity_col(c)]
            if not id_idx:
                return rows
            out = [self._fragment_row(row, id_idx, cols) for row in rows]
            changed_cells = sum(1 for a, b in zip(rows, out) if a != b)
            self._record_effect(before, len(out), changed_cells)
            return out

        if cfg.kind == "face_occl":
            if "face_observations" not in self._tables:
                return rows
            out = self._occlude_faces(rows, allow_drop)
            changed_cells = sum(1 for a, b in zip(rows, out) if a != b)
            self._record_effect(before, len(out), changed_cells)
            return out

        if cfg.kind == "dropout":
            if not (self._tables & OBS_TABLES):
                return rows
            if not allow_drop:
                return rows
            rng = self._state.rng
            out = [row for row in rows if rng.random() >= cfg.rate]
            self._record_effect(before, len(out), before - len(out))
            return out

        if cfg.kind == "fp_inject":
            out, injected = self._inject_false_positives(rows, cols)
            self._record_effect(before, len(out), injected)
            return out

        if cfg.kind == "temporal":
            out, changed = self._jitter_timestamps(rows, cols)
            self._record_effect(before, len(out), changed)
            return out

        if cfg.kind == "action_noise":
            if "interactions" not in self._tables:
                return rows
            out, changed = self._noise_actions(rows, cols)
            self._record_effect(before, len(out), changed)
            return out

        if cfg.kind == "class_confuse":
            out, changed = self._confuse_classes(rows, cols)
            self._record_effect(before, len(out), changed)
            return out

        if cfg.kind == "identity_swap":
            if not (self._tables & {"object_observations", "interactions"}):
                return rows
            out, changed = self._swap_identities(rows, cols)
            self._record_effect(before, len(out), changed)
            return out

        return rows

    def _record_effect(self, rows_in: int, rows_out: int, changed: int) -> None:
        """Accumulate diagnostics so a no-op condition is visible in the output."""
        if changed > 0 or rows_out != rows_in:
            self._state.queries_touched += 1
            self._state.perturbed_rows += changed if changed > 0 else abs(rows_out - rows_in)

    # -- per-kind helpers --------------------------------------------------------
    def _apply_name_fixup(self, new_row: list, cols: list[str], id_idx: int, new_id) -> None:
        """After rewriting an identity cell to ``new_id``, rewrite the paired name
        cell to the name of ``new_id`` (``None`` for synthetic/unknown ids), so the
        text channel the LLM actually reads stays consistent with the perturbed id.
        """
        if not isinstance(new_id, int):
            return
        name_col = _paired_name_col(cols[id_idx])
        if name_col is None or name_col not in cols:
            return
        new_row[cols.index(name_col)] = self._state.id_names.get(new_id)

    def _fragment_row(self, row, oid_indices: list[int], cols: list[str]):
        cfg, st = self._config, self._state
        new_row = list(row)
        changed = False
        for oid_i in oid_indices:
            oid = row[oid_i]
            if not isinstance(oid, int):
                continue
            if oid not in st.frag_map:
                # Decide once per identity whether its track gets fragmented.
                st.frag_map[oid] = st.next_synth_id() if st.rng.random() < cfg.rate else None
                st.frag_seen[oid] = 0
            syn = st.frag_map[oid]
            if syn is None:
                continue
            seen = st.frag_seen[oid]
            st.frag_seen[oid] = seen + 1
            # First sighting keeps the original id; later sightings split ~50/50.
            if seen >= 1 and st.rng.random() < 0.5:
                new_row[oid_i] = syn
                # The synthetic fragment is an unnamed new cluster.
                self._apply_name_fixup(new_row, cols, oid_i, syn)
                changed = True
        return tuple(new_row) if changed else row

    def _occlude_faces(self, rows: list, allow_drop: bool) -> list:
        cfg, rng = self._config, self._state.rng
        out = []
        for row in rows:
            if rng.random() >= cfg.rate:
                out.append(row)
                continue
            byte_idx = [
                i for i, v in enumerate(row)
                if isinstance(v, (bytes, bytearray, memoryview)) and len(v) > 0
            ]
            if byte_idx:
                new_row = list(row)
                for i in byte_idx:
                    new_row[i] = b""
                out.append(tuple(new_row))
            elif allow_drop:
                continue  # occluded face is simply lost
            else:
                out.append(row)
        return out

    def _inject_false_positives(self, rows: list, cols: list[str]) -> tuple[list, int]:
        cfg, st = self._config, self._state
        # Pick the identity column to corrupt for this query. Generalized to any
        # perception-identity alias (object_id / subject_id / person_id /
        # subject_object_id / object_object_id) so the interaction/ownership path
        # is actually corrupted; previously only exact names matched, which made
        # fp_inject a no-op on aliased interaction queries.
        target = None
        if "object_observations" in self._tables or "interactions" in self._tables:
            for i, c in enumerate(cols):
                if _is_object_identity_col(c):
                    target = i
                    break
        if target is None:
            return rows, 0

        col_name = cols[target]
        pool = st.id_pools.setdefault(col_name, set())
        out: list = []
        injected = 0
        for row in rows:
            out.append(row)
            value = row[target]
            if isinstance(value, int):
                pool.add(value)
                candidates = [pid for pid in pool if pid != value]
                if candidates and st.rng.random() < cfg.rate:
                    new_row = list(row)
                    new_row[target] = st.rng.choice(candidates)
                    # The false sighting is attributed to the other identity,
                    # including its name (the channel the LLM reads).
                    self._apply_name_fixup(new_row, cols, target, new_row[target])
                    out.append(tuple(new_row))
                    injected += 1
        return out, injected

    # -- temporal -------------------------------------------------------------
    def _jitter_timestamps(self, rows: list, cols: list[str]) -> tuple[list, int]:
        from datetime import datetime, timedelta

        cfg, rng = self._config, self._state.rng
        ts_idx = [
            i for i, c in enumerate(cols)
            if _is_timestamp_col(c) and any(isinstance(r[i], datetime) for r in rows)
        ]
        if not ts_idx or cfg.seconds <= 0:
            return rows, 0
        out: list = []
        changed = 0
        for row in rows:
            new_row = list(row)
            row_changed = False
            for i in ts_idx:
                v = row[i]
                if isinstance(v, datetime):
                    # Uniform jitter in [-seconds, +seconds]; breaks "most recent"
                    # ordering without destroying the overall timeline.
                    new_row[i] = v + timedelta(seconds=rng.uniform(-cfg.seconds, cfg.seconds))
                    row_changed = True
            if row_changed:
                changed += 1
            out.append(tuple(new_row) if row_changed else row)
        return out, changed

    # -- action noise -----------------------------------------------------------
    def _noise_actions(self, rows: list, cols: list[str]) -> tuple[list, int]:
        cfg, rng = self._config, self._state.rng
        act_idx = [i for i, c in enumerate(cols) if _is_action_col(c)]
        if not act_idx:
            return rows, 0
        # The caption narrates the true action ("...is holding a controller"),
        # an uncorrupted side channel that let the agent recover every swapped
        # verb (observed: action_noise scored exactly baseline). Keep the two
        # channels consistent: rewrite the verb inside the caption when it
        # appears verbatim, otherwise blank the caption.
        cap_idx = next((i for i, c in enumerate(cols) if c == "caption"), None)
        out: list = []
        changed = 0
        for row in rows:
            new_row = list(row)
            row_changed = False
            for i in act_idx:
                v = row[i]
                if isinstance(v, str) and v.strip() and rng.random() < cfg.rate:
                    candidates = [a for a in ACTION_SWAP_POOL if a.lower() != v.strip().lower()]
                    if candidates:
                        new_action = rng.choice(candidates)
                        new_row[i] = new_action
                        if cap_idx is not None and isinstance(new_row[cap_idx], str):
                            cap = new_row[cap_idx]
                            m = re.search(re.escape(v.strip()), cap, re.IGNORECASE)
                            new_row[cap_idx] = (
                                cap[: m.start()] + new_action + cap[m.end():] if m else ""
                            )
                        row_changed = True
            if row_changed:
                changed += 1
            out.append(tuple(new_row) if row_changed else row)
        return out, changed

    # -- class confusion -----------------------------------------------------------
    def _confuse_classes(self, rows: list, cols: list[str]) -> tuple[list, int]:
        cfg, rng = self._config, self._state.rng
        cls_idx = [i for i, c in enumerate(cols) if _is_class_id_col(c)]
        if not cls_idx:
            return rows, 0
        # Same side-channel closure as action_noise: a misclassification would
        # also corrupt the generated caption, so blank it when the class flips.
        cap_idx = next((i for i, c in enumerate(cols) if c == "caption"), None)
        out: list = []
        changed = 0
        for row in rows:
            new_row = list(row)
            row_changed = False
            for i in cls_idx:
                v = row[i]
                if isinstance(v, int) and v in CLASS_CONFUSION and rng.random() < cfg.rate:
                    new_row[i] = CLASS_CONFUSION[v]
                    if cap_idx is not None and isinstance(new_row[cap_idx], str):
                        new_row[cap_idx] = ""
                    row_changed = True
            if row_changed:
                changed += 1
            out.append(tuple(new_row) if row_changed else row)
        return out, changed

    # -- identity swap (consistent bijection) --------------------------------------
    def _swap_identities(self, rows: list, cols: list[str]) -> tuple[list, int]:
        cfg, st = self._config, self._state
        id_idx = [i for i, c in enumerate(cols) if _is_object_identity_col(c)]
        if not id_idx:
            return rows, 0
        out: list = []
        changed = 0
        for row in rows:
            new_row = list(row)
            row_changed = False
            for i in id_idx:
                v = row[i]
                if not isinstance(v, int):
                    continue
                mapped = self._swap_partner(v)
                if mapped is not None and mapped != v:
                    new_row[i] = mapped
                    # Name follows the swapped identity (mislabeled tracker).
                    self._apply_name_fixup(new_row, cols, i, mapped)
                    row_changed = True
            if row_changed:
                changed += 1
            out.append(tuple(new_row) if row_changed else row)
        return out, changed

    def _swap_partner(self, oid: int) -> int | None:
        """Return the consistent swap partner for ``oid`` (``oid`` itself if not swapped).

        The bijection is precomputed at install time over the whole identity
        universe (see ``_build_identity_swap_map``), so the mapping is fully
        consistent and bijective for the entire run — a systematically mislabeled
        tracker, unlike random fp_inject. Identities not enrolled (or appearing
        after install) map to themselves.
        """
        return self._state.swap_map.get(oid, oid)


class PerturbedConnection:
    """Drop-in replacement for a psycopg2 connection returning perturbed cursors."""

    def __init__(self, real, config: PerturbationConfig, state: _RunState):
        self._real = real
        self._config = config
        self._state = state

    def cursor(self, *args, **kwargs):
        return PerturbedCursor(self._real.cursor(*args, **kwargs), self._config, self._state)

    def __enter__(self):
        self._real.__enter__()
        return self

    def __exit__(self, *exc):
        return self._real.__exit__(*exc)

    def __getattr__(self, name):
        return getattr(self._real, name)


# ---------------------------------------------------------------------------
# install
# ---------------------------------------------------------------------------

def install(config: PerturbationConfig, tools_module) -> _RunState:
    """Patch ``tools_module._get_conn`` so every tool query is perturbed.

    Must be called before the agent answers any question. Returns the shared
    run state (useful for inspection). The real database is never modified.
    """
    import psycopg2

    state = _RunState(rng=random.Random(config.seed))

    # identity_swap needs a *fully consistent* bijection (a systematically
    # mislabeled tracker), which lazy per-arrival pairing cannot guarantee (the
    # first-seen half of a pair would flip from itself to its partner mid-run).
    # Precompute the swap over the whole identity universe once, up front.
    if config.kind == "identity_swap":
        state.swap_map = _build_identity_swap_map(tools_module, config)

    # Identity perturbations must keep the paired *name* cells consistent with
    # the rewritten ids (the LLM answers from names, not id numbers). Load the
    # id -> name universe once, up front. Read-only.
    if config.kind in ("identity_swap", "fp_inject", "track_frag"):
        state.id_names = _load_identity_names(tools_module)

    def _perturbed_get_conn():
        real = psycopg2.connect(tools_module.DATABASE_URL)
        return PerturbedConnection(real, config, state)

    tools_module._get_conn = _perturbed_get_conn
    return state


def _load_identity_names(tools_module) -> dict[int, str | None]:
    """Return ``objects.id -> objects.name`` for the whole identity universe.

    Used to keep paired name columns consistent when an identity id is
    rewritten (identity_swap / fp_inject / track_frag). Read-only: only
    SELECTs. Synthetic ids are absent and therefore map to ``None`` (unnamed).
    """
    import psycopg2

    try:
        conn = psycopg2.connect(tools_module.DATABASE_URL)
        cur = conn.cursor()
        cur.execute("SELECT id, name FROM objects")
        names = {int(r[0]): r[1] for r in cur.fetchall()}
        cur.close()
        conn.close()
        return names
    except Exception:
        return {}


def _build_identity_swap_map(tools_module, config: PerturbationConfig) -> dict[int, int]:
    """Return a deterministic swap bijection over all perception identity ids.

    Enrolls a ``rate`` fraction of identities and pairs them (A<->B) with the
    seeded RNG, so the mapping is consistent and bijective for the entire run.
    Identities not enrolled map to themselves. Read-only: only SELECTs ids.
    """
    import psycopg2

    ids: set[int] = set()
    try:
        conn = psycopg2.connect(tools_module.DATABASE_URL)
        cur = conn.cursor()
        cur.execute("SELECT DISTINCT object_id FROM object_observations WHERE object_id IS NOT NULL")
        ids |= {int(r[0]) for r in cur.fetchall()}
        cur.execute("SELECT DISTINCT person_id FROM object_observations WHERE person_id IS NOT NULL")
        ids |= {int(r[0]) for r in cur.fetchall()}
        cur.close()
        conn.close()
    except Exception:
        # If the universe can't be read, fall back to no swapping (identity map).
        return {}

    rng = random.Random(config.seed)
    enrolled = sorted(oid for oid in ids if rng.random() < config.rate)
    rng.shuffle(enrolled)
    swap: dict[int, int] = {}
    # Pair consecutive enrolled identities; an odd one out maps to itself.
    for i in range(0, len(enrolled) - 1, 2):
        a, b = enrolled[i], enrolled[i + 1]
        swap[a] = b
        swap[b] = a
    return swap
