#!/usr/bin/env python3
"""Generate synthetic office datasets v2 — intent-driven decision model.

Design doc: INTENT_GENERATOR_V2_DESIGN.md (decision logic §10, config schema §9).

Key properties vs. v1 (generate_scene_change_dataset.py):
    • Config-driven (TOML, stdlib tomllib; optional `extends` deep-merge)
    • All motion is an Intent (person, dest, carry) — teleport between steps,
      departure cue rendered in the origin room in the decision step
    • Objects never flicker: carried items derive their room from the carrier;
      temp items (plate, cup, ...) are real tracked entities with a TTL
    • Emits the labeled scene_changes_* schema directly (change_type + interest)
    • Optional intent episodes with an intent_episodes.json ground-truth sidecar

Usage:
    python curiosity/generate_scene_change_dataset_v2.py --config data/curiosity/room_configs/intent_cued.toml
"""

import argparse
import copy
import json
import random
import re
import tomllib
from dataclasses import dataclass
from pathlib import Path

INTERVAL_MINUTES = 5

# ---------------------------------------------------------------------------
# Time utilities (int minutes since midnight internally, "HH:MM" at the edges)
# ---------------------------------------------------------------------------

def to_min(t: str) -> int:
    h, m = t.split(":")
    return int(h) * 60 + int(m)


def to_str(minutes: int) -> str:
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


# ---------------------------------------------------------------------------
# Config loading (TOML + optional `extends` deep-merge; lists are replaced)
# ---------------------------------------------------------------------------

def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        v = copy.deepcopy(v)
        if isinstance(v, dict) and v.pop("__replace__", False):
            out[k] = v  # marker: replace subtree entirely (used for world overrides)
        elif k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: Path) -> dict:
    with path.open("rb") as f:
        cfg = tomllib.load(f)
    if "extends" in cfg:
        parent = load_config((path.parent / cfg.pop("extends")).resolve())
        cfg = _deep_merge(parent, cfg)
    return cfg


# ---------------------------------------------------------------------------
# World (resolved from config)
# ---------------------------------------------------------------------------

class World:
    def __init__(self, cfg: dict):
        self.rooms: dict[str, dict] = cfg["world"]["rooms"]
        self.room_names = list(self.rooms)
        self.people: dict[str, dict] = cfg["world"]["people"]
        self.personal_objects: dict[str, dict] = cfg["world"].get("objects", {})
        self.activities: dict[str, list[dict]] = cfg["activities"]

        d = cfg["world"].get("distances", {})
        default_d = d.get("default", 2)
        exc = {}
        for e in d.get("exceptions", []):
            a, b = e["pair"]
            exc[frozenset((a, b))] = e["d"]

        def distance(a: str, b: str) -> int:
            if a == b:
                return 0
            return exc.get(frozenset((a, b)), default_d)

        self.distance = distance

    def room_type(self, room: str) -> str:
        return self.rooms[room]["type"]

    def rooms_of_type(self, rtype: str) -> list[str]:
        return [r for r in self.room_names if self.rooms[r]["type"] == rtype]

    def activity_spec(self, room: str, text: str) -> dict | None:
        for spec in self.activities.get(self.room_type(room), []):
            if spec["text"] == text:
                return spec
        return None


# Conditioned arrival activities (design §10.4): text -> duration range (min)
ARRIVAL_ACTIVITIES = {
    "shares a snack with the group": (10, 30),
    "hands over a mug, chats": (10, 20),
    "puts the box on the shelf": (5, 10),
}

# Fixture attribute -> caption clause (design §5)
_NUMWORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight"}
FIXTURE_CLAUSES = {
    "cups_set_out": lambda v: f"{_NUMWORDS.get(v, v)} cups are set out on the counter",
    "whiteboard_fresh": lambda v: "the whiteboard is freshly written on",
    "chairs_pulled_out": lambda v: "the chairs are pulled away from the table",
    "printer_stack": lambda v: f"the printer output tray holds a stack of {v} pages",
    "screen_on": lambda v: "the screen is on",
}


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

@dataclass
class PersonState:
    name: str
    home: str
    room: str | None
    activity: str = ""
    activity_end: int = 0            # minutes; decision fires when t >= activity_end
    present: bool = True
    departing: bool = False          # render departure cue this step
    pending: dict | None = None      # intent to execute on arrival next step
    carrying: str | None = None      # item currently carried (temp kind or personal object)
    spawned: list[int] = None        # temp item iids created by current activity

    def __post_init__(self):
        if self.spawned is None:
            self.spawned = []


@dataclass
class TempItem:
    iid: int
    kind: str                        # display name: plate, cup, two_mugs, ...
    carried_by: str | None = None
    location: str | None = None      # room when not carried
    despawn_at: int | None = None    # minutes; only when loose

    def effective_room(self, people: dict[str, PersonState]) -> str | None:
        if self.carried_by:
            return people[self.carried_by].room
        return self.location


@dataclass
class PersonalObject:
    name: str
    owner: str
    carried: bool = False
    location: str | None = None      # room when not carried

    def effective_room(self, people: dict[str, PersonState]) -> str | None:
        if self.carried:
            return people[self.owner].room
        return self.location


@dataclass
class Intent:
    cue_step: int                    # minutes: departure cue renders in origin
    person: str
    dest: str
    carry: str | None = None         # temp kind or personal object name
    force_activity: str | None = None
    exit_after: bool = False         # person leaves for the day on arrival
    source: str = "random"           # schedule | episode | random
    episode_id: str | None = None


# ---------------------------------------------------------------------------
# Sampling helpers (design §10.3)
# ---------------------------------------------------------------------------

def sample_duration(rng: random.Random, dur: list[int] | tuple) -> int:
    d = rng.randint(int(dur[0]), int(dur[1]))
    # round up to a whole step for clean boundaries (as v1)
    return max(INTERVAL_MINUTES, -(-d // INTERVAL_MINUTES) * INTERVAL_MINUTES)


def softmax_sample(rng: random.Random, scores: dict[str, float], tau: float) -> str:
    import math
    mx = max(scores.values())
    weights = {k: math.exp((v - mx) / tau) for k, v in scores.items()}
    total = sum(weights.values())
    r = rng.random() * total
    for k, w in weights.items():
        r -= w
        if r <= 0:
            return k
    return next(reversed(weights))


def p_move(cfg: dict, world: World, room: str, t: int) -> float:
    dec = cfg["decisions"]
    base = dec["p_move"]["base"].get(world.room_type(room), 0.4)
    mult = 1.0
    # per-room override (distractor-heavy rooms churn more), multiplicative
    mult *= dec["p_move"].get("room_multipliers", {}).get(room, 1.0)
    for tm in dec["p_move"].get("time_multipliers", []):
        lo, hi = to_min(tm["window"][0]), to_min(tm["window"][1])
        if lo <= t <= hi:
            mult *= tm["factor"]
    return min(base * mult, 0.95)


def choose_carry(cfg: dict, person: PersonState, rng: random.Random) -> str | None:
    """Design §10.3b: one item per activity, conditioned on the activity that ended."""
    if person.carrying:
        return person.carrying
    entry = cfg["decisions"].get("carry_propensity", {}).get(person.activity)
    if entry and rng.random() < entry["p"]:
        return entry["item"]
    return None


def _rule_matches(when: dict, room: str, person: PersonState, world: World,
                  occupants: dict[str, int]) -> bool:
    if "occupants_gte" in when and occupants.get(room, 0) < when["occupants_gte"]:
        return False
    if "room_type" in when and world.room_type(room) != when["room_type"]:
        return False
    if "room" in when and room != when["room"]:
        return False
    if when.get("is_home") and room != person.home:
        return False
    return True


def choose_dest(cfg: dict, world: World, person: PersonState, carry: str | None,
                t: int, rng: random.Random, occupants: dict[str, int],
                upcoming_events: list[dict]) -> str:
    """Design §10.3c: base affinity + event boost + social pull + intent bonus, softmax."""
    dec = cfg["decisions"]
    aff = dec["base_affinity"]
    scores: dict[str, float] = {}
    for r in world.room_names:
        if r == person.room:
            continue
        if r == person.home:
            s = aff.get("home", 5.0)
        elif world.room_type(r) == "office":
            s = aff.get("office_other", 0.5)
        else:
            s = aff.get(r, aff.get(world.room_type(r), 0.5))
        scores[r] = s

    eb = dec.get("event_boost", {})
    if eb.get("value"):
        lead = eb.get("lead_minutes", 15)
        for ev in upcoming_events:
            ev_start = to_min(ev["start"])
            if person.name in ev["_people_resolved"] and 0 <= ev_start - t <= lead:
                scores[ev["room"]] = scores.get(ev["room"], 0) + eb["value"]

    sp = dec.get("social_pull", {})
    if sp.get("per_person"):
        for r in scores:
            scores[r] += min(sp["per_person"] * occupants.get(r, 0), sp.get("cap", 1.5))

    if carry:
        # rules are evaluated per room; all matching rules stack (additive)
        for r in list(scores):
            for rule in dec.get("intent_bonus", {}).get(carry, []):
                if _rule_matches(rule.get("when", {}), r, person, world, occupants):
                    scores[r] += rule["add"]

    return softmax_sample(rng, scores, dec.get("tau", 1.0))


# ---------------------------------------------------------------------------
# Episodes (design §3, §10.5): pre-rolled intent bundles + fixture mutations
# ---------------------------------------------------------------------------

def sample_episodes(cfg: dict, world: World, day_idx: int, start: int, end: int,
                    rng: random.Random, busy_windows: list[tuple[int, int]] = [],
                    event_windows: dict[str, list[tuple[int, int]]] | None = None) -> tuple[dict[int, list[Intent]], list[dict], list[dict]]:
    """Returns (intents_by_step, fixture_mutations, sidecar_records).

    busy_windows: (start, end) minute ranges (scheduled events) that episodes
    must not overlap — an event would pull actors out mid-episode.
    event_windows: per-person scheduled-event windows; actors busy in an
    event during the episode window are never picked (a schedule cue would
    otherwise win the shared cue step and silently drop the episode intent).
    """
    eps_cfg = cfg.get("episodes")
    if not eps_cfg:
        return {}, [], []
    event_windows = event_windows or {}
    exclude_event_actors = eps_cfg.get("exclude_event_actors", False)

    intents_by_step: dict[int, list[Intent]] = {}
    mutations: list[dict] = []   # {step, room, attr, value}
    sidecar: list[dict] = []

    def add_intent(it: Intent):
        intents_by_step.setdefault(it.cue_step, []).append(it)

    templates = eps_cfg["templates"]
    names = list(templates)
    weights = [templates[n].get("weight", 1) for n in names]
    window = eps_cfg.get("cue_to_change_window", [10, 30])
    n_eps = rng.randint(*eps_cfg.get("per_day", [4, 6]))
    social_rooms = world.rooms_of_type("social") or [world.room_names[0]]
    meeting_rooms = world.rooms_of_type("meeting")
    people_names = list(world.people)
    # Pass 1: plan episodes (template, t0, head-fake), then sort chronologically.
    # Actor assignment must happen in time order so that an actor who exits for
    # the day (pack_up) is never assigned to a later episode.
    plan: list[dict] = []
    for i in range(n_eps):
        tmpl = rng.choices(names, weights=weights, k=1)[0]
        spec = templates[tmpl]
        head_fake = rng.random() < eps_cfg.get("head_fake_rate", 0.10)
        lo = start + 3 * INTERVAL_MINUTES
        hi = end - max(window) - 3 * INTERVAL_MINUTES
        span_min = max(window) + 8 * INTERVAL_MINUTES  # cue lead + multi-actor spread
        for _attempt in range(20):
            t0 = lo + INTERVAL_MINUTES * rng.randint(0, max(0, (hi - lo) // INTERVAL_MINUTES))
            if all(t0 + span_min <= bs or t0 - 2 * INTERVAL_MINUTES >= be
                   for bs, be in busy_windows):
                break
        actors_n = spec.get("actors", 1)
        if isinstance(actors_n, list):
            actors_n = rng.randint(*actors_n)
        dest_pool = spec.get("dest_pool", ["@meeting"])
        dest = rng.choice(meeting_rooms) if dest_pool == ["@meeting"] and meeting_rooms else rng.choice(dest_pool)
        plan.append({"tmpl": tmpl, "spec": spec, "head_fake": head_fake,
                     "t0": t0, "actors_n": actors_n, "dest": dest})
    plan.sort(key=lambda e: e["t0"])

    used_actors: set[str] = set()
    exited: set[str] = set()
    actor_busy: dict[str, list[tuple[int, int]]] = {}
    span_min = max(window) + 8 * INTERVAL_MINUTES  # cue lead + multi-actor spread

    def is_free(p: str, lo: int, hi: int) -> bool:
        """No committed episode window overlapping [lo, hi]; with
        [episodes] exclude_event_actors also no scheduled event (a schedule
        cue would otherwise win a shared cue step and silently drop the
        episode intent). Off by default so legacy configs stay bit-identical."""
        if not all(hi <= bs or lo >= be for bs, be in actor_busy.get(p, [])):
            return False
        if exclude_event_actors:
            return all(hi <= s or lo >= e for s, e in event_windows.get(p, []))
        return True

    def pick(pool: list[str], k: int, lo: int, hi: int) -> list[str]:
        """Prefer actors not already used today; never pick exited actors or
        actors committed to an overlapping episode (a later episode's intents
        would otherwise yank them out mid-episode, breaking cue->arrival)."""
        avail = [p for p in pool if p not in exited and is_free(p, lo, hi)]
        fresh = [p for p in avail if p not in used_actors]
        chosen = rng.sample(fresh, min(k, len(fresh)))
        if len(chosen) < k:
            rest = [p for p in avail if p not in chosen]
            chosen += rng.sample(rest, min(k - len(chosen), len(rest)))
        return chosen[:k]

    def commit(actors: list[str], lo: int, hi: int):
        for a in actors:
            actor_busy.setdefault(a, []).append((lo, hi))

    # Pass 2: assign actors + build intents in chronological order.
    for i, ep in enumerate(plan):
        tmpl, spec, head_fake, t0 = ep["tmpl"], ep["spec"], ep["head_fake"], ep["t0"]
        actors_n, dest = ep["actors_n"], ep["dest"]
        eid = f"ep_d{day_idx}_{i:02d}"
        ep_lo, ep_hi = t0 - 2 * INTERVAL_MINUTES, t0 + span_min

        actors: list[str] = []
        cue_room = None
        follow_step = None

        if tmpl == "snack_delivery":
            chosen = pick(people_names, 1, ep_lo, ep_hi)
            if not chosen:
                continue  # nobody free in this window: drop the episode
            actor = chosen[0]
            kitchen = rng.choice(social_rooms)
            actors = [actor]
            cue_room = kitchen
            add_intent(Intent(t0 - 2 * INTERVAL_MINUTES, actor, kitchen,
                              force_activity="prepares a snack", source="episode", episode_id=eid))
            if head_fake:
                add_intent(Intent(t0, actor, world.people[actor]["home"],
                                  carry="plate", source="episode", episode_id=eid))
            else:
                add_intent(Intent(t0, actor, dest, carry="plate", source="episode", episode_id=eid))
            follow_step = t0 + INTERVAL_MINUTES

        elif tmpl == "coordinated_exodus":
            by_home: dict[str, list[str]] = {}
            for p in people_names:
                by_home.setdefault(world.people[p]["home"], []).append(p)
            # largest home group first, but fall back if its members are all
            # committed to overlapping episodes
            g_avail: list[str] = []
            for group in sorted(by_home.values(), key=len, reverse=True):
                g_avail = [p for p in group
                           if p not in exited and is_free(p, ep_lo, ep_hi)]
                if len(g_avail) >= 2:
                    break
            else:
                continue  # no home group free in this window: drop the episode
            actors = pick(g_avail, min(actors_n, len(g_avail)), ep_lo, ep_hi)
            cue_room = world.people[actors[0]]["home"]
            for a in actors:
                # route home first so the "gets up and leaves together" cue at t0
                # is truthful (actor may have wandered off before the episode)
                add_intent(Intent(t0 - 2 * INTERVAL_MINUTES, a, world.people[a]["home"],
                                  force_activity="works at their desk",
                                  source="episode", episode_id=eid))
            for k, a in enumerate(actors):
                add_intent(Intent(t0 + k * INTERVAL_MINUTES, a,
                                  world.people[a]["home"] if head_fake else dest,
                                  source="episode", episode_id=eid))
            follow_step = t0 + len(actors) * INTERVAL_MINUTES

        elif tmpl == "refreshment_prep":
            kitchen = rng.choice(social_rooms)
            cue_room = kitchen
            actors = pick(people_names, min(actors_n, len(people_names)), ep_lo, ep_hi)
            if not actors:
                continue
            mutations.append({"step": t0, "room": kitchen, "attr": "cups_set_out", "value": len(actors)})
            if not head_fake:
                for k, a in enumerate(actors):
                    add_intent(Intent(t0 + (2 + k) * INTERVAL_MINUTES, a, dest,
                                      source="episode", episode_id=eid))
                follow_step = t0 + (2 + len(actors)) * INTERVAL_MINUTES

        elif tmpl == "pack_up":
            chosen = pick(people_names, 1, ep_lo, ep_hi)
            if not chosen:
                continue
            actor = chosen[0]
            actors = [actor]
            home = world.people[actor]["home"]
            cue_room = home
            personal = [o for o, info in world.personal_objects.items() if info["owner"] == actor]
            carry = "backpack" if "backpack" in personal else (personal[0] if personal else None)
            if not head_fake:
                add_intent(Intent(t0, actor, home, source="episode", episode_id=eid))
                add_intent(Intent(t0 + 2 * INTERVAL_MINUTES, actor, home, carry=carry,
                                  exit_after=True, source="episode", episode_id=eid))
                exited.add(actor)  # gone for the day: unusable by later episodes
            follow_step = None  # pack_up predicts *absence*, not a destination change

        elif tmpl == "printer_run":
            room = "printer room" if "printer room" in world.rooms else (
                world.rooms_of_type("utility") or [world.room_names[0]])[0]
            chosen = pick(people_names, 1, ep_lo, ep_hi)
            if not chosen:
                continue
            actor = chosen[0]
            actors = [actor]
            cue_room = room
            mutations.append({"step": t0, "room": room, "attr": "printer_stack",
                              "value": rng.randint(2, 6)})
            if not head_fake:
                add_intent(Intent(t0 + 2 * INTERVAL_MINUTES, actor, room,
                                  force_activity="collects printouts", source="episode", episode_id=eid))
                add_intent(Intent(t0 + 4 * INTERVAL_MINUTES, actor, world.people[actor]["home"],
                                  source="episode", episode_id=eid))
                mutations.append({"step": t0 + 3 * INTERVAL_MINUTES, "room": room,
                                  "attr": "printer_stack", "value": 0})
                follow_step = None  # printer_run predicts a *transient* (distractor-class) visit

        used_actors.update(actors)
        commit(actors, ep_lo, ep_hi)
        window_end_min = t0 + rng.randint(*window)
        if follow_step:
            window_end_min = max(window_end_min, follow_step)
        sidecar.append({
            "episode_id": eid,
            "template": tmpl,
            "date": None,  # filled by caller
            "cue_ts": to_str(t0),
            "cue_room": cue_room,
            "dest_room": dest if not head_fake else None,
            "intended_dest_room": dest,
            "window_end": to_str(window_end_min) if follow_step else None,
            "followup_step": to_str(follow_step) if follow_step else None,
            "actors": actors,
            "head_fake": head_fake,
        })

    return intents_by_step, mutations, sidecar


# ---------------------------------------------------------------------------
# Caption rendering (design §5 — observability contract)
# ---------------------------------------------------------------------------

# Heading clauses appended to departure cues when [cues] p_heading > 0
# (design: intent-heading cues). Kept as module constants so the validator's
# caption lint can strip exactly these forms before the foreign-room check.
HEADING_FORMS = [
    ", heading for the {dest}",
    ", heading toward the {dest}",
    ", on the way to the {dest}",
]
HEADING_HOME = ", heading home"
HEADING_RE = re.compile(
    r",?\s*(?:heading for the|heading toward the|on the way to the|heading home)\b[^.]*")


def describe_scene_v2(world: World, room: str, people: dict[str, PersonState],
                      people_here: list[str], loose_objects: list[str],
                      fixture_attrs: dict, temp_kinds: dict[int, str],
                      headings: dict[str, str] | None = None) -> str:
    parts: list[str] = []
    if not people_here:
        parts.append("No one is here")
    else:
        for p in sorted(people_here):
            ps = people[p]
            if ps.departing and ps.pending:
                carry = ps.pending.get("carry")
                suffix = (headings or {}).get(p, "")
                if carry:
                    parts.append(f"{p} walks toward the door carrying a {carry.replace('_', ' ')}{suffix}")
                else:
                    parts.append(f"{p} walks toward the door{suffix}")
            else:
                parts.append(f"{p} {ps.activity}")

    for attr, val in fixture_attrs.items():
        if val and attr in FIXTURE_CLAUSES:
            parts.append(FIXTURE_CLAUSES[attr](val))

    fixture_names = set()
    for f in world.rooms[room].get("fixtures", []):
        fixture_names.add(f)
        fixture_names.add(f.replace("_", " "))
    movable = [o.replace("_", " ") for o in loose_objects
               if o not in fixture_names and o.replace("_", " ") not in fixture_names]
    if movable:
        if len(movable) == 1:
            parts.append(f"a {movable[0]} is on the table")
        elif len(movable) == 2:
            parts.append(f"a {movable[0]} and a {movable[1]} are on the table")
        else:
            obj_str = ", ".join(f"a {o}" for o in movable[:-1])
            parts.append(f"{obj_str}, and a {movable[-1]} are on the table")

    return ". ".join(parts) + "."


# ---------------------------------------------------------------------------
# Day simulation (design §10.2 main loop; timing convention §10.6:
# departure cue renders in origin at step t, person + carried item are in
# dest from step t+1 on)
# ---------------------------------------------------------------------------

def resolve_schedule(cfg: dict, world: World, day_idx: int, start: int,
                     rng: random.Random) -> tuple[dict[int, list[Intent]], dict[str, list], list[dict]]:
    """Layer 1: scheduled events -> intents + event guards + resolved events."""
    intents_by_step: dict[int, list[Intent]] = {}
    in_event: dict[str, list] = {p: [] for p in world.people}
    resolved: list[dict] = []
    for ev in cfg.get("schedule", []):
        if ev.get("day_index", 0) != day_idx:
            continue
        ppl = ev.get("people", [])
        if ppl == "all":
            ppl = list(world.people)
        elif "people_sample" in ev:
            ppl = rng.sample(list(world.people), min(ev["people_sample"], len(world.people)))
        ev = dict(ev)
        ev["_people_resolved"] = ppl
        resolved.append(ev)
        s, e = to_min(ev["start"]), to_min(ev["end"])
        for p in ppl:
            in_event[p].append((s, e, ev["room"]))
            cue = max(start, s - INTERVAL_MINUTES)
            intents_by_step.setdefault(cue, []).append(
                Intent(cue, p, ev["room"], source="schedule"))
    return intents_by_step, in_event, resolved


def simulate_day(cfg: dict, world: World, day_idx: int, date_str: str,
                 rng: random.Random) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    start, end = to_min(cfg["start_time"]), to_min(cfg["end_time"])
    times = list(range(start, end + INTERVAL_MINUTES, INTERVAL_MINUTES))
    dec = cfg["decisions"]
    ttl = dec.get("temp_item_ttl", [15, 30])

    people = {n: PersonState(n, info["home"], info["home"]) for n, info in world.people.items()}
    objects = {n: PersonalObject(n, info["owner"], location=world.people[info["owner"]]["home"])
               for n, info in world.personal_objects.items()}
    temp_items: dict[int, TempItem] = {}
    next_iid = [0]
    fixture_attrs: dict[str, dict] = {r: {} for r in world.room_names}

    intents_by_step, in_event, events = resolve_schedule(cfg, world, day_idx, start, rng)
    busy = [(to_min(ev["start"]), to_min(ev["end"])) for ev in events]
    event_windows = {p: [(s, e) for s, e, _ in wins] for p, wins in in_event.items()}
    ep_intents, mutations, sidecar = sample_episodes(
        cfg, world, day_idx, start, end, rng, busy, event_windows)
    for step, its in ep_intents.items():
        intents_by_step.setdefault(step, []).extend(its)
    for ep in sidecar:
        ep["date"] = date_str
    # Actors with a layer-1/2 intent coming up must not start a random move
    # that would make the scripted intent a silent no-op (validator §6.2).
    protected: dict[str, set[int]] = {}
    for step, its in intents_by_step.items():
        for it in its:
            protected.setdefault(it.person, set()).update((step - INTERVAL_MINUTES, step))
    # Episode actors must not wander off mid-episode: protect the full window
    # from first cue to window_end (validator checks presence at followup_step).
    for ep in sidecar:
        if ep["head_fake"] or not ep.get("window_end"):
            continue
        t_lo = to_min(ep["cue_ts"]) - 2 * INTERVAL_MINUTES
        t_hi = to_min(ep["window_end"])
        for a in ep["actors"]:
            protected.setdefault(a, set()).update(
                range(t_lo, t_hi + INTERVAL_MINUTES, INTERVAL_MINUTES))
    mutations_by_step: dict[int, list[dict]] = {}
    for m in mutations:
        mutations_by_step.setdefault(m["step"], []).append(m)

    # --- helpers -----------------------------------------------------------

    def spawn_items(p: PersonState, spec: dict, t: int):
        for kind in spec.get("spawns", []):
            iid = next_iid[0]
            next_iid[0] += 1
            temp_items[iid] = TempItem(iid, kind, location=p.room,
                                       despawn_at=t + rng.randint(*ttl))
            p.spawned.append(iid)

    def set_new_activity(p: PersonState, t: int, force: str | None = None):
        spec = None
        if force:
            spec = world.activity_spec(p.room, force)
            if spec is None and force in ARRIVAL_ACTIVITIES:
                spec = {"text": force, "dur": list(ARRIVAL_ACTIVITIES[force])}
            if spec is None:
                spec = {"text": force, "dur": [5, 10]}
        else:
            pool = world.activities.get(world.room_type(p.room), [])
            spec = rng.choice(pool) if pool else {"text": "is here", "dur": [10, 30]}
        p.activity = spec["text"]
        p.activity_end = t + sample_duration(rng, spec.get("dur", [10, 30]))
        p.spawned = []
        spawn_items(p, spec, t)

    def bind_carry(p: PersonState, kind: str):
        """Bind a carry item at cue time (design §10.2 step 2)."""
        if kind in objects:                      # personal object: force-take
            objects[kind].carried = True
            p.carrying = kind
            p.pending["_carry_iid"] = None
            return
        cand = [it for it in temp_items.values()
                if it.kind == kind and it.carried_by is None and it.location == p.room]
        if cand:
            it = max(cand, key=lambda x: x.iid)
        else:                                    # activity item gone: materialize one
            iid = next_iid[0]
            next_iid[0] += 1
            it = TempItem(iid, kind)
            temp_items[iid] = it
        it.carried_by = p.name
        it.despawn_at = None
        p.carrying = kind
        p.pending["_carry_iid"] = it.iid

    def fire_intent(p: PersonState, intent: Intent, t: int):
        if not p.present:
            return
        if p.room == intent.dest and not intent.exit_after:
            # already at destination: no move to cue; apply forced activity directly
            if intent.force_activity:
                set_new_activity(p, t, intent.force_activity)
            return
        p.departing = True
        p.pending = vars(intent).copy()
        # personal objects: one take-roll per item per move (design §2.4)
        take_p = dec.get("personal_carry_p", 0.9)
        for o in objects.values():
            if o.owner == p.name and not o.carried and o.location == p.room:
                if rng.random() < take_p:
                    o.carried = True
        if intent.carry:
            bind_carry(p, intent.carry)

    def execute_arrival(p: PersonState, t: int):
        intent = p.pending
        p.departing = False
        p.pending = None
        p.room = intent["dest"]
        # place carried temp item in dest (personal objects stay with owner)
        if intent.get("_carry_iid") is not None:
            it = temp_items[intent["_carry_iid"]]
            it.carried_by = None
            it.location = p.room
            it.despawn_at = t + rng.randint(*ttl)
        p.carrying = None
        if intent.get("exit_after"):
            p.present = False
            p.room = None
            return
        # arrival activity (design §10.4): conditioned on carry + occupancy
        carry = intent.get("carry")
        occupants_here = sum(1 for q in people.values()
                             if q.present and q.room == p.room and q.name != p.name)
        if intent.get("force_activity"):
            set_new_activity(p, t, intent["force_activity"])
        elif carry == "plate" and occupants_here >= 1:
            set_new_activity(p, t, "shares a snack with the group")
        elif carry == "two_mugs" and occupants_here >= 1:
            set_new_activity(p, t, "hands over a mug, chats")
        elif carry == "box" and world.room_type(p.room) == "utility":
            set_new_activity(p, t, "puts the box on the shelf")
        else:
            set_new_activity(p, t)

    def in_scheduled_event(pname: str, t: int) -> bool:
        return any(s <= t < e for s, e, _ in in_event[pname])

    # --- initial activities ------------------------------------------------
    for p in people.values():
        set_new_activity(p, start)

    scene_log: list[dict] = []
    person_locations: list[dict] = []
    object_locations: list[dict] = []

    for t in times:
        # 1. arrivals (intents fired one step earlier)
        for p in people.values():
            if p.pending is not None and p.pending.get("_arrive_at") == t:
                execute_arrival(p, t)

        # 2. expire temp items
        for iid in [i for i, it in temp_items.items()
                    if it.carried_by is None and it.despawn_at is not None and it.despawn_at <= t]:
            del temp_items[iid]

        # 3. fixture mutations (episodes)
        for m in mutations_by_step.get(t, []):
            fixture_attrs[m["room"]][m["attr"]] = m["value"]

        # 4. fire layer 1/2 intents scheduled at this step
        for intent in intents_by_step.get(t, []):
            p = people[intent.person]
            if p.pending is None and not p.departing:
                fire_intent(p, intent, t)
                if p.pending is not None:
                    p.pending["_arrive_at"] = t + INTERVAL_MINUTES

        # 5. layer 3: stochastic background decisions (design §10.3)
        occupants = {r: sum(1 for q in people.values() if q.present and q.room == r)
                     for r in world.room_names}
        for p in people.values():
            if not p.present or p.departing or p.pending is not None:
                continue
            if in_scheduled_event(p.name, t):
                continue
            if t in protected.get(p.name, ()):
                continue
            if t < p.activity_end:
                continue
            if rng.random() < p_move(cfg, world, p.room, t):
                carry = choose_carry(cfg, p, rng)
                dest = choose_dest(cfg, world, p, carry, t, rng, occupants, events)
                intent = Intent(t, p.name, dest, carry=carry, source="random")
                fire_intent(p, intent, t)
                if p.pending is not None:
                    p.pending["_arrive_at"] = t + INTERVAL_MINUTES
            else:
                set_new_activity(p, t)

        # 6. emit per-room records + trails
        # Heading cues (design: intent-heading clues): with per-source
        # probability a departure cue also names the destination. Only rolled
        # when enabled so [cues]-less configs stay bit-identical.
        cue_cfg = cfg.get("cues", {})
        p_heading_default = cue_cfg.get("p_heading", 0.0)
        p_heading_by_source = cue_cfg.get("p_heading_by_source", {})
        headings: dict[str, str] = {}
        if p_heading_default > 0 or p_heading_by_source:
            for p in people.values():
                if not (p.present and p.departing and p.pending):
                    continue
                src = p.pending.get("source", "random")
                ph = p_heading_by_source.get(src, p_heading_default)
                if p.pending.get("exit_after"):
                    if rng.random() < ph:
                        headings[p.name] = HEADING_HOME
                elif rng.random() < ph:
                    form = rng.choice(HEADING_FORMS)
                    headings[p.name] = form.format(dest=p.pending["dest"])

        for room in world.room_names:
            here = [p.name for p in people.values() if p.present and p.room == room]
            loose_personal = [o.name for o in objects.values()
                              if not o.carried and o.location == room]
            loose_temp = [it.kind for it in temp_items.values()
                          if it.carried_by is None and it.location == room]
            fixture_pseudo = sorted(a for a, v in fixture_attrs[room].items() if v)
            objects_present = sorted(loose_personal + loose_temp) + fixture_pseudo

            person_activities = {}
            for pname in here:
                ps = people[pname]
                if ps.departing and ps.pending:
                    c = ps.pending.get("carry")
                    base_act = ("walks toward the door carrying a " + c.replace("_", " ")) if c else "walks toward the door"
                    person_activities[pname] = base_act + headings.get(pname, "")
                else:
                    person_activities[pname] = ps.activity

            scene_log.append({
                "date": date_str,
                "time": to_str(t),
                "room": room,
                "scene": describe_scene_v2(world, room, people, here,
                                           loose_personal + loose_temp,
                                           fixture_attrs[room], None,
                                           headings=headings),
                "people_present": sorted(here),
                "objects_present": objects_present,
                "person_activities": person_activities,
            })

        for p in people.values():
            person_locations.append({"time": to_str(t), "person": p.name,
                                     "room": p.room, "activity": p.activity})
        for o in objects.values():
            object_locations.append({"time": to_str(t), "object": o.name,
                                     "room": o.effective_room(people), "with_owner": o.carried})
        for it in temp_items.values():
            object_locations.append({"time": to_str(t), "object": f"{it.kind}#{it.iid}",
                                     "room": it.effective_room(people),
                                     "with_owner": it.carried_by is not None})

    return scene_log, person_locations, object_locations, sidecar


# ---------------------------------------------------------------------------
# Labeler (design §B2.5): diff consecutive per-room states -> change_type,
# people/objects added/removed, activities, interest label
# ---------------------------------------------------------------------------

def temp_kinds_from_config(cfg: dict) -> set[str]:
    kinds = set()
    for specs in cfg["activities"].values():
        for spec in specs:
            kinds.update(spec.get("spawns", []))
    for entry in cfg["decisions"].get("carry_propensity", {}).values():
        kinds.add(entry["item"])
    return kinds


def label_day(scene_log: list[dict], world: World, cfg: dict,
              temp_kinds: set[str]) -> list[dict]:
    persist = cfg.get("labeling", {}).get("persistence_steps", 6)
    fixture_names = set(FIXTURE_CLAUSES)

    by_room: dict[str, list[dict]] = {r: [] for r in world.room_names}
    for e in scene_log:
        by_room[e["room"]].append(e)

    out: list[dict] = []
    for room, hist in by_room.items():
        prev_ppl: set[str] = set()
        prev_obj: set[str] = set()
        prev_act: dict = {}
        for i, e in enumerate(hist):
            ppl, obj = set(e["people_present"]), set(e["objects_present"])
            p_added = sorted(ppl - prev_ppl)
            p_removed = sorted(prev_ppl - ppl)
            o_added = sorted(obj - prev_obj)
            o_removed = sorted(prev_obj - obj)
            act_changed = e["person_activities"] != prev_act
            act_added = [f"{p} {a}" for p, a in e["person_activities"].items()
                         if prev_act.get(p) != a]
            act_removed = [f"{p} {a}" for p, a in prev_act.items()
                           if e["person_activities"].get(p) != a]

            fixture_changed = any(x in fixture_names for x in o_added + o_removed)
            n_obj = len(o_added) + len(o_removed)
            if p_added or p_removed or fixture_changed or n_obj >= 2:
                change_type = "major_change"
            elif n_obj == 1 or act_changed:
                change_type = "minor_change"
            else:
                change_type = "no_change"

            # --- interest label (design: personal/fixture/people + persistence) ---
            interest = None
            if change_type != "no_change":
                touches_temp = any(x in temp_kinds for x in o_added + o_removed)
                touches_personal = any(x in world.personal_objects for x in o_added + o_removed)
                touches_people = bool(p_added or p_removed)
                j = min(i + persist, len(hist) - 1)
                fut_ppl = set(hist[j]["people_present"])
                fut_obj = set(hist[j]["objects_present"])
                persistent = (all(p in fut_ppl for p in p_added)
                              and all(p not in fut_ppl for p in p_removed)
                              and all(o in fut_obj for o in o_added)
                              and all(o not in fut_obj for o in o_removed))
                if (touches_personal or fixture_changed or touches_people) and persistent:
                    interest = "interesting"
                elif touches_temp or touches_people or n_obj or act_changed:
                    interest = "distractor"

            e2 = dict(e)
            e2.update({
                "change_type": change_type,
                "people_added": p_added,
                "people_removed": p_removed,
                "objects_added": o_added,
                "objects_removed": o_removed,
                "activities_changed": act_changed,
                "activity_change_severity": ("major" if (p_added or p_removed or fixture_changed)
                                             else ("minor" if change_type != "no_change" else "none")),
                "activities_added": act_added,
                "activities_removed": act_removed,
                "interest": interest,
            })
            out.append(e2)
            prev_ppl, prev_obj, prev_act = ppl, obj, e["person_activities"]

    # restore chronological order (room-major within step, as emitted)
    order = {(e["time"], e["room"]): k for k, e in enumerate(scene_log)}
    out.sort(key=lambda e: order[(e["time"], e["room"])])
    return out


# ---------------------------------------------------------------------------
# Validator (design §6): hard-fail invariants + informational marginals
# ---------------------------------------------------------------------------

def validate(all_entries: list[dict], sidecar: list[dict], world: World,
             cfg: dict) -> list[str]:
    errors: list[str] = []

    # 1. caption lint: no foreign room names (observability contract).
    #    Heading clauses ("..., heading for the kitchen") intentionally name
    #    the destination room — strip them before the foreign-room check.
    for e in all_entries:
        scene_clean = HEADING_RE.sub("", e["scene"])
        for r in world.room_names:
            if r != e["room"] and r in scene_clean:
                errors.append(f"caption lint: room '{r}' in caption of '{e['room']}' "
                              f"at {e['date']} {e['time']}")

    # 2. cue -> arrival: non-head-fake episodes with a followup step must show
    #    actors (and carried items) present in dest at the followup step
    idx = {(e["date"], e["time"], e["room"]): e for e in all_entries}
    for ep in sidecar:
        if ep["head_fake"] or not ep.get("followup_step") or not ep.get("dest_room"):
            continue
        e = idx.get((ep["date"], ep["followup_step"], ep["dest_room"]))
        if e is None:
            errors.append(f"{ep['episode_id']}: no entry at followup "
                          f"{ep['followup_step']} {ep['dest_room']}")
            continue
        for a in ep["actors"]:
            if a not in e["people_present"]:
                errors.append(f"{ep['episode_id']}: actor {a} not in "
                              f"{ep['dest_room']} at {ep['followup_step']}")
        if ep["template"] == "snack_delivery" and "plate" not in e["objects_present"]:
            errors.append(f"{ep['episode_id']}: plate not in {ep['dest_room']} "
                          f"at {ep['followup_step']}")

    # 3. heading -> arrival: a "..., heading for the R" clause must be followed
    #    by the person being present in R at the next step (skips end-of-day,
    #    where no next entry exists).
    heading_clause = re.compile(
        r"^(\w+) walks toward the door.*?heading (?:for|toward) the (.+)$")
    for e in all_entries:
        nxt = to_str(to_min(e["time"]) + INTERVAL_MINUTES)
        for sentence in e["scene"].split(". "):
            m = heading_clause.match(sentence.strip().rstrip("."))
            if not m:
                continue
            person, dest = m.group(1), m.group(2)
            ne = idx.get((e["date"], nxt, dest))
            if ne is None:
                continue
            if person not in ne["people_present"]:
                errors.append(f"heading: '{person}' announced '{dest}' at "
                              f"{e['date']} {e['time']} but not there at {nxt}")

    return errors


def print_marginals(all_entries: list[dict], world: World, cfg: dict,
                    compare_dir: Path | None = None):
    days = len(cfg["days"])
    hours = (to_min(cfg["end_time"]) - to_min(cfg["start_time"])) / 60 * days
    print("\n--- marginals (changes per room-hour) ---")
    for room in world.room_names:
        entries = [e for e in all_entries if e["room"] == room]
        changed = sum(1 for e in entries if e["change_type"] != "no_change")
        interesting = sum(1 for e in entries if e.get("interest") == "interesting")
        distractor = sum(1 for e in entries if e.get("interest") == "distractor")
        print(f"  {room:15s} {changed / hours:5.1f}/h  "
              f"(interesting {interesting}, distractor {distractor})")

    if compare_dir and compare_dir.exists():
        print(f"--- comparison target: {compare_dir} ---")
        for room in world.room_names:
            total = 0
            n_days = 0
            for f in sorted(compare_dir.glob("scene_changes_2*.json")):
                data = json.loads(f.read_text())
                n_days += 1
                total += sum(1 for e in data
                             if e["room"] == room and e["change_type"] != "no_change")
            if n_days:
                ref_hours = 5 * n_days  # reference datasets are 09:00-14:00
                print(f"  {room:15s} {total / ref_hours:5.1f}/h")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--compare", type=Path, default=None,
                    help="reference dataset dir for marginal comparison")
    args = ap.parse_args()

    cfg = load_config(args.config)
    rng = random.Random(cfg["seed"])
    world = World(cfg)
    temp_kinds = temp_kinds_from_config(cfg)

    out_dir = Path(cfg["output"]["dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    all_entries: list[dict] = []
    all_person_locations: list[dict] = []
    all_object_locations: list[dict] = []
    all_sidecar: list[dict] = []

    for day_idx, date_str in enumerate(cfg["days"]):
        print(f"Simulating {date_str} (day {day_idx})...")
        scene_log, person_locs, object_locs, sidecar = simulate_day(
            cfg, world, day_idx, date_str, rng)
        labeled = label_day(scene_log, world, cfg, temp_kinds)
        all_entries.extend(labeled)
        all_person_locations.extend(person_locs)
        all_object_locations.extend(object_locs)
        all_sidecar.extend(sidecar)

        (out_dir / f"scene_changes_{date_str}.json").write_text(
            json.dumps(labeled, indent=2), encoding="utf-8")
        n_changed = sum(1 for e in labeled if e["change_type"] != "no_change")
        print(f"  {len(labeled)} entries, {n_changed} changes, "
              f"{len(sidecar)} episodes")

    (out_dir / "scene_changes_all.json").write_text(
        json.dumps(all_entries, indent=2), encoding="utf-8")
    if cfg["output"].get("write_person_object_trails", True):
        (out_dir / "person_locations_all.json").write_text(
            json.dumps(all_person_locations, indent=2), encoding="utf-8")
        (out_dir / "object_locations_all.json").write_text(
            json.dumps(all_object_locations, indent=2), encoding="utf-8")
    if cfg["output"].get("write_sidecar", True) and all_sidecar:
        (out_dir / "intent_episodes.json").write_text(
            json.dumps(all_sidecar, indent=2), encoding="utf-8")

    errors = validate(all_entries, all_sidecar, world, cfg)
    if errors:
        print("\nVALIDATION ERRORS:")
        for e in errors[:20]:
            print(f"  ✗ {e}")
        raise SystemExit(f"generation failed validation ({len(errors)} errors)")
    print("\nValidation: OK")

    print_marginals(all_entries, world, cfg, args.compare)
    print(f"\nDone. Output in {out_dir}")


if __name__ == "__main__":
    main()
