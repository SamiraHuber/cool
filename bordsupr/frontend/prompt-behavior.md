# VLM Navigation Strategy — Prompt Behavior Documentation

> **Last updated:** 2026-05-05  
> **Scope:** `agent/prompts.py`, `agent/exploration_simulator.py`  
> **Models:** Qwen/Qwen3-VL-4B-Instruct via vLLM (local) or Kimi (cloud)

---

## 1. Overview

Two VLM-driven strategies exist for the office exploration simulator:

| Strategy | Prompt File Key | UI Label | Core Idea |
|----------|----------------|----------|-----------|
| **Default** | `NAVIGATION_DEFAULT` | "VLM Agent (Default)" | Stay-or-move based on room schedule + observed scene |
| **Next Destination** | `NAVIGATION_NEXT_DESTINATION` | "VLM Agent (Next Destination)" | Same as default but with richer system-prompt guidance about schedule prioritization |
| **Scene-Only** | `NAVIGATION_SCENE_ONLY` | "VLM Agent (Scene-only)" | Decides purely from current vs. previous scene text; no historical statistics |

Both **Default** and **Next Destination** share **identical user-message construction** and **identical runtime state**. The only difference is the **system prompt** text.

---

## 2. Next Destination Strategy

### 2.1 System Prompt (`NAVIGATION_NEXT_DESTINATION`)

```text
You are a robot navigation strategist.

Your goal is to decide where the robot should go next and when to return, based on:
1. The current scene in the room you are standing in
2. The ROOM SCHEDULE showing when each room was last visited and when to revisit it
3. Your own visit history and event counts

ROOM SCHEDULE LEGEND:
- "stay" = you are currently in this room
- "in 3 min"  = revisit very soon (room was recently active)
- "in 10 min" = revisit soon (room was active or unseen)
- "in 20 min" = revisit later (room was static recently)
- "in 30 min" = revisit much later (room was static)
- "in >45 min"= only revisit if nothing else is happening
- "never visited → in 10 min" = unseen room, check it soon

DECISION:
- "stay" if the current room is dynamic, events are happening, or the scene is interesting
- "move" if the current room is boring, AND pick the target room with the most urgent revisit time
- When moving, prefer: (1) rooms marked "in 3 min", (2) unseen rooms, (3) rooms marked "in 10 min", (4) rooms you haven't seen in a while

Output ONLY a JSON object:
{"reasoning": "...", "action": "stay" or "move", "target_room": "Room Name"}
```

**What the system prompt gives the model:**
- A **schedule legend** that translates numeric revisit estimates into human-readable urgency labels.
- Explicit **prioritization rules** for target-room selection.
- Strict **output schema** (JSON only).

**What the system prompt does NOT give the model:**
- No explicit "patience" rules (those are enforced in code, see §2.4).
- No mention of the `event_detected` field that is actually required in the output (the user message appends this requirement).

---

### 2.2 User Message Construction (`_build_vlm_user_message`)

The user message is built fresh for **every VLM call** and contains four sections:

#### Section A — Current Simulation State
```text
CURRENT SIMULATION STATE:
- Current time: 08:42
- Current room: kitchen
- Currently in kitchen for 3 minute(s)
- Events observed this visit: 1
- Minutes since last event in this room: 0
```

**Values included:**
| Value | Source | Persisted? |
|-------|--------|------------|
| `current_time` | Log timestamp | No (new each minute) |
| `current_room` | `self.current_room` | Yes — updated on every move |
| `minutes_in_room` | `self.minutes_in_room` | Yes — incremented each minute in same room; reset to 0 on move |
| `events_this_visit` | `self.current_visit_events` | Yes — VLM-detected events this visit; reset to 0 on move |
| `last_event_minutes_ago` | `self.no_event_streak` | Yes — counts consecutive no-event minutes; reset to 0 when event detected |

#### Section B — Room Schedule
```text
ROOM SCHEDULE (all rooms — last visit & recommended next visit):
  office 1: last visited 08:30 (12 min ago), 2 events → in 10 min
  office 2: never visited → in 10 min
  kitchen: currently here → stay
  lounge: last visited 08:15 (27 min ago), 0 events → in 30 min
  ...
```

**How the schedule is computed:**
For each room in `ROOM_ORDER`:
- **Current room** → `"currently here → stay"`
- **Visited before** → `"last visited {time} ({minutes_ago} min ago), {events} events → {guess}"`
- **Never visited** → `"never visited → in 10 min"`

The `guess` string comes from `_next_visit_guess(minutes_since_last_visit, events_last_visit)`:

| Events last visit | Minutes since last visit | Guess |
|-------------------|-------------------------|-------|
| > 0 | ≤ 5 | "in 3 min" |
| > 0 | ≤ 15 | "in 10 min" |
| > 0 | > 15 | "in 20 min" |
| 0 | ≤ 10 | "in 20 min" |
| 0 | ≤ 25 | "in 30 min" |
| 0 | > 25 | "in >45 min" |

**Critical note:** The schedule is based on **the robot's own observations** (`room_last_visit_time` and `room_last_visit_events`), NOT on ground-truth event data. An unvisited room is treated as potentially interesting even if it has no real events.

**Values persisted across minutes:**
| Value | Type | Reset condition |
|-------|------|-----------------|
| `room_last_visit_time[room]` | `datetime` | Updated in `_finalize_visit()` when leaving a room |
| `room_last_visit_events[room]` | `int` | Updated in `_finalize_visit()` — stores VLM-detected events count |

#### Section C — Current Scene
```text
CURRENT SCENE (kitchen):
- Scene: Person cooking at stove, steam rising from pan.
```

The scene text is looked up in the ground-truth log for the **current room at current time**.

#### Section D — Rules & Output Format
```text
RULES:
- The robot observes minute-by-minute and may leave after just one observation if the scene is boring.
- If action is "stay", the robot remains in the current room for another minute.
- If action is "move", the robot immediately goes to target_room.
- Choose "move" if no events have been observed recently and another room looks more promising.
- Choose "stay" if events are happening or the room seems dynamic.

Output ONLY a JSON object:
{"reasoning": "...", "action": "stay" or "move", "target_room": "Room Name", "event_detected": true or false}
```

**Note:** The `event_detected` field is required in the output but is **not mentioned in the system prompt** — it is appended only in the user message. This is an inconsistency.

---

### 2.3 State Persisted Between VLM Calls

The `_SteppableVLM` class maintains the following state that survives across minutes:

```python
# Visit tracking
self.current_room: str | None = None           # Where the robot is NOW
self.pending_move_room: str | None = None      # Target room decided last step (applied next minute)
self.minutes_in_room: int = 0                  # Consecutive minutes in current room
self.no_event_streak: int = 0                  # Consecutive minutes without VLM-detected event

# Decision caching
self.last_vlm_decision: dict | None = None     # Parsed JSON from last fresh VLM call
self.last_vlm_prompt: str | None = None        # The prompt string sent for last VLM call
self.minutes_since_vlm_call: int = 0           # Minutes since last fresh call

# Room history (drives the schedule in the prompt)
self.room_last_visit_time: dict[str, datetime] = {}   # When robot last LEFT each room
self.room_last_visit_events: dict[str, int] = {}      # VLM-detected events during that last visit

# Scene tracking (for cached-minute event detection)
self.last_scene_in_room: dict[str, str] = {}    # Last observed scene text per room

# Counters
self.vlm_observed: int = 0                      # Total VLM-detected events across all rooms
self.current_visit_events: int = 0              # VLM-detected events in current visit only
self.stat_vlm_calls: int = 0                    # How many actual API calls made
self.stat_cache_hits: int = 0                   # How many times cache was used
```

**What gets reset on room change:**
- `minutes_in_room` → 0
- `current_visit_events` → 0
- `no_event_streak` → 0
- `last_vlm_decision` → None
- `last_vlm_prompt` → None

---

### 2.4 When Is the VLM Actually Called?

The code decides whether to make a fresh VLM call or reuse the cached decision:

```python
MIN_DWELL_BEFORE_BORED = 1  # Must stay at least 1 minute

if self.minutes_in_room >= MIN_DWELL_BEFORE_BORED:
    pred = _room_prediction(self.log, self.current_room, current_time)
    patience = _patience_for_prediction(pred)

    # Trigger 1: Boredom — no events for "patience" minutes
    if self.no_event_streak >= patience:
        need_vlm = True

    # Trigger 2: Reconsider — events are happening AND cooldown expired
    elif self.current_visit_events > 0 and self.minutes_since_vlm_call >= self.reconsider_cooldown:
        need_vlm = True
```

**Patience table** (derived from ground-truth room prediction):

| Prediction | Patience (no-event minutes before calling VLM) |
|------------|------------------------------------------------|
| "burst" | 6 |
| "3min" | 3 |
| "10min" | 4 |
| "30min" | 2 |
| ">45min" | 1 |

**The `reconsider_cooldown` parameter:**
- Controls how many minutes to wait between fresh VLM calls **when events are actively happening**.
- Default = 1 (reconsider every minute if events are occurring).
- Set to 0 → reconsider every single minute (expensive).
- Set to 10 → only reconsider every 10 minutes even if events are happening.

**Cached minutes:** When the VLM is NOT called, the previous `last_vlm_decision` is reused. For event detection during cached minutes, the code falls back to **scene-text comparison**:
```python
last_scene = self.last_scene_in_room.get(self.current_room)
if last_scene is not None and current_scene != last_scene:
    vlm_ev = 1  # scene changed → something happened
else:
    vlm_ev = 0
```

---

### 2.5 VLM Response Handling

```python
action = str(parsed.get("action") or "").strip().lower()
target_room = str(parsed.get("target_room") or "").strip()

if action == "move" and target_room and target_room != self.current_room:
    self.pending_move_room = target_room  # Applied at START of next minute
else:
    self.pending_move_room = None
```

**Important:** The move is **deferred by one minute**. If the VLM says "move to office 2" at 08:42, the robot:
1. Records 08:42 as still being in the current room
2. Moves to office 2 at the **start** of 08:43

This is why `pending_move_room` exists — it bridges the decision-to-action gap.

---

## 3. Default Strategy

### 3.1 System Prompt (`NAVIGATION_DEFAULT`)

```text
You are a robot navigation strategist.

Your goal is to decide whether the robot should stay in the current room or move to another room.

Output ONLY a JSON object:
{"reasoning": "...", "action": "stay" or "move", "target_room": "Room Name"}

If action is "stay", target_room is ignored.
Do not output markdown or extra text. Only the JSON object.
```

### 3.2 Comparison with Next Destination

| Aspect | Default | Next Destination |
|--------|---------|------------------|
| **System prompt** | Minimal — just asks for stay/move | Rich — includes schedule legend, prioritization rules, urgency labels |
| **User message** | Identical | Identical |
| **State persisted** | Identical | Identical |
| **Decision logic** | Identical | Identical |
| **VLM call triggers** | Identical | Identical |

**In practice, the Next Destination system prompt is a "superset" of Default** — it gives the model more guidance on how to interpret the room schedule and rank target rooms, but the actual information in the user message is the same.

---

## 4. Scene-Only Strategy

### 4.1 System Prompt (`NAVIGATION_SCENE_ONLY`)

```text
You are a robot navigation strategist.

Your goal is to decide whether the robot should stay in the current room or move to another room, based only on scene dynamics.
You do NOT have access to historical statistics — only the current scene and the previous scene.
```

### 4.2 User Message Differences

The user message for Scene-Only is **much shorter**:
- No room schedule
- No visit history counters
- Only: current time, current room, previously visited rooms (last 3), previous scene, current scene

The model is expected to decide purely by comparing the two scene texts.

---

## 5. Suggestions for Improvement

### 5.1 Prompt Issues

| # | Issue | Impact | Suggested Fix |
|---|-------|--------|---------------|
| 1 | `event_detected` field is required in output but **not mentioned in system prompts** | Model may omit it, causing fallback to `false` | Add `"event_detected": true or false` to ALL system prompts |
| 2 | System prompt and user message both repeat "Output ONLY a JSON object" | Redundant, wastes tokens | Keep it in system prompt only; remove from user message |
| 3 | Room schedule shows `"never visited → in 10 min"` for ALL unvisited rooms regardless of ground-truth activity | Robot may waste time on genuinely empty rooms | Use ground-truth event density to weight unseen rooms (e.g., high-activity rooms → "in 5 min", low-activity → "in 20 min") |
| 4 | `NAVIGATION_NEXT_DESTINATION` legend says `"in 3 min" = revisit very soon` but doesn't explain **why** that room is urgent | Model can't learn the pattern | Add a one-line explanation: `"in 3 min" = this room had events very recently — it may still be active` |
| 5 | The "minutes since last event" value shown in the user message is actually `no_event_streak` (VLM-detected), not ground-truth | If VLM misses events, the prompt shows "0 minutes since last event" when GT says otherwise | Rename the field to `"Minutes since last VLM-detected event"` to be honest about its source |

### 5.2 State / Algorithm Issues

| # | Issue | Impact | Suggested Fix |
|---|-------|--------|---------------|
| 6 | `room_last_visit_events` stores **VLM-detected** events, not ground-truth | A room with many real events but poor VLM detection gets deprioritized | Store both VLM-detected AND ground-truth counts; show GT count to model in schedule |
| 7 | Scene-change fallback for cached minutes (`current_scene != last_scene`) is brittle | Two different descriptions of the same scene trigger false positive events | Use semantic similarity (embeddings) or keyword extraction instead of exact string match |
| 8 | `reconsider_cooldown=0` calls VLM on **every minute** after boredom timeout expires | ~400 API calls, 15–60 min runtime, high cost | Add a "batch mode" where the VLM plans the next N minutes in one call |
| 9 | Move is deferred by 1 minute (`pending_move_room`) | The minute where "move" is decided is still recorded in the old room; events in the new room at that exact minute are missed | Apply move immediately and observe the new room in the same minute (would need log reordering) |
| 10 | `_room_prediction()` uses **ground-truth** events to set patience, but the VLM doesn't know this | The model sees a room as "dynamic" in the prompt but the code may force a move after only 1 minute (for `>45min` rooms) | Either expose the prediction to the model in the prompt, or base patience on VLM-detected history instead of GT |

### 5.3 Architectural Improvements

| # | Idea | Benefit |
|---|------|---------|
| 11 | **Multi-step planning:** Ask VLM to output a plan like `"stay for 3 min, then move to kitchen"` | Reduces API calls by ~60–80% |
| 12 | **Confidence thresholding:** If `event_detected` confidence is low, schedule a recheck sooner | Better precision on event detection |
| 13 | **Room embedding memory:** Maintain a vector embedding of each room's "activity profile" based on scene texts | Enables semantic similarity for scene-change detection and better room ranking |
| 14 | **Separate event detector + planner:** Use a cheap classifier for `event_detected`, only call the expensive VLM for move/stay decisions | Faster, cheaper, more deterministic event counting |
| 15 | **Ground-truth curriculum:** During training/evaluation, periodically show the model its own error (GT events it missed) in the prompt | Could improve event detection accuracy over a session |

---

## 6. Quick Reference: State Flow Diagram

```
Minute N:
  ├─ Observe current room → get scene text, GT events
  ├─ Check if VLM call needed (boredom? reconsider?)
  │   ├─ YES → Build user message (room schedule + scene)
  │   │       → Call VLM → parse JSON → cache as last_vlm_decision
  │   │       → Use VLM's own event_detected for this minute
  │   └─ NO  → Reuse last_vlm_decision
  │           → Detect event by scene-text comparison
  ├─ Update counters (vlm_observed, no_event_streak, etc.)
  ├─ Record MinuteStep
  └─ If action == "move": set pending_move_room (applied at start of N+1)

Minute N+1:
  ├─ Apply pending_move_room (if any)
  │   → Reset minutes_in_room, current_visit_events, no_event_streak
  │   → Update room_last_visit_time for previous room
  │   → Clear last_vlm_decision
  └─ (loop continues)
```

---

## 7. Files to Edit for Prompt Changes

| File | Purpose |
|------|---------|
| `agent/prompts.py` | Edit `NAVIGATION_DEFAULT`, `NAVIGATION_NEXT_DESTINATION`, `NAVIGATION_SCENE_ONLY` |
| `agent/exploration_simulator.py` | Edit `_build_vlm_user_message()` to change what goes into the user message |
| `agent/exploration_simulator.py` | Edit `_SteppableVLM.step()` to change when VLM is called or how events are detected |
