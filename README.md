<div align="center">

<img src="assets/logo.svg" alt="LIFE logo" width="120" height="120" />

# Project - LIFE

### Last-minute Intervention, Full Existence

**A real-time computer-vision guardian that detects life-threatening moments and alerts responders.**

![status](https://img.shields.io/badge/status-MVP-ef4444)
![python](https://img.shields.io/badge/python-3.10%2B-1a1e25)
![license](https://img.shields.io/badge/license-MIT-1a1e25)
![tests](https://img.shields.io/badge/tests-passing-22c55e)

</div>

---

Project - LIFE watches a camera feed and recognizes emergencies the instant they happen &mdash;
a person **falling**, **fire or smoke**, an **armed threat** &mdash; then decides how
serious the incident is, which agency should respond, **sounds a local alarm**, and
**dispatches an alert**. A polished web console shows the live annotated feed, a
status banner, and a running incident log.

When seconds decide outcomes, Project - LIFE buys back the minutes that matter.

> **Responsible use:** assistive alerting tool, **not** a certified life-safety system.
> See [Limitations & responsible use](#limitations--responsible-use).

---

## Highlights

- **Multi-hazard detection** (MVP): fall, fire/smoke, weapon + weapon-near-person threat.
- **Fusion + severity/rules engine**: confidence + temporal persistence per signal,
  combination escalation (e.g. `fire + person -> CRITICAL`), per-incident cooldown.
- **Pluggable, simulated notifications**: console, webhook, email (SMTP), SMS (Twilio).
  Real providers activate automatically when credentials are present.
- **Local audible alarm** with acknowledge/silence.
- **Public-facing web app**: black/grey theme, custom logo, hero + capabilities +
  pipeline + live operator console, MJPEG feed, WebSocket status/events.
- **Runs on a laptop CPU**, no cloud required. Works out-of-the-box via a synthetic
  `demo` source (no camera needed).
- **Config-driven** and **extensible**: add a hazard = one `Detector` subclass + one rule.

---

## Architecture

```
Camera / RTSP / file / demo
        |
   Capture (frame buffer)
        |
   +----+-----------------------------+
   | Shared YOLO pass (person, knife) |
   +----+-----------------------------+
        |
   Detectors:  Fall (MediaPipe pose)   Fire (YOLO | HSV+flicker)   Weapon (YOLO + proximity)
        |
   Fusion + severity/rules engine  (confidence + persistence + combinations)
        |
   Event manager  (debounce/cooldown -> snapshot + JSONL log)
        |            \
   Local alarm       Notification dispatcher -> console / webhook / email / sms
        |
   FastAPI  (MJPEG + WebSocket + REST) -> Web console
```

Component map:

- `life/capture.py` - webcam / file / rtsp / synthetic `demo` source
- `life/detectors/` - `base.py`, `yolo_engine.py`, `fall.py`, `fire.py`, `weapon.py`
- `life/fusion.py` - declarative rules + severity + combinations
- `life/events.py` - cooldown, snapshot, JSONL log, dispatch, alarm
- `life/alarm.py` - siren generation + async playback + acknowledge
- `life/notify/` - dispatcher + adapters
- `life/pipeline.py` - orchestration + thread-safe shared state
- `life/server.py` + `life/web/index.html` - web console
- `assets/logo.svg` - brand mark
- `config.yaml` - all tunables
- `run.py` - entrypoint

---

## Quick start

```bash
git clone https://github.com/RoboX2020/LIFE-system.git
cd LIFE-system
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python run.py                      # opens the demo (synthetic fire) source
```

Then open the console at **http://127.0.0.1:8000**.

On first run, model weights download automatically:
- `yolo26n.pt` (falls back to `yolo11n.pt` / `yolov8n.pt`)
- `models/pose_landmarker_lite.task` (MediaPipe pose)

### Use a real source

```bash
python run.py --source webcam --path 0                 # laptop webcam
python run.py --source file   --path /path/to/clip.mp4 # a video file
python run.py --source rtsp   --path rtsp://user:pass@camera/stream
```

Or set `source:` in `config.yaml`. The `demo` source needs no hardware and is the
easiest way to see the full detection -> alarm -> dispatch -> console flow.

---

## How detection works

### Fall detection (MediaPipe pose + voting)
Per frame, using the 33-point pose landmarks:
1. **Torso angle** - shoulder->hip vector vs. vertical; near-horizontal => on the ground.
2. **Vertical drop** - rapid downward motion of the body centroid within a short window
   (the "standing -> ground in a short span" signal).
3. **Aspect ratio** - pose bounding box switches from tall (standing) to wide (lying).

A weighted vote feeds a per-person state machine
`STANDING -> FALLING -> ON_GROUND -> CONFIRMED_FALL`. A fall is only confirmed after
it persists on the ground, which rejects normal sitting/bending.

### Fire / smoke
- **Primary**: a custom YOLO model with fire/smoke classes (`models/fire.pt`).
- **Fallback**: HSV color mask for fire-like regions + a flicker (frame-difference)
  check, so it still demos without custom weights.

### Weapon + threat reasoning
- Weapons come from the shared YOLO pass (COCO includes `knife`) and/or an optional
  custom model (`models/weapon.pt`) for gun/rifle/etc.
- **Escalation**: a weapon close to a detected person becomes a `weapon_threat`
  (CRITICAL, Police) instead of a bare `weapon`.

### Fusion + severity
`config.yaml -> fusion` maps signals to incidents:

- `fall` -> **FALL** / HIGH / EMS
- `fire` -> **FIRE** / CRITICAL / Fire
- `smoke` -> **SMOKE** / HIGH / Fire
- `weapon` -> **WEAPON** / HIGH / Police
- `weapon_threat` -> **WEAPON_THREAT** / CRITICAL / Police
- `fire + person` -> **FIRE_WITH_PERSON** / CRITICAL / Fire (combination escalation)

Each rule requires a **minimum confidence** and **minimum persistence** (consecutive
frames) before firing, and each incident type has a **cooldown** so you get one alert
per incident, not one per frame.

---

## Notifications (simulated by default)

All adapters are simulated unless you provide credentials. Enable them under
`notifications.adapters` in `config.yaml`.

- **console** - prints a dispatch line (on by default).
- **webhook** - POSTs the event JSON to a URL (point it at any stub server).
- **email** - real SMTP send if `SMTP_HOST`, `SMTP_PORT`, `ALERT_EMAIL_TO`
  (and optionally `SMTP_USER`/`SMTP_PASS`/`ALERT_EMAIL_FROM`) are set; otherwise logs
  the payload it would send.
- **sms** - real Twilio SMS if `TWILIO_SID`, `TWILIO_TOKEN`, `TWILIO_FROM`,
  `ALERT_SMS_TO` are set and `twilio` is installed; otherwise logs the message.

### Path to real emergency dispatch
Swap the simulated adapters for sanctioned integrations by adding credentials (email,
SMS) or writing a small `NotificationAdapter` subclass that calls your monitoring
center / agency API. **Do not** wire this directly to public emergency numbers without
authorization - route through an approved alarm-monitoring provider.

---

## Web console & API

- `GET  /`               - public site + live console
- `GET  /logo.svg`       - brand mark
- `GET  /video`          - MJPEG annotated stream
- `GET  /api/status`     - current status JSON
- `GET  /api/events`     - recent incidents
- `POST /api/ack`        - acknowledge / silence the alarm
- `POST /api/detector/{name}?enabled=true|false` - toggle a detector
- `WS   /ws`             - live status + new events

Incidents are also written to `data/events.jsonl` with snapshots in
`data/snapshots/`.

---

## Configuration

Everything tunable lives in `config.yaml`: source, per-detector thresholds, the
fusion rule/combination table, responder routing + contacts, notification adapters,
alarm minimum severity, storage paths, and server host/port. See the inline comments.

Custom weights (optional): drop `models/fire.pt` and/or `models/weapon.pt` in and set
their paths under `detectors.*.model`.

---

## Testing

```bash
pip install pytest
python -m pytest tests/ -q
```

`tests/test_fusion.py` covers the fusion engine (confidence gating, persistence,
combination escalation) and the event manager (cooldown debounce) with no model
downloads.

---

## Limitations & responsible use

- **Not a certified life-safety system.** Treat outputs as assistive hints; false
  positives and false negatives are expected. Keep a human in the loop.
- **Privacy**: processing is local by default - no cloud upload. Snapshots are stored
  locally under `data/`. For any real deployment, post clear "area under monitoring"
  notices and comply with local surveillance/consent laws.
- **Accuracy** depends on camera placement, lighting, occlusion, and model quality.
  The heuristic fire path is a demo-grade fallback; use a trained `fire.pt` in
  production. Gun detection needs a custom `weapon.pt` (COCO only provides `knife`).
- **Fall detection** currently tracks a single primary person (MediaPipe pose).
  Multi-person fall tracking is a planned extension.
- **Legal**: never auto-dial public emergency services without authorization; route
  through sanctioned monitoring providers.

---

## Extending

Add a new hazard (e.g. debris, shaking, water overflow, animal threat):
1. Create `life/detectors/<hazard>.py` subclassing `Detector`, returning
   `Detection(signal="<your_signal>", ...)`.
2. Register it in `Pipeline.__init__` (`self.detectors`).
3. Add a rule (and/or combination) under `fusion` in `config.yaml`.

No changes to the fusion engine, event manager, alarm, or web console are required.

---

## License

MIT &mdash; see [LICENSE](LICENSE).
