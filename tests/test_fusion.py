"""Unit tests for the fusion engine + event manager (pure logic, no models)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from life.config import Config
from life.events import EventManager
from life.fusion import FusionEngine
from life.models import Detection, Severity


FUSION_CFG = Config({
    "rules": [
        {"name": "confirmed_fall", "signal": "fall", "event_type": "FALL",
         "severity": "HIGH", "responder": "EMS", "min_confidence": 0.5,
         "min_persistence": 1, "cooldown_seconds": 30},
        {"name": "fire_detected", "signal": "fire", "event_type": "FIRE",
         "severity": "CRITICAL", "responder": "FIRE", "min_confidence": 0.4,
         "min_persistence": 3, "cooldown_seconds": 30},
        {"name": "weapon_threat", "signal": "weapon_threat", "event_type": "WEAPON_THREAT",
         "severity": "CRITICAL", "responder": "POLICE", "min_confidence": 0.4,
         "min_persistence": 1, "cooldown_seconds": 20},
    ],
    "combinations": [
        {"name": "fire_with_person", "requires": ["fire", "person"],
         "event_type": "FIRE_WITH_PERSON", "severity": "CRITICAL",
         "responder": "FIRE", "min_persistence": 2, "cooldown_seconds": 30},
    ],
})


def _det(signal, conf=0.9):
    return Detection(label=signal, confidence=conf, signal=signal, detector="test")


def test_fall_fires_immediately():
    fusion = FusionEngine(FUSION_CFG)
    cands = fusion.update([_det("fall", 0.8)])
    assert any(c.event_type == "FALL" and c.severity == Severity.HIGH for c in cands)


def test_low_confidence_is_ignored():
    fusion = FusionEngine(FUSION_CFG)
    cands = fusion.update([_det("fall", 0.3)])  # below 0.5 threshold
    assert cands == []


def test_fire_requires_persistence():
    fusion = FusionEngine(FUSION_CFG)
    assert fusion.update([_det("fire")]) == []          # frame 1
    assert fusion.update([_det("fire")]) == []          # frame 2
    cands = fusion.update([_det("fire")])               # frame 3 -> fires
    assert any(c.event_type == "FIRE" for c in cands)


def test_persistence_resets_when_signal_absent():
    fusion = FusionEngine(FUSION_CFG)
    fusion.update([_det("fire")])
    fusion.update([])                                    # gap resets counter
    assert fusion.update([_det("fire")]) == []          # only 1 consecutive again


def test_combination_escalation():
    fusion = FusionEngine(FUSION_CFG)
    fusion.update([_det("fire"), _det("person")])
    cands = fusion.update([_det("fire"), _det("person")])  # persistence 2 -> combo
    assert any(c.event_type == "FIRE_WITH_PERSON" and c.severity == Severity.CRITICAL
               for c in cands)


# ---- event manager cooldown ----------------------------------------------
class _FakeDispatcher:
    def __init__(self):
        self.sent = []

    def dispatch(self, event):
        self.sent.append(event)


class _FakeAlarm:
    def __init__(self):
        self.triggers = []

    def trigger(self, severity):
        self.triggers.append(severity)


def _event_manager(tmp_path):
    cfg = Config({
        "notifications": {"responders": {
            "EMS": {"name": "EMS", "contact": "108"},
            "FIRE": {"name": "Fire", "contact": "101"},
            "POLICE": {"name": "Police", "contact": "100"},
        }},
        "storage": {
            "snapshots_dir": str(tmp_path / "snaps"),
            "log_file": str(tmp_path / "events.jsonl"),
        },
    })
    dispatcher = _FakeDispatcher()
    alarm = _FakeAlarm()
    em = EventManager(cfg, dispatcher, alarm)
    em.set_cooldowns({"FALL": 30, "FIRE": 30})
    return em, dispatcher, alarm


def test_cooldown_debounces_duplicate_events(tmp_path):
    fusion = FusionEngine(FUSION_CFG)
    em, dispatcher, alarm = _event_manager(tmp_path)

    t = 1000.0
    for i in range(5):  # 5 consecutive fall frames within cooldown window
        cands = fusion.update([_det("fall", 0.9)])
        em.handle(cands, frame=None, timestamp=t + i * 0.1)

    assert len(dispatcher.sent) == 1  # only one dispatch despite 5 candidate frames
    assert alarm.triggers[0] == Severity.HIGH


def test_cooldown_allows_new_event_after_window(tmp_path):
    fusion = FusionEngine(FUSION_CFG)
    em, dispatcher, _ = _event_manager(tmp_path)

    em.handle(fusion.update([_det("fall", 0.9)]), frame=None, timestamp=1000.0)
    em.handle(fusion.update([_det("fall", 0.9)]), frame=None, timestamp=1040.0)  # > 30s later
    assert len(dispatcher.sent) == 2
