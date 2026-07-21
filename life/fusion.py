"""Fusion + severity/rules engine.

Turns per-frame detector signals into EventCandidates using a declarative,
config-driven rule table:

  * single-signal rules  -> require one signal with confidence + persistence
  * combination rules    -> require several signals present together (escalation)

Persistence = number of consecutive frames a signal has held. This is the
temporal-confirmation layer that cuts single-frame false positives. Cooldown is
applied later, by the EventManager.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List

from .models import Detection, EventCandidate, Severity


class Rule:
    def __init__(self, spec: dict) -> None:
        self.name = spec["name"]
        self.signal = spec["signal"]
        self.event_type = spec["event_type"]
        self.severity = Severity.parse(spec.get("severity", "HIGH"))
        self.responder = spec.get("responder", "POLICE")
        self.min_confidence = float(spec.get("min_confidence", 0.4))
        self.min_persistence = int(spec.get("min_persistence", 1))
        self.cooldown_seconds = float(spec.get("cooldown_seconds", 30))


class Combination:
    def __init__(self, spec: dict) -> None:
        self.name = spec["name"]
        self.requires = [str(s) for s in spec["requires"]]
        self.event_type = spec["event_type"]
        self.severity = Severity.parse(spec.get("severity", "CRITICAL"))
        self.responder = spec.get("responder", "POLICE")
        self.min_persistence = int(spec.get("min_persistence", 1))
        self.cooldown_seconds = float(spec.get("cooldown_seconds", 30))


class FusionEngine:
    def __init__(self, config) -> None:
        self.rules = [Rule(r) for r in config.get("rules", [])]
        self.combinations = [Combination(c) for c in config.get("combinations", [])]
        # persistence counters keyed by signal
        self._persistence: Dict[str, int] = defaultdict(int)
        # expose cooldowns so the EventManager can read per-event-type values
        self.cooldowns: Dict[str, float] = {}
        for r in self.rules:
            self.cooldowns[r.event_type] = r.cooldown_seconds
        for c in self.combinations:
            self.cooldowns[c.event_type] = c.cooldown_seconds

    def update(self, detections: Iterable[Detection]) -> List[EventCandidate]:
        """Feed one frame's detections; return any event candidates."""
        detections = list(detections)

        # best confidence seen per signal this frame
        best: Dict[str, float] = {}
        by_signal: Dict[str, List[Detection]] = defaultdict(list)
        for d in detections:
            sig = d.signal or d.label
            by_signal[sig].append(d)
            best[sig] = max(best.get(sig, 0.0), d.confidence)

        # update persistence counters (each tracked signal exactly once)
        present = set(best.keys())
        for sig in set(self._persistence.keys()) | present:
            if sig in present:
                self._persistence[sig] += 1
            else:
                self._persistence[sig] = 0

        candidates: List[EventCandidate] = []

        # single-signal rules
        for rule in self.rules:
            conf = best.get(rule.signal, 0.0)
            if conf < rule.min_confidence:
                continue
            if self._persistence[rule.signal] < rule.min_persistence:
                continue
            candidates.append(
                EventCandidate(
                    event_type=rule.event_type,
                    severity=rule.severity,
                    responder=rule.responder,
                    confidence=conf,
                    rule=rule.name,
                    message=self._msg(rule.event_type, conf),
                    detections=list(by_signal.get(rule.signal, [])),
                )
            )

        # combination rules (escalation)
        for combo in self.combinations:
            if not all(sig in present for sig in combo.requires):
                continue
            if any(self._persistence[sig] < combo.min_persistence for sig in combo.requires):
                continue
            conf = min(best[sig] for sig in combo.requires)
            combined_dets: List[Detection] = []
            for sig in combo.requires:
                combined_dets.extend(by_signal.get(sig, []))
            candidates.append(
                EventCandidate(
                    event_type=combo.event_type,
                    severity=combo.severity,
                    responder=combo.responder,
                    confidence=conf,
                    rule=combo.name,
                    message=self._msg(combo.event_type, conf),
                    detections=combined_dets,
                )
            )

        return candidates

    @staticmethod
    def _msg(event_type: str, conf: float) -> str:
        pretty = event_type.replace("_", " ").title()
        return f"{pretty} detected (confidence {conf:.0%})."
