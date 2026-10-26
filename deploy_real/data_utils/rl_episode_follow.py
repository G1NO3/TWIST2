"""Episode boundaries for server_data_record_d435.py --episode_status.

The RL bridge (VLAPolicyBridge rlt_bridge) rewrites a small status json at
least every 0.5 s and immediately on every terminal: `episode` is the open
replay episode id (None between episodes), `last_episode` the one before,
and `reason` the terminal outcome right after it closes. Following it, the
recorder captures every frame of every RL episode (policy, takeover and
pause alike) instead of waiting for controller_data toggles, and labels
each episode with the replay's own outcome.

Labels keep the teleop recorder's words where they exist: success ->
"successful", failure -> "unsuccessful"; plus "discarded" and "aborted".
A discard pressed after the episode closed reaches an already-saved
episode: it is reported as a relabel of that RL episode id.
"""
import json
import os
import time

LABELS = {"success": "successful", "failure": "unsuccessful", "discarded": "discarded"}


def outcome_label(reason):
    if not isinstance(reason, str):
        return None
    if reason in LABELS:
        return LABELS[reason]
    if reason.startswith("aborted"):
        return "aborted"
    return None


class RLEpisodeFollower:
    def __init__(self, status_path, instruction_path=None, stale_s=5.0):
        self.status_path = status_path
        self.instruction_path = instruction_path
        self.stale_s = stale_s
        self.current = None        # RL episode id being recorded
        self.saved = {}            # RL episode id -> label it was saved / relabelled with
        self._instruction = (None, None)

    def read(self):
        try:
            with open(self.status_path, encoding="utf-8") as f:
                return json.load(f)
        except (OSError, ValueError):
            return None

    def instruction(self):
        """The bridge's current phase sentence, re-read only when the file changes."""
        if self.instruction_path is None:
            return None
        try:
            mtime = os.stat(self.instruction_path).st_mtime_ns
            if mtime != self._instruction[0]:
                with open(self.instruction_path, encoding="utf-8") as f:
                    self._instruction = (mtime, f.read().strip())
        except OSError:
            return None
        return self._instruction[1]

    def step(self, status, now=None):
        """Events for this status: ("save", rl_id, label, reason), ("start", rl_id),
        ("relabel", rl_id, label, reason). A missing or stale status ends a
        recording episode as aborted (bridge gone); it never starts one."""
        now = time.time() if now is None else now
        fresh = (isinstance(status, dict) and isinstance(status.get("time"), (int, float))
                 and now - status["time"] <= self.stale_s)
        events = []
        episode = status.get("episode") if fresh else None
        if self.current is not None and episode != self.current:
            label, reason = "aborted", "status_stale"
            if fresh:
                reason = status.get("reason")
                if status.get("last_episode") == self.current:
                    label = outcome_label(reason) or "unknown"
            events.append(("save", self.current, label, reason))
            self.saved[self.current] = label
            self.current = None
        if fresh and episode and episode != self.current:
            events.append(("start", episode))
            self.current = episode
        if fresh and episode is None:
            last = status.get("last_episode")
            label = outcome_label(status.get("reason"))
            if (last in self.saved and label is not None and label != self.saved[last]
                    and (label == "discarded" or self.saved[last] == "unknown")):
                events.append(("relabel", last, label, status.get("reason")))
                self.saved[last] = label
        return events

    def frame_fields(self, status):
        """Per-frame RL context; t_mono_ns is the clock the replay rows use."""
        fields = {"t_mono_ns": time.monotonic_ns(), "rl_episode": self.current,
                  "instruction": self.instruction()}
        if isinstance(status, dict):
            fields.update(rl_phase=status.get("phase"), rl_paused=status.get("paused"))
            # native bridge lanes (VLAPolicyBridge native_episode_status.v1): the takeover's
            # correcting state, rewritten the moment it changes -- the HG-DAgger human mask
            if "human" in status:
                fields.update(human=status.get("human"), hg_mode=status.get("hg_mode"))
        return fields
