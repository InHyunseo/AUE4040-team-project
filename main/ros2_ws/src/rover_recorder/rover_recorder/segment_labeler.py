"""Keyboard segment labeler. c=common, l=left, r=right, n=pause."""
from typing import Optional


VALID = {"c": "common", "l": "left", "r": "right", "n": "pause"}


class SegmentLabeler:
    def __init__(self):
        self.label: str = "pause"

    def on_key(self, ch: str) -> Optional[str]:
        if ch in VALID:
            self.label = VALID[ch]
            return self.label
        return None
