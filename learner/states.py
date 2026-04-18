"""State enum for the orchestrator loop."""

from enum import Enum


class State(str, Enum):
    # IDLE is human-in-the-loop wait: Claude has requested a physical scene
    # change and the loop is blocked until the operator hits "Scene ready".
    IDLE = "IDLE"
    # THINK is Claude-in-the-loop wait: a `claude -p` subprocess is running
    # and its return value drives the next state.
    THINK = "THINK"
    VERIFY = "VERIFY"
    EXPLORE = "EXPLORE"
    RETRAIN = "RETRAIN"
