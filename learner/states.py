"""State enum for the orchestrator loop."""

from enum import Enum


class State(str, Enum):
    IDLE = "IDLE"
    VERIFY = "VERIFY"
    EXPLORE = "EXPLORE"
    RETRAIN = "RETRAIN"
