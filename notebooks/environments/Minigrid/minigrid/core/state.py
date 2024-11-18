from dataclasses import dataclass, field
from minigrid.core.constants import (
    COLOR_NAMES,
    IDX_TO_COLOR
)

@dataclass(frozen=True, eq=True)
class KeyState:
    color: str = ""
    col: int = 0
    row: int = 0

@dataclass(frozen=True, eq=True)
class BallState:
    color: str = ""
    col: int = 0
    row: int = 0

@dataclass(frozen=True, eq=True)
class BoxState:
    color: str = ""
    col: int = 0
    row: int = 0

@dataclass(frozen=True, eq=True)
class DoorState:
    color: str = ""
    locked: bool = True

@dataclass(frozen=True, eq=True)
class AdversaryState:
    color: str = ""
    col: int = 0
    row: int = 0
    view: int = 0
    carrying: str = ""

@dataclass(frozen=True, eq=True)
class State:
    colAgent: int
    rowAgent: int
    viewAgent: int
    carrying: str
    adversaries: tuple = field(default_factory=tuple)
    balls: tuple = field(default_factory=tuple)
    boxes: tuple = field(default_factory=tuple)
    keys: tuple = field(default_factory=tuple)
    doors: tuple = field(default_factory=tuple)
    lockeddoors: tuple = field(default_factory=tuple)

def to_state(ints, booleans):
    ints = {key:int(value) for key, value in ints.items()}
    any_carrying = dict()
    for formula, value in booleans.items():
        if not value: continue
        if "Carrying" in formula:
            pos = formula.find("Carrying")
            l = len("Carrying")
            any_carrying[formula[0:pos]] = formula[pos+l:]
    agentState = (ints["colAgent"], ints["rowAgent"], ints["viewAgent"], any_carrying.get("Agent", ""))
    adversaries = tuple()
    boxes = tuple()
    balls = tuple()
    keys = tuple()
    lockeddoors = tuple()
    doors = tuple()
    for color in COLOR_NAMES:
        color = color.capitalize()
        if "col" + color in ints:
            adversaries += (AdversaryState(color, ints["col"+color], ints["row"+color], ints["view"+color], carrying=any_carrying.get(color, "")),)
        if "col" + color + "Box" in ints:
            pass
        if "col" + color + "Key" in ints:
            identifier = color + "Key"
            balls += (KeyState(color, ints["col"+identifier], ints["row"+identifier]),)
        if color + "DoorOpen" in booleans:
            if booleans[color + "DoorOpen"]:
                doors += (DoorState(color, locked=False),)
            else:
                doors += (DoorState(color, locked=True),)
        elif color + "LockedDoorOpen" in booleans:
            assert False


    return State(*agentState, adversaries=adversaries, doors=doors)
