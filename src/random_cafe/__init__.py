"""random-cafe: seeded background scenes (café babble from real talkers, clatter, kitchen, traffic, rain, wind, recorded beds),
rendered as separate stems at one microphone."""
from .render import default_load, mix, render
from .scene import Babble, Bed, Clatter, Kitchen, Rain, Room, Scene, Traffic, Wind

__all__ = ["render", "mix", "default_load", "Scene", "Room", "Babble", "Clatter", "Kitchen", "Traffic", "Rain", "Wind", "Bed"]
__version__ = "0.1.0"
