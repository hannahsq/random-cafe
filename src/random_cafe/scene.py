"""Scene description: what is in the room, where, and how loud. Every field has a default, so `Scene()` renders a
plausible indoor café; ranges are drawn per render from the scene's seed."""
from dataclasses import dataclass, field


@dataclass
class Room:
    indoor: bool = True
    t60_s: tuple[float, float] = (0.3, 0.9)  # reverberation time, drawn once per scene (one room for every source)
    drr_db: tuple[float, float] = (0.0, 10.0)  # direct-to-reverberant ratio for a source 1 m from the microphone
    transmission_db: tuple[float, float] = (25.0, 40.0)  # wall / window loss for outdoor sources heard indoors


@dataclass
class Babble:
    """Tables of talkers taking turns, from a talker pool the caller supplies."""
    enabled: bool = True
    level_db: float = 0.0
    tables: tuple[int, int] = (2, 6)
    per_table: tuple[int, int] = (2, 4)
    radius_m: tuple[float, float] = (1.5, 12.0)  # table distances, drawn uniformly over floor area
    gap_s: tuple[float, float] = (0.25, 0.35)  # turn gap: mean and SD; negative gaps are overlaps
    lombard_db: tuple[float, float] = (0.0, 6.0)  # per-talker vocal effort raise (level only)


@dataclass
class Clatter:
    """Cups, cutlery, glass: modal grains (decaying resonances) at random places."""
    enabled: bool = True
    level_db: float = -8.0
    rate_hz: float = 1.5
    radius_m: tuple[float, float] = (1.0, 10.0)


@dataclass
class Traffic:
    """Passing vehicles (tyre noise and engine harmonics on a lane, with Doppler) and a distant road bed."""
    enabled: bool = False
    level_db: float = -4.0
    vehicles_per_min: float = 12.0
    lane_distance_m: tuple[float, float] = (4.0, 30.0)
    speed_kmh: tuple[float, float] = (20.0, 60.0)
    bed_db: float = -10.0  # distant traffic relative to the passing vehicles


@dataclass
class Rain:
    """Drop impacts (ticks) and Minnaert bubbles on the ground outdoors, or on window glass plus the transmitted
    outdoor rain indoors."""
    enabled: bool = False
    level_db: float = -6.0
    rain_mm_h: float = 5.0


@dataclass
class Bed:
    """Recorded beds (room tone, espresso machine, ...) from files the caller supplies; looped from a random offset."""
    level_db: float = -12.0
    paths: tuple[str, ...] = ()


@dataclass
class Scene:
    duration_s: float = 10.0
    fs: int = 16000
    seed: int = 0
    mic: tuple[float, float, float] = (0.0, 0.0, 1.2)
    reference_dbfs: float = -30.0  # a stem at level_db 0 has this RMS
    room: Room = field(default_factory=Room)
    babble: Babble = field(default_factory=Babble)
    clatter: Clatter = field(default_factory=Clatter)
    traffic: Traffic = field(default_factory=Traffic)
    rain: Rain = field(default_factory=Rain)
    bed: Bed = field(default_factory=Bed)
