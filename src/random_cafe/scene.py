"""Scene description: what is in the room, where, and how loud. Every field has a default, so `Scene()` renders a
plausible indoor café; ranges are drawn per render from the scene's seed."""
from dataclasses import dataclass, field


@dataclass
class Room:
    indoor: bool = True
    t60_s: tuple[float, float] = (0.3, 0.9)  # reverberation time, drawn once per scene (one room for every source)
    drr_db: tuple[float, float] = (0.0, 10.0)  # direct-to-reverberant ratio for a source 1 m from the microphone
    # outdoor sources heard indoors come in through the facade: windows (the weakest part of a wall, so they set the
    # loss) and, often in a café, an open front door (an opening: little loss, almost no filtering). Losses are facade
    # insulation, outdoor level minus indoor level, so they already include spreading and the room
    # heuristic: a cafe's facade is leaky (windows ajar, gaps, often single glazing), so traffic and rain stay
    # audible indoors, as wanted for augmentation (a sealed modern facade takes 25-30 dB(A))
    window_db: tuple[float, float] = (6.0, 12.0)
    window_corner_hz: float = 400.0
    p_door_open: float = 0.5
    door_db: tuple[float, float] = (2.0, 6.0)
    facade_distance_m: tuple[float, float] = (3.0, 8.0)  # from the facade to the microphone


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
class Kitchen:
    """Extractor fan, dish clatter and espresso steam at one place; open (shares the room) or closed (through a
    wall / door)."""
    enabled: bool = False
    open: bool = True
    level_db: float = -8.0
    distance_m: tuple[float, float] = (5.0, 12.0)
    clatter_rate_hz: float = 3.0
    steam_rate_hz: float = 0.08
    wall_db: tuple[float, float] = (6.0, 14.0)  # a door or partition, less than an outside wall (often ajar)


@dataclass
class Traffic:
    """A street past the microphone: one or two lanes (the second running the other way) of vehicles following each
    other, each with its own driver, optionally stopping at a pedestrian crossing's lights; engines, exhausts and
    tyres driven by the simulated speed and load, heard with Doppler; and a distant road bed."""
    enabled: bool = False
    level_db: float = 0.0  # a busy street at the facade is about as loud as a busy café (both ~65-75 dB(A))
    vehicles_per_min: float = 12.0  # both lanes together
    lanes: int = 2
    lane_distance_m: tuple[float, float] = (4.0, 30.0)  # to the near lane, drawn per scene
    speed_kmh: tuple[float, float] = (30.0, 60.0)  # drivers' desired speeds
    headway_s: tuple[float, float] = (1.6, 0.4)  # drivers' time gap to the vehicle ahead: mean, SD
    p_heavy: float = 0.1  # share of prime movers (trucks, buses: manual diesels)
    p_diesel: float = 0.3  # share of cars with diesel engines (the rest petrol; all automatics)
    density_depth: float = 0.5  # slow swings in the flow (platoons, quiet spells), log-SD of the arrival rate
    density_time_s: float = 30.0
    lights: bool = False  # a pedestrian crossing with traffic lights
    crossing_m: tuple[float, float] = (-60.0, 60.0)  # where it is along the street, drawn per scene
    green_s: float = 40.0
    red_s: float = 20.0
    bed_db: float = -10.0  # distant traffic relative to the passing vehicles


@dataclass
class Rain:
    """Drop impacts (click and splash) on the ground outdoors, Minnaert bubbles from large drops landing in puddles;
    indoors, drops on a window pane plus the transmitted outdoor rain."""
    enabled: bool = False
    level_db: float = -6.0
    rain_mm_h: float = 5.0
    radius_m: float = 20.0  # rain heard out to this distance (every log-spaced ring carries similar power)
    puddles: float = 0.1  # share of the ground under standing water
    grains_hz: float = 500.0  # per drop-size bin: the nearest drops one by one, the rest as a dense continuum
    gust_depth: float = 0.25  # slow rain-rate fluctuation, fraction of the mean
    gust_time_s: float = 4.0  # its correlation time
    # heuristic: a drop on glass shakes the whole pane, which radiates into the room far better than a splash on the
    # ground radiates into the air; chosen so a windward window is heard well but rarely
    # drowns the room (to be checked by ear)
    pane_db: float = 20.0


@dataclass
class Wind:
    """Turbulence buffeting the microphone outdoors, and the rush of wind around buildings and trees. The speed is
    the scene's weather: rain reads it too (wind drives rain onto the windows) whether or not this layer is on."""
    enabled: bool = False
    level_db: float = -6.0
    speed_ms: float = 4.0  # mean wind speed at the microphone (a moderate breeze)
    gust_depth: float = 0.3  # turbulence intensity: speed fluctuation as a fraction of the mean
    gust_time_s: float = 5.0  # correlation time of the gusts


@dataclass
class Bed:
    """Recorded beds (room tone, espresso machine, ...) from files the caller supplies; looped from a random offset."""
    level_db: float = -12.0
    paths: tuple[str, ...] = ()


# 16 kHz for speech models; 44.1 or 48 kHz for full-band ones (a small turbo's whistle, the top of rain and hiss).
# Any rate renders; at these the layers' levels agree within about a dB (tested), so one calibration serves them all
SAMPLE_RATES = (16000, 44100, 48000)


@dataclass
class Scene:
    duration_s: float = 10.0
    fs: int = 16000  # see SAMPLE_RATES
    seed: int = 0
    mic: tuple[float, float, float] = (0.0, 0.0, 1.2)
    listener_yaw_deg: float = 90.0  # binaural renders only: the way the head faces (90: +y, the street side)
    reference_dbfs: float = -30.0  # a stem at level_db 0 has this RMS
    room: Room = field(default_factory=Room)
    babble: Babble = field(default_factory=Babble)
    clatter: Clatter = field(default_factory=Clatter)
    kitchen: Kitchen = field(default_factory=Kitchen)
    traffic: Traffic = field(default_factory=Traffic)
    rain: Rain = field(default_factory=Rain)
    wind: Wind = field(default_factory=Wind)
    bed: Bed = field(default_factory=Bed)
