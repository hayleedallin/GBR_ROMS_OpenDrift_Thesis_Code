# Adapted from OpenDrift LarvalSnappers (Copyright 2015, Knut-Frode Dagestad, MET Norway)
# Modified for Acropora millepora larval dispersal, Moore Reef cluster.
# Haylee Dallin, GENG4412

import numpy as np
from opendrift.models.oceandrift import Lagrangian3DArray, OceanDrift
import logging
from opendrift.config import CONFIG_LEVEL_ESSENTIAL, CONFIG_LEVEL_BASIC
from opendrift.models.physics_methods import (verticaldiffusivity_Sundby1983, verticaldiffusivity_Large1994)
import geopandas as gpd
from shapely.geometry import Point
from shapely import contains, points as shapely_points

logger = logging.getLogger(__name__)

class CoralLarvaeElement(Lagrangian3DArray):
    """
    Extending Lagrangian3DArray with relevant properties for
    Acropora millepora coral larvae.
    """
    variables = Lagrangian3DArray.add_variables([
        ('stage_duration', {'dtype': np.float32,       # age since spawning
                            'units': 'days',
                            'default': 0.})])


class CoralLarvae(OceanDrift):
    """
    Pelagic particle trajectory model representing dispersal of
    Acropora millepora larvae from buoyant egg-sperm bundle release.

    Background pelagic mortality can be evaluated live during the run (stochastic,
    constant daily hazard rate - see background_mortality()).
    Set background_mortality() = 0 to use PostProcessingV2 to process mortality (bundle method)
    Settlement and competency-window/depth checks are NOT evaluated during the run.
    Those remain assessed in post-processing against the full saved trajectory
    (o.history), since they depend on reef-contact geometry handled downstream.
    """

    ElementType = CoralLarvaeElement

    required_variables = {
        'x_sea_water_velocity': {'fallback': 0},
        'y_sea_water_velocity': {'fallback': 0},
        'x_wind': {'fallback': 0},
        'y_wind': {'fallback': 0},
        'sea_surface_height': {'fallback': 0},
        'land_binary_mask': {'fallback': None},
        'sea_floor_depth_below_sea_level': {'fallback': 50},
        'ocean_vertical_diffusivity': {'fallback': 1e-5, 'profiles': True},
        'ocean_mixed_layer_thickness': {'fallback': 20},  # where wind driven diffusivity diminishes
        'sea_water_temperature': {'fallback': 27, 'profiles': True},
        'sea_water_salinity': {'fallback': 35, 'profiles': True}
    }

    def __init__(self, *args, **kwargs):
        super(CoralLarvae, self).__init__(*args, **kwargs)

        self._add_config({
            'pelagic_larval_duration':
                {'type': 'float', 'default': 28.0,
                 'min': 0.0, 'max': 90.0, 'units': 'days',
                 'description': 'Total time larvae remain viable in the water column '
                                '(hard cutoff - particles deactivate here so the run '
                                'terminates; settlement/competency assessed in post-processing)',
                 'level': CONFIG_LEVEL_ESSENTIAL},
            'vertical_behaviour:buoyant_speed':
                {'type': 'float', 'default': 0.0,
                 'min': 0.0, 'max': 0.1, 'units': 'm/s',
                 'description': 'Constant passive upward buoyancy speed can be applied for the remainder of '
                                'the pelagic phase, once the surface-holding period '
                                '(surface_period_hours) has ended. Combines with turbulent '
                                'vertical_mixing (ROMS Kz / background_diffusivity) for the rest of the run. ',
                 'level': CONFIG_LEVEL_BASIC},
            'vertical_behaviour:surface_period_hours':
                {'type': 'float', 'default': 8.0,
                 'min': 0.0, 'max': 720.0, 'units': 'hours',
                 'description': 'Duration post-fertilisation over which larvae are held fully at the '
                                'surface: z is pinned to 0, with no buoyant push and no turbulent '
                                'vertical_mixing applied. After this period ends, '
                                'buoyant_speed and vertical_mixing both apply for the rest of the run.',
                 'level': CONFIG_LEVEL_BASIC},
            'vertical_behaviour:sink_speed':
                {'type': 'float', 'default': 0.0,
                 'min': 0.0, 'max': 0.1, 'units': 'm/s',
                 'description': 'Optional constant downward speed, nets against buoyant_speed for the '
                                'remainder of the pelagic phase (age >= surface_period_hours). Net '
                                'speed applied is (buoyant_speed - sink_speed). Defaults to 0, so '
                                'behaviour is unchanged unless explicitly set.',
                 'level': CONFIG_LEVEL_BASIC},
            'biology:background_mortality_rate':
                {'type': 'float', 'default': 0.05,
                 'min': 0.0, 'max': 1.0, 'units': 'day^-1',
                 'description': 'Constant instantaneous daily background mortality rate (M), can be applied '
                                'live each timestep as a stochastic per-particle death probability '
                                'P = 1 - exp(-M * dt_days). Best-estimate value from combined in situ '
                                'and laboratory data. Set to 0 to disable and post process mortality instead.',
                 'level': CONFIG_LEVEL_ESSENTIAL},
            'biology:mortality_seed':
                {'type': 'float', 'default': -1.0,
                 'min': -1.0, 'max': 2**31 - 1, 'units': '',
                 'description': 'Seed for the background-mortality random draws, for reproducible runs. '
                                'Use -1 for an unseeded (non-reproducible) draw.',
                 'level': CONFIG_LEVEL_BASIC},
            'general:max_shelf_depth':
                {'type': 'float', 'default': 250.0,
                 'min': 0.0, 'max': 6000.0, 'units': 'm',
                 'description': 'Deactivate a particle once the seafloor depth at its position '
                                '(sea_floor_depth_below_sea_level, from the reader) exceeds this value '
                                '- i.e. once it has drifted past the continental shelf break, out of '
                                'the GBR shelf domain of interest. This is a bathymetry-based offshore '
                                'boundary, independent of any lat/lon polygon set via '
                                'set_domain_boundary(). Set to a very large value (e.g. 1e6) to disable.'
                                'NOT USED IN 2026 STUDY but could be used in future research',
                 'level': CONFIG_LEVEL_BASIC},
        })

        self._set_config_default('drift:vertical_mixing', True)
        self._set_config_default('drift:vertical_advection', True)  # let terminal_velocity/buoyancy act

        self._mortality_rng = None  # lazily created on first call to background_mortality()

    def update_stage(self):
        """Advance larval age and apply the hard PLD cutoff. No settlement/mortality here."""
        days_in_timestep = self.time_step.total_seconds() / 86400.0
        self.elements.stage_duration += days_in_timestep

        pld = self.get_config('pelagic_larval_duration')

        # hard PLD cutoff - only reason particles deactivate here, so the
        # simulation has a defined end point. Settlement/competency are NOT
        # evaluated live - handled in post-processing. Background mortality
        # is handled separately, in background_mortality().
        past_pld = self.elements.stage_duration > pld
        if past_pld.any():
            self.deactivate_elements(past_pld, reason='pelagic_larval_duration_exceeded')

    def background_mortality(self):
        """
        Apply constant background pelagic mortality, live, each timestep.

        Treats 'biology:background_mortality_rate' (M, day^-1) as an
        instantaneous hazard rate and converts it to a per-timestep death
        probability assuming exponential decay of the surviving cohort:

            P(death in dt) = 1 - exp(-M * dt_days)

        Each active particle then gets an independent random draw; particles
        that draw below P are deactivated this timestep. This reproduces the
        expected exponential survival curve N(t) = N0 * exp(-M*t) in
        aggregate, while keeping mortality a per-particle, per-timestep event
        rather than a single post-hoc culling pass.
        """
        mortality_rate = self.get_config('biology:background_mortality_rate')
        if mortality_rate is None or mortality_rate <= 0:
            return

        n_active = self.num_elements_active()
        if n_active == 0:
            return

        if self._mortality_rng is None:
            seed = self.get_config('biology:mortality_seed')
            self._mortality_rng = np.random.default_rng(None if seed is None or seed < 0 else int(seed))

        dt_days = self.time_step.total_seconds() / 86400.0
        prob_death = 1.0 - np.exp(-mortality_rate * dt_days)

        draw = self._mortality_rng.random(n_active)
        dies = draw < prob_death

        if dies.any():
            self.deactivate_elements(dies, reason='background_mortality')
            logger.debug(f'Background mortality: deactivated {dies.sum()} of {n_active} '
                         f'active particles (P={prob_death:.5f} this timestep)')

    def vertical_behaviour(self):
        """
        Multi-phase vertical behaviour:

          1. Surface-held (age < surface_period_hours): larvae are pinned
             exactly at the surface (z=0). No buoyant push and no turbulent
             mixing act during this phase -- see update(), which re-pins
             these particles to z=0 after vertical_mixing() runs, since
             OpenDrift's vertical_mixing() has no per-particle opt-out.
          2. Passive buoyancy or sink speed can be applied with this function too(age >= surface_period_hours):
             can be set up to test passive upward buoyancy too
        Stores self._surface_phase_mask each call so update() knows which
        particles to re-pin after vertical_mixing().
        """
        buoyant_speed = self.get_config('vertical_behaviour:buoyant_speed')
        sink_speed = self.get_config('vertical_behaviour:sink_speed')
        surface_period_hours = self.get_config('vertical_behaviour:surface_period_hours')
        net_speed = buoyant_speed - sink_speed

        age_hours = self.elements.age_seconds / 3600.0
        self._surface_phase_mask = age_hours < surface_period_hours
        post_surface = ~self._surface_phase_mask

        dt = self.time_step.total_seconds()

        if np.any(self._surface_phase_mask):
            self.elements.z[self._surface_phase_mask] = 0.0

        if np.any(post_surface):
            self.elements.z[post_surface] = np.minimum(
                0, self.elements.z[post_surface] + net_speed * dt)

    def set_domain_boundary(self, boundary_polygon):
        """
        Register a bounding polygon. Particles that leave this polygon are
        deactivated each timestep. Use this to represent domain edges,
        continental shelf break, or north/south limits - whatever your
        boundary_polygon.shp encodes.

        boundary_polygon : GeoDataFrame or single shapely Polygon, in EPSG:4326
        Used to deactivate particles when they reach ROMS domain edge
        """
        if isinstance(boundary_polygon, gpd.GeoDataFrame):
            boundary_polygon = boundary_polygon.to_crs(epsg=4326)
            # dissolve to a single geometry in case the shapefile has multiple features
            self._domain_geom = boundary_polygon.geometry.union_all()
        else:
            self._domain_geom = boundary_polygon

        logger.info(f'Registered domain boundary polygon for deactivation')

    def check_domain_boundary(self):
        """Deactivate any active particle whose position has left the registered domain."""
        if not hasattr(self, '_domain_geom') or self._domain_geom is None:
            return

        active = np.where(self.elements.status == 0)[0] if hasattr(self.elements, 'status') else np.arange(self.num_elements_active())
        if len(active) == 0:
            return

        pts = shapely_points(self.elements.lon, self.elements.lat)
        inside = contains(self._domain_geom, pts)  # vectorized, one call for all particles

        left_domain = ~inside
        if left_domain.any():
            self.deactivate_elements(left_domain, reason='left_domain')

    def update(self):
        self.update_stage()
        self.background_mortality()
        self.advect_ocean_current()
        self.vertical_behaviour()
        self.check_domain_boundary()
        #self.check_shelf_depth()
        if self.get_config('drift:vertical_mixing') is True:
            self.vertical_mixing()
            if hasattr(self, '_surface_phase_mask') and np.any(self._surface_phase_mask):
                self.elements.z[self._surface_phase_mask] = 0.0
        elif self.get_config('drift:vertical_advection') is True:
            self.vertical_advection()