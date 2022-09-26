############################### Basin #######################################
"""
Identifiers: NA - North Atlantic
             SA - South Atlantic
             EP - Eastern Pacific
             WP - Western Pacific
             SP - South Pacific
             NI - North Indian
             SI - South Indian
             GL - Global (no basin)
"""
valid_basins = ['NA', 'SA', 'EP', 'WP', 'SP', 'NI', 'SI', 'GL']
valid_basins_long = ['North Atlantic', 'South Atlantic', 'Eastern Pacific', 'Western Pacific',
                     'South Pacific', 'North Indian', 'South Indian', 'Global']

"""
The basin bounds dictionary maps a basin identifier to the basin boundaries.
The bounds are ordered as (LL - Lower Left, UR - Upper Right):
[LL Longitude, LL Latitude, UR Longitude, UR Latitude].
Note the basins bounds are extended slightly to allow tracks to
extend slightly beyond the bounds, as this may be the case when a TC starts
in one basin and goes to another.
"""
basin_bounds = {'NA': ['260E', '0N', '350E', '50N'],
                'SA': ['260E', '45S', '359.9E', '0N'],
                'EP': ['205E', '0N', '285E', '45N'],
                'WP': ['100E', '0N', '180E', '45N'],
                'SP': ['100E', '45S', '180E', '0N'],
                'NI': ['35E', '0N', '115E', '45N'],
                'SI': ['40E', '38S', '110E', '0N'],
                'GL': ['0E', '90S', '360E', '90N']}
