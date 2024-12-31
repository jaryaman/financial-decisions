"""
From https://www.ssa.gov/oact/STATS/table4c6.html
"""

age_to_death_probability_male: dict[int, float] = {
    0: 0.005860,
    1: 0.000420,
    2: 0.000272,
    3: 0.000225,
    4: 0.000184,
    5: 0.000157,
    6: 0.000140,
    7: 0.000128,
    8: 0.000122,
    9: 0.000123,
    10: 0.000129,
    11: 0.000138,
    12: 0.000164,
    13: 0.000220,
    14: 0.000310,
    15: 0.000446,
    16: 0.000637,
    17: 0.000868,
    18: 0.001100,
    19: 0.001270,
    20: 0.001373,
    21: 0.001488,
    22: 0.001605,
    23: 0.001714,
    24: 0.001835,
    25: 0.001963,
    26: 0.002082,
    27: 0.002202,
    28: 0.002330,
    29: 0.002457,
    30: 0.002574,
    31: 0.002683,
    32: 0.002787,
    33: 0.002881,
    34: 0.002974,
    35: 0.003074,
    36: 0.003175,
    37: 0.003295,
    38: 0.003444,
    39: 0.003608,
    40: 0.003780,
    41: 0.003958,
    42: 0.004144,
    43: 0.004337,
    44: 0.004540,
    45: 0.004774,
    46: 0.005064,
    47: 0.005399,
    48: 0.005796,
    49: 0.006214,
    50: 0.006671,
    51: 0.007167,
    52: 0.007736,
    53: 0.008351,
    54: 0.009035,
    55: 0.009770,
    56: 0.010567,
    57: 0.011398,
    58: 0.012291,
    59: 0.013224,
    60: 0.014267,
    61: 0.015353,
    62: 0.016484,
    63: 0.017617,
    64: 0.018759,
    65: 0.019914,
    66: 0.021104,
    67: 0.022423,
    68: 0.023847,
    69: 0.025357,
    70: 0.027050,
    71: 0.028970,
    72: 0.031188,
    73: 0.033754,
    74: 0.036747,
    75: 0.040563,
    76: 0.044308,
    77: 0.048498,
    78: 0.053229,
    79: 0.058778,
    80: 0.064617,
    81: 0.070947,
    82: 0.077834,
    83: 0.085686,
    84: 0.094809,
    85: 0.105090,
    86: 0.116592,
    87: 0.129306,
    88: 0.142732,
    89: 0.157638,
    90: 0.174458,
    91: 0.193027,
    92: 0.212930,
    93: 0.232657,
    94: 0.251826,
    95: 0.270943,
    96: 0.289756,
    97: 0.307998,
    98: 0.325393,
    99: 0.341662,
    100: 0.358746,
    101: 0.376683,
    102: 0.395517,
    103: 0.415293,
    104: 0.436058,
    105: 0.457860,
    106: 0.480753,
    107: 0.504791,
    108: 0.530031,
    109: 0.556532,
    110: 0.584359,
    111: 0.613577,
    112: 0.644256,
    113: 0.676468,
    114: 0.710292,
    115: 0.745806,
    116: 0.783097,
    117: 0.822251,
    118: 0.863364,
    119: 0.906532,
}


age_to_death_probability_female: dict[int, float] = {
    0: 0.005063,
    1: 0.000393,
    2: 0.000223,
    3: 0.000177,
    4: 0.000144,
    5: 0.000122,
    6: 0.000109,
    7: 0.000102,
    8: 0.000098,
    9: 0.000097,
    10: 0.000103,
    11: 0.000113,
    12: 0.000131,
    13: 0.000157,
    14: 0.000190,
    15: 0.000233,
    16: 0.000291,
    17: 0.000355,
    18: 0.000418,
    19: 0.000461,
    20: 0.000507,
    21: 0.000556,
    22: 0.000610,
    23: 0.000666,
    24: 0.000722,
    25: 0.000775,
    26: 0.000831,
    27: 0.000889,
    28: 0.000952,
    29: 0.001025,
    30: 0.001104,
    31: 0.001192,
    32: 0.001289,
    33: 0.001383,
    34: 0.001465,
    35: 0.001544,
    36: 0.001626,
    37: 0.001719,
    38: 0.001824,
    39: 0.001940,
    40: 0.002066,
    41: 0.002202,
    42: 0.002351,
    43: 0.002482,
    44: 0.002622,
    45: 0.002789,
    46: 0.002994,
    47: 0.003219,
    48: 0.003467,
    49: 0.003729,
    50: 0.004011,
    51: 0.004306,
    52: 0.004634,
    53: 0.004981,
    54: 0.005370,
    55: 0.005831,
    56: 0.006326,
    57: 0.006837,
    58: 0.007399,
    59: 0.008033,
    60: 0.008687,
    61: 0.009411,
    62: 0.010139,
    63: 0.010849,
    64: 0.011550,
    65: 0.012216,
    66: 0.012952,
    67: 0.013844,
    68: 0.014863,
    69: 0.016028,
    70: 0.017329,
    71: 0.018859,
    72: 0.020609,
    73: 0.022620,
    74: 0.024958,
    75: 0.027906,
    76: 0.030925,
    77: 0.034140,
    78: 0.037620,
    79: 0.041725,
    80: 0.046324,
    81: 0.051334,
    82: 0.056911,
    83: 0.063279,
    84: 0.070704,
    85: 0.079184,
    86: 0.088697,
    87: 0.099240,
    88: 0.110480,
    89: 0.123078,
    90: 0.137152,
    91: 0.152605,
    92: 0.169494,
    93: 0.187623,
    94: 0.206647,
    95: 0.225890,
    96: 0.245054,
    97: 0.263815,
    98: 0.281828,
    99: 0.298738,
    100: 0.316662,
    101: 0.335662,
    102: 0.355802,
    103: 0.377150,
    104: 0.399779,
    105: 0.423766,
    106: 0.449192,
    107: 0.476143,
    108: 0.504712,
    109: 0.534994,
    110: 0.567094,
    111: 0.601120,
    112: 0.637187,
    113: 0.675418,
    114: 0.710292,
    115: 0.745806,
    116: 0.783097,
    117: 0.822251,
    118: 0.863364,
    119: 0.906532,
}


age_to_life_expectancy_male: dict[int, float] = {
    0: 73.54,
    1: 72.97,
    2: 72,
    3: 71.02,
    4: 70.04,
    5: 69.05,
    6: 68.06,
    7: 67.07,
    8: 66.08,
    9: 65.09,
    10: 64.1,
    11: 63.1,
    12: 62.11,
    13: 61.12,
    14: 60.14,
    15: 59.16,
    16: 58.18,
    17: 57.22,
    18: 56.27,
    19: 55.33,
    20: 54.4,
    21: 53.47,
    22: 52.55,
    23: 51.64,
    24: 50.72,
    25: 49.82,
    26: 48.91,
    27: 48.01,
    28: 47.12,
    29: 46.23,
    30: 45.34,
    31: 44.46,
    32: 43.57,
    33: 42.69,
    34: 41.82,
    35: 40.94,
    36: 40.06,
    37: 39.19,
    38: 38.32,
    39: 37.45,
    40: 36.58,
    41: 35.72,
    42: 34.86,
    43: 34,
    44: 33.15,
    45: 32.3,
    46: 31.45,
    47: 30.61,
    48: 29.77,
    49: 28.94,
    50: 28.12,
    51: 27.3,
    52: 26.5,
    53: 25.7,
    54: 24.91,
    55: 24.14,
    56: 23.37,
    57: 22.61,
    58: 21.87,
    59: 21.13,
    60: 20.41,
    61: 19.7,
    62: 19,
    63: 18.31,
    64: 17.63,
    65: 16.95,
    66: 16.29,
    67: 15.63,
    68: 14.98,
    69: 14.33,
    70: 13.69,
    71: 13.06,
    72: 12.43,
    73: 11.82,
    74: 11.21,
    75: 10.62,
    76: 10.05,
    77: 9.49,
    78: 8.95,
    79: 8.42,
    80: 7.92,
    81: 7.43,
    82: 6.96,
    83: 6.5,
    84: 6.07,
    85: 5.65,
    86: 5.26,
    87: 4.88,
    88: 4.53,
    89: 4.21,
    90: 3.9,
    91: 3.62,
    92: 3.36,
    93: 3.14,
    94: 2.94,
    95: 2.76,
    96: 2.6,
    97: 2.45,
    98: 2.32,
    99: 2.2,
    100: 2.09,
    101: 1.98,
    102: 1.87,
    103: 1.77,
    104: 1.67,
    105: 1.58,
    106: 1.49,
    107: 1.4,
    108: 1.32,
    109: 1.24,
    110: 1.16,
    111: 1.09,
    112: 1.01,
    113: 0.95,
    114: 0.88,
    115: 0.82,
    116: 0.76,
    117: 0.7,
    118: 0.65,
    119: 0.6,
}


age_to_life_expectancy_female: dict[int, float] = {
    0: 79.3,
    1: 78.7,
    2: 77.74,
    3: 76.75,
    4: 75.77,
    5: 74.78,
    6: 73.79,
    7: 72.79,
    8: 71.8,
    9: 70.81,
    10: 69.82,
    11: 68.82,
    12: 67.83,
    13: 66.84,
    14: 65.85,
    15: 64.86,
    16: 63.88,
    17: 62.9,
    18: 61.92,
    19: 60.94,
    20: 59.97,
    21: 59,
    22: 58.03,
    23: 57.07,
    24: 56.11,
    25: 55.15,
    26: 54.19,
    27: 53.23,
    28: 52.28,
    29: 51.33,
    30: 50.38,
    31: 49.44,
    32: 48.5,
    33: 47.56,
    34: 46.62,
    35: 45.69,
    36: 44.76,
    37: 43.83,
    38: 42.91,
    39: 41.98,
    40: 41.07,
    41: 40.15,
    42: 39.24,
    43: 38.33,
    44: 37.42,
    45: 36.52,
    46: 35.62,
    47: 34.73,
    48: 33.84,
    49: 32.95,
    50: 32.07,
    51: 31.2,
    52: 30.33,
    53: 29.47,
    54: 28.62,
    55: 27.77,
    56: 26.93,
    57: 26.1,
    58: 25.27,
    59: 24.46,
    60: 23.65,
    61: 22.86,
    62: 22.07,
    63: 21.29,
    64: 20.52,
    65: 19.75,
    66: 18.99,
    67: 18.23,
    68: 17.48,
    69: 16.74,
    70: 16,
    71: 15.27,
    72: 14.56,
    73: 13.85,
    74: 13.16,
    75: 12.49,
    76: 11.83,
    77: 11.19,
    78: 10.57,
    79: 9.96,
    80: 9.38,
    81: 8.81,
    82: 8.26,
    83: 7.73,
    84: 7.21,
    85: 6.72,
    86: 6.26,
    87: 5.82,
    88: 5.41,
    89: 5.02,
    90: 4.65,
    91: 4.31,
    92: 3.99,
    93: 3.71,
    94: 3.45,
    95: 3.22,
    96: 3.01,
    97: 2.82,
    98: 2.66,
    99: 2.5,
    100: 2.35,
    101: 2.21,
    102: 2.08,
    103: 1.95,
    104: 1.82,
    105: 1.71,
    106: 1.59,
    107: 1.49,
    108: 1.39,
    109: 1.29,
    110: 1.2,
    111: 1.11,
    112: 1.03,
    113: 0.95,
    114: 0.88,
    115: 0.82,
    116: 0.76,
    117: 0.7,
    118: 0.65,
    119: 0.6,
}
