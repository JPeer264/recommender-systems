import numpy as np
import random

MAX_ARTIST = 1
# Parameters
UAM_FILE = "./test_data/C1ku/C1ku_UAM.txt"                # user-artist-matrix (UAM)
ARTISTS_FILE = "./data/C1ku_artists_extended.txt"    # artist names for UAM
USERS_FILE = "./data/C1ku_user_extended.txt"        # user names for UAM


# Function that implements a dumb random recommender. It predicts a number of randomly chosen items.
# It returns a dictionary of recommended artist indices (and corresponding scores).
def recommend_RB(artists_idx, no_items):
    """

    :param artists_idx: list of artist indices to draw random sample from
    :param no_items: no of items to predict
    :return:
    """

    # Let's predict a number of random items that equal the number of items in the user's test set
    random_aidx = random.sample(artists_idx, no_items)

    # Insert scores into dictionary
    dict_random_aidx = {}
    for aidx in random_aidx:
        dict_random_aidx[aidx] = 1.0  # for random recommendations, all scores are equal

    # Return dict of recommended artist indices as keys (and scores as values)
    return dict_random_aidx
