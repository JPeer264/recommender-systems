import numpy as np
import random

MAX_ARTIST = 1
# Parameters
UAM_FILE = "./test_data/C1ku_UAM.txt"                # user-artist-matrix (UAM)
ARTISTS_FILE = "./data/C1ku_idx_artists.txt"    # artist names for UAM
USERS_FILE = "./data/C1ku_idx_users.txt"        # user names for UAM

def recommend_random_artists_RB(UAM, u_idx, train_aidx):
    """
    randomly generates a list of artists which the target_user never heard.
    It will compare the artists by a random generated user

    :param target_user: the username of the targetuser

    :return: an array with new artists
    """
    all_idx = range(0, UAM.shape[0])
    random_u_idx = random.sample(np.setdiff1d(all_idx, [u_idx]), 1)[0]

    # cannot generate the own user
    if random_u_idx == u_idx:
        recommend_random_artists_RB(UAM, u_idx)

    u_aidx = np.nonzero(UAM[u_idx, :])[0]
    random_u_aidx = np.nonzero(UAM[random_u_idx, :])[0]

    # this will return new artists the target_user never heard about
    result = np.setdiff1d(random_u_aidx, u_aidx)

    if len(result) > MAX_ARTIST:
        result = result[:MAX_ARTIST]

    return result
