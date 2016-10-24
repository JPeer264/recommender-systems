# Implementation of a simple user-based CF recommender
__author__ = 'mms'

# Load required modules
import csv
import numpy as np
import random
import helper # helper.py
import sys

reload(sys)
sys.setdefaultencoding('utf-8')

VERBOSE = True
VERBOSE_DEPTH = 1

# Parameters
LE_FILE      = "../Task1/output/user_info/listening_history.txt"
OUTPUT_DIR   = './output'
UAM_FILE     = OUTPUT_DIR + "/UAM.txt"           # user-artist-matrix (UAM)
ARTISTS_FILE = OUTPUT_DIR + "/UAM_artists_5.txt" # artist names for UAM
USERS_FILE   = OUTPUT_DIR + "/UAM_users_5.txt"   # user names for UAM


K = 4  # Neighbours

artists_user_one = []
artists_user_two = []

def save_artists_for_two_users(user_one, user_two):
    """
    TODO implement user_one into if

    fills artists_user_one and artists_user_two global arrays with data

    :param user_one: username of the first person
    :param user_two: username of the second person
    """
    # Read listening events from provided file
    with open(LE_FILE, 'r') as f:
        reader  = csv.reader(f, delimiter='\t')      # create reader
        headers = reader.next()                     # skip header

        for row in reader:
            user   = row[0]
            artist = row[2]

            if (user == 'Typheem'):
                artists_user_one.append(artist.encode('utf-8'))

            if (user == user_two):
                artists_user_two.append(artist.encode('utf-8'))
# /save_artists_for_two_users

def recommend_random_artists_RB(target_user):
    """
    randomly generates a list of artists which the target_user never heard.
    It will compare the artists by a random generated user

    :param target_user: the username of the targetuser

    :return: an array with new artists
    """
    users       = helper.read_csv(USERS_FILE)
    random_user = random.sample(users, 1)[0]

    # cannot generate the own user
    if random_user == target_user:
        recommend_random_artists_RB(target_user)

    save_artists_for_two_users(target_user, random_user)

    # this will return new artists the target_user never heard about
    return np.setdiff1d(artists_user_two, artists_user_one)
# /recomment_random_artists_RB

def recommend_CF(UAM, seed_uidx, seed_aidx_train):
    """
    Function that implements a CF recommender. It takes as input the UAM, metadata (artists and users),
    the index of the seed user (to make predictions for) and the indices of the seed user's training artists.
    It returns a list of recommended artist indices

    :param UAM:               user-artist-matrix
    :param seed_uidx:         user index of seed user
    :param seed_aidx_train:   indices in UAM of training artists for seed user

    :return: a list of recommended artist indices
    """

    # Get playcount vector for seed user
    pc_vec = UAM[seed_uidx, :]

    # Remove information on test artists from seed's listening vector
    aidx_nz   = np.nonzero(pc_vec)[0] # artists with non-zero listening events
    aidx_test = np.setdiff1d(aidx_nz, seed_aidx_train)

    # Set to 0 the listening events of seed user for testing (in UAM; pc_vec just points to UAM, is thus automatically updated)
    UAM[seed_uidx, aidx_test] = 0.0

    # Seed user needs to be normalized again
    # Perform sum-to-1 normalization
    UAM[seed_uidx, :] = UAM[seed_uidx, :] / np.sum(UAM[seed_uidx, :])

    # Compute similarities as inner product between pc_vec of user and all users via UAM (assuming that UAM is normalized)
    sim_users = np.inner(pc_vec, UAM)  # similarities between u and other users

    # Alternatively, compute cosine similarities as inverse cosine distance between pc_vec of user and all users via UAM (assuming that UAM is normalized)
    #    sim_users = np.zeros(shape=(UAM.shape[0]), dtype=np.float32)
    #    for u in range(0, UAM.shape[0]):
    #        sim_users[u] = 1.0 - scidist.cosine(pc_vec, UAM[u,:])

    # Sort similarities to all others
    sort_idx = np.argsort(sim_users)  # sort in ascending order

    # Select the closest neighbor to seed user (which is the last but one; last one is user u herself!)
    neighbor_idx = sort_idx[-2:-1][0]
    # print "The closest user to user " + str(seed_uidx) + " is " + str(neighbor_idx) + "."
    # print "The closest user to user " + users[seed_uidx] + " is user " + users[neighbor_idx] + "."

    # Get all artist indices the seed user and her closest neighbor listened to, i.e., element with non-zero entries in UAM
    artist_idx_u = seed_aidx_train                      # indices of artists in training set user
    artist_idx_n = np.nonzero(UAM[neighbor_idx, :])     # indices of artists user u's neighbor listened to

    # Compute the set difference between seed user's neighbor and seed user,
    # i.e., artists listened to by the neighbor, but not by seed user.
    # These artists are recommended to seed user.

    # np.nonzero returns a tuple of arrays, so we need to take the first element only when computing the set difference
    recommended_artists_idx = np.setdiff1d(artist_idx_n[0], artist_idx_u)
    # or alternatively, convert to a numpy array by ...
    # artist_idx_np.setdiff1d(np.array(artist_idx_n), np.array(artist_idx_u))

    # Return list of recommended artist indices
    return recommended_artists_idx

# Main program
if __name__ == '__main__':
    print recommend_random_artists_RB(1)
    # Initialize variables
    #    artists = []   # artists
    #    users   = []   # users
    #    UAM     = []   # user-artist-matrix

    # Load metadata from provided files into lists
    artists = helper.read_csv(ARTISTS_FILE)
    users   = helper.read_csv(USERS_FILE)
#    print users
#    print artists

    # Load UAM - Konstruiert Matrix aus einem File
    UAM = np.loadtxt(UAM_FILE, delimiter='\t', dtype=np.float32)

    # For all users
    for u in range(0, UAM.shape[0]):
        # get playcount vector for current user u
        pc_vec = UAM[u,:]

        # Compute similarities as inner product between pc_vec of user and all users via UAM (assuming that UAM is already normalized)
        sim_users = np.inner(pc_vec, UAM)     # similarities between u and other users
        # print sim_users

        # Sort similarities to all others
        sort_idx = np.argsort(sim_users)        # sort in ascending order
        # print sort_idx

        # Select the closest neighbor to seed user u (which is the last but one; last one is user u herself!)
        # neighbor_idx = sort_idx[k:-1][0] k definiert anzahl der nachbarn
        neighbour_array = sort_idx[-(K + 1):-1]
        artists_array = []

        for index, neighbor in enumerate(neighbour_array, start = 1):
            a_neighbour = neighbour_array[-(index)]
            # print "The closest user to user " + str(u) + " is " + str(a_neighbour) + "."
            # print "The closest user to user " + users[u] + " is user " + users[a_neighbour] + "."

            # Get np.argsort(sim_users)l artist indices user u and her closest neighbor listened to, i.e., element with non-zero entries in UAM
            artist_idx_u = np.nonzero(UAM[u,:])                 # indices of artists user u listened to
            artist_idx_n = np.nonzero(UAM[a_neighbour,:])     # indices of artists user u's neighbor listened to

            # Compute the set difference between u's neighbor and u,
            # i.e., artists listened to by the neighbor, but not by u.
            # These artists can be recommended to u.

            # np.nonzero returns a tuple of arrays, so we need to take the first element only when computing the set difference
            recommended_artists_idx = np.setdiff1d(artist_idx_n[0], artist_idx_u[0])

            # Output recommendations
            # artist indices
            # print "Indices of the " + str(len(recommended_artists_idx)) + " recommended artists: ", recommended_artists_idx

            # artist names
            artist = np.asarray(artists)    # convert list of artists to array of artists (for convenient indexing)
            artists_array.append(artist)
            # print "Names of the " + str(len(recommended_artists_idx)) + " recommended artists: ", artist[recommended_artists_idx]
