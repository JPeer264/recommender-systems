# Load required modules
import csv
import json
import numpy as np
import random
import helper # helper.py
import sys

reload(sys)
sys.setdefaultencoding('utf-8')

VERBOSE = True # set to True prints information about the current state into the console
VERBOSE_DEPTH = 3  # describes how deep the verbose mode goes - maximum 4 - recommended 3

# Parameters
LE_FILE      = "../Task1/output/user_info/listening_history.txt"
OUTPUT_DIR   = './output'
UAM_FILE     = OUTPUT_DIR + "/UAM.txt"           # user-artist-matrix (UAM)
ARTISTS_FILE = OUTPUT_DIR + "/UAM_artists_5.txt" # artist names for UAM
USERS_FILE   = OUTPUT_DIR + "/UAM_users_5.txt"   # user names for UAM

K = 4  # Neighbors

artists_user_one = []
artists_user_two = []


def get_user_artist_playcounts():
    """
    calculates and sorts the playcounts of each user and its artist

    :return: a sorted array of all artists with given playcounts
    """
    all_artists_count = {}
    artists_file = {}
    artist_object = {}

    with open(LE_FILE, 'r') as f:
        reader  = csv.reader(f, delimiter='\t')      # create reader
        headers = reader.next()                     # skip header

        for row in reader:
            user   = row[0]
            artist = row[2]

            try:
                artist_object[user][artist] += 1
            except:
                try:
                    artist_object[user][artist] = 1
                except:
                    artist_object[user] = {}
                    artist_object[user][artist] = 1

        for user in artist_object:
            for artist in artist_object[user]:
                try:
                    all_artists_count[user].append({
                        'artist_name': artist,
                        'play_count': artist_object[user][artist]
                    })
                except:
                    all_artists_count[user] = []
                    all_artists_count[user].append({
                        'artist_name': artist,
                        'play_count': artist_object[user][artist]
                    })
    for index, user_name in enumerate(all_artists_count):
        sorted_artists = sorted(all_artists_count[user_name], key=lambda x: x['play_count'], reverse=True)
        all_artists_count[user_name] = sorted_artists

    return all_artists_count
# /calc_artists_of_users

# TODO method to combine the predictions for the same artists among the set of nearest neighbors
# def predict_artist(user_name_array, all_artists_count):
#     """
#     TODO:
#     1. search for all artists that Target and neigbors have listend to
#     2. search for all artists that all Neigbors have Listened to but Target not
#     3. normalize playcounts on all artists between 0-1
#     4. calculate the prdiction for the artists that Target have not listened to yet
#     """

#     #1
#     # for user_name in user_name_array:
#     relation_artists(all_artists_count[user_name_array[0]], all_artists_count[user_name_array[1]])
#     #return artist

# def relation_artists(user_one_artists, user_two_artists):
#     relation_artists = {}

#     for user_one_artist in user_one_artists:
#         for user_two_artist in user_two_artists:
#             if (user_one_artists[user_one_artist] == user_two_artists[user_two_artist]):
#                 print user_one_artist


def save_artists_for_two_users(user_one, user_two):
    """
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

            if (user == user_one):
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
# /recommend_random_artists_RB

def recommend_CF(UAM, user_id, user):
    """
    Function that implements a CF recommender. It takes as input the UAM, metadata (artists and users),
    the index of the seed user (to make predictions for) and the indices of the seed user's training artists.
    It returns a list of recommended artist indices

    :param UAM:               user-artist-matrix
    :param seed_uidx:         user index of seed user

    :return: a list of recommended artist indices
    """
    user_name = user[user_id]

    if VERBOSE:
        print 'CF recommendation for user [' + str(user_id + 1) + ' of ' + str(len(user)) + ']'

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
    neighbor_array = sort_idx[-(K + 1):-1]
    artists_array = []
    artists_obj = {}

    artists_obj[user_name] = {}
    artists_obj[user_name]['id'] = user_id
    artists_obj[user_name]['neighbors_CF'] = {}

    for neighbor_index, neighbor in enumerate(neighbor_array, start = 1):
        a_neighbor = neighbor_array[-(neighbor_index)]

        if VERBOSE and VERBOSE_DEPTH == 2:
            print '    The ' + helper.number_to_text(neighbor_index) + ' closest user to ' + user_name + ' is ' + users[a_neighbor]
        # print "The closest user to user " + str(u) + " is " + str(a_neighbor) + "."
        # print "The closest user to user " + users[u] + " is user " + users[a_neighbor] + "."

        # Get np.argsort(sim_users)l artist indices user u and her closest neighbor listened to, i.e., element with non-zero entries in UAM
        artist_idx_u = np.nonzero(UAM[u,:])                 # indices of artists user u listened to
        artist_idx_n = np.nonzero(UAM[a_neighbor,:])     # indices of artists user u's neighbor listened to

        # Compute the set difference between u's neighbor and u,
        # i.e., artists listened to by the neighbor, but not by u.
        # These artists can be recommended to u.

        # np.nonzero returns a tuple of arrays, so we need to take the first element only when computing the set difference
        recommended_artists_idx = np.setdiff1d(artist_idx_n[0], artist_idx_u[0])

        # artist names
        artist = np.asarray(artists)   # convert list of artists to array of artists (for convenient indexing)
        artists_obj[user_name]['neighbors_CF'][neighbor_index] = artist[recommended_artists_idx]
        artists_array.append(artists_obj)
        # print "Names of the " + str(len(recommended_artists_idx)) + " recommended artists: ", artist[recommended_artists_idx]

        if VERBOSE and VERBOSE_DEPTH == 3:
            print '        Recommended artists for the ' + helper.number_to_text(neighbor_index) + ' neighbor [' + str(len(artist[recommended_artists_idx]))  + ']'

            if VERBOSE and VERBOSE_DEPTH == 4:
                for value in artist[recommended_artists_idx]:
                    print '          ' + value

    # Return list of recommended artist indices
    return artists_obj
# /recommend_CF

# Main program
if __name__ == '__main__':
    # Initialize variables
    #    artists = []   # artists
    #    users   = []   # users
    #    UAM     = []   # user-artist-matrix

    # Load metadata from provided files into lists
    artists           = helper.read_csv(ARTISTS_FILE)
    users             = helper.read_csv(USERS_FILE)
    recommender_users = {}

    # Load UAM - Konstruiert Matrix aus einem File
    if VERBOSE:
        helper.log_highlight('Loading UAM')

    UAM = np.loadtxt(UAM_FILE, delimiter='\t', dtype=np.float32)

    if VERBOSE:
        print '\nSuccessfully read UAM\n'

    # For all users
    if VERBOSE:
        helper.log_highlight('Initialize CF recommendation for users')

    for u in range(0, UAM.shape[0]):
        recommender = recommend_CF(UAM, u, users)
        recommender_users[users[u]] = recommender[users[u]]

    if VERBOSE:
        print '\nCF recommendation complete\n'
        helper.log_highlight('Initialize RB recommendation for Sam00')

        print recommend_random_artists_RB('Sam00')
