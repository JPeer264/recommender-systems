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
    artist_object     = {}
    result = {}

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

    # for index, user in enumerate(all_artists_count, start=1):
    #     for test in all_artists_count[user]:
    #         try:
    #             result[user][test['artist_name']] = test['play_count']
    #         except:
    #             result[user] = {}
    #             result[user][test['artist_name']] = test['play_count']
    #             # result[user]
                
            
    #         # result[user]
            # result[[user]['artist_name']] = test[user]['play_count']
        # for artist in all_artists_count[user]
        #     print artist

    return all_artists_count
# /calc_artists_of_users

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

    users                                  = helper.read_csv(USERS_FILE)
    users_playcounts                       = get_user_artist_playcounts()
    artists                                = helper.read_csv(ARTISTS_FILE)
    artists_obj                            = {}
    artists_array                          = []
    artists_obj[user_name]                 = {}
    artists_obj[user_name]['id']           = user_id
    artists_obj[user_name]['neighbors_CF'] = {}
    neighbor_array                         = get_user_neighbors(UAM, user_id)['neighbor_array']
    sim_users                              = get_user_neighbors(UAM, user_id)['sim_users']
    artist_idx_u                           = np.nonzero(UAM[u,:]) # indices of artists user u listened to
    total_artist_rating                    = {}

    for neighbor_index, neighbor in enumerate(neighbor_array, start = 1):
        a_neighbor = neighbor_array[-(neighbor_index)]

        if VERBOSE and VERBOSE_DEPTH == 2:
            print '    The ' + helper.number_to_text(neighbor_index) + ' closest user to ' + user_name + ' is ' + users[a_neighbor]

        artist_idx_n  = np.nonzero(UAM[a_neighbor,:]) # indices of artists user u's neighbor listened to
        artists_array += artist_idx_n[0].tolist()

    artists_unique = helper.get_unique_items(artists_array)

    for artist in artists_unique:
        artist_count_of_neighbors = 0

        for neighbor_index, neighbor in enumerate(neighbor_array, start = 1):
            playcount_of_user = UAM[neighbor, artist]
            rating = playcount_of_user * sim_users[neighbor]

            if artist in total_artist_rating:
                total_artist_rating[artist] += rating
            else:
                total_artist_rating[artist] = rating

    # Return list of 10 recommended artist indices
    return sorted(total_artist_rating, key=total_artist_rating.__getitem__, reverse=True)[:10]
# /recommend_CF

def get_user_neighbors(UAM, user_id):
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

    return {
        'neighbor_array': neighbor_array,
        'sim_users': sim_users
    }
# /get_user_neighbors

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

        if VERBOSE:
            helper.log_highlight('Recommendation for ' + users[u] + ' [' + str(u + 1) + ' of ' + str(UAM.shape[0]) + ']')

        for i, r in enumerate(recommender, start = 1):
            if VERBOSE:
                print "The " + helper.number_to_text(i) + " is " + artists[r]
