# Implementation of a simple evaluation framework for recommender systems algorithms
__author__ = 'mms'

# Load required modules
import csv
import numpy as np
from sklearn import cross_validation  # machine learning & evaluation module
import random
import helper # helper.py

# Parameters
UAM_FILE = "./data/C1ku_UAM.txt"                # user-artist-matrix (UAM)
ARTISTS_FILE = "./data/C1ku_idx_artists.txt"    # artist names for UAM
USERS_FILE = "./data/C1ku_idx_users.txt"        # user names for UAM

NF = 10              # number of folds to perform in cross-validation
K = 5
VERBOSE = True
VERBOSE_DEPTH = 2


# Function to read metadata (users or artists)
def read_from_file(filename):
    data = []
    with open(filename, 'r') as f:  # open file for reading
        reader = csv.reader(f, delimiter='\t')  # create reader
        headers = reader.next()  # skip header
        for row in reader:
            item = row[0]
            data.append(item)
    f.close()
    return data

def shortenUam(UAM, user_id, artists):
    # get playcount vector for current user u
    pc_vec = UAM[user_id,:]

    # Compute similarities as inner product between pc_vec of user and all users via UAM (assuming that UAM is already normalized)
    aidx_nz = np.nonzero(pc_vec)[0]
    aidx_test = np.setdiff1d(aidx_nz, artists)  # compute set difference between all artist indices of user and train indices gives test artist indices

    # Set to 0 the listening events of seed user for testing (in UAM; pc_vec just points to UAM, is thus automatically updated)
    UAM[user_id, aidx_test] = 0.0

    UAM[user_id, :] = UAM[user_id, :] / np.sum(UAM[user_id, :])

    return UAM

# /get_user_neighbors

def get_user_neighbors(UAM, user_id):
    # get playcount vector for current user u
    pc_vec = UAM[user_id,:]

    # Compute similarities as inner product between pc_vec of user and all users via UAM (assuming that UAM is already normalized)
    sim_users = np.inner(pc_vec, UAM)     # similarities between u and other users

    # Sort similarities to all others
    sort_idx = np.argsort(sim_users)        # sort in ascending order
    # print sort_idx

    # Select the closest neighbor to seed user u (which is the last but one; last one is user u herself!)
    # neighbor_idx = sort_idx[k:-1][0] k definiert anzahl der nachbarn
    neighbor_array = sort_idx[-(K + 1):-1]

    return {
        'pc_vec': pc_vec,
        'neighbor_array': neighbor_array,
        'sim_users': sim_users,
        'sort_idx': sort_idx
    }
# /get_user_neighbors

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

    u_aidx = np.nonzero(UAM[u_idx,:])[0]
    random_u_aidx = np.nonzero(UAM[random_u_idx,:])[0]

    # this will return new artists the target_user never heard about
    return np.setdiff1d(random_u_aidx, u_aidx)
# /recommend_random_artists_RB

def recommend_CF_our(UAM, user_id, artists):
    """
    Function that implements a CF recommender. It takes as input the UAM, metadata (artists and users),
    the index of the seed user (to make predictions for) and the indices of the seed user's training artists.
    It returns a list of recommended artist indices

    :param UAM:               user-artist-matrix
    :param seed_uidx:         user index of seed user

    :return: a list of recommended artist indices
    """


    users               = helper.read_csv(USERS_FILE)
    artists_array       = []
    neighbor_array      = get_user_neighbors(UAM, user_id)['neighbor_array']
    sim_users           = get_user_neighbors(UAM, user_id)['sim_users']
    artist_idx_u        = artists # indices of artists user u listened to
    total_artist_rating = {}

    for neighbor_index, neighbor in enumerate(neighbor_array, start = 1):
        a_neighbor = neighbor_array[-(neighbor_index)]

        if VERBOSE and VERBOSE_DEPTH == 2:
            print '    The ' + helper.number_to_text(neighbor_index) + ' closest user to ' + ' is ' + str(a_neighbor)

        artist_idx_n  = np.nonzero(UAM[a_neighbor,:]) # indices of artists user u's neighbor listened to
        artists_array += artist_idx_n[0].tolist()

    artists_unique = np.unique(artists_array)
    # artists_unique = np.setdiff1d(artist_idx_u, artists_unique)

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



# Function that implements a CF recommender. It takes as input the UAM, metadata (artists and users),
# the index of the seed user (to make predictions for) and the indices of the seed user's training artists.
# It returns a list of recommended artist indices
def recommend_CF(UAM, seed_uidx, seed_aidx_train):


    """
    DASSELBE create_training_UAM
    ausgelagert
    """


    # UAM               user-artist-matrix
    # seed_uidx         user index of seed user
    # seed_aidx_train   indices in UAM of training artists for seed user

    # Get playcount vector for seed user
    pc_vec = UAM[seed_uidx, :]

    # Remove information on test artists from seed's listening vector
    aidx_nz = np.nonzero(pc_vec)[0]                            # artists with non-zero listening events
    aidx_test = np.setdiff1d(aidx_nz, seed_aidx_train)         # compute set difference between all artist indices of user and train indices gives test artist indices
#    print aidx_test

    # Set to 0 the listening events of seed user for testing (in UAM; pc_vec just points to UAM, is thus automatically updated)
    UAM[seed_uidx, aidx_test] = 0.0

    # Seed user needs to be normalized again
    # Perform sum-to-1 normalization
    UAM[seed_uidx, :] = UAM[seed_uidx, :] / np.sum(UAM[seed_uidx, :])

    """
    /DASSELBE
    """

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
#    print "The closest user to user " + str(seed_uidx) + " is " + str(neighbor_idx) + "."
#    print "The closest user to user " + users[seed_uidx] + " is user " + users[neighbor_idx] + "."

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


# Function that implements dumb random recommender. It predicts a number of randomly chosen items.
# It returns a list of recommended artist indices.
def recommend_RB(artists_idx, no_items):
    # artists_idx           list of artist indices to draw random sample from
    # no_items              no of items to predict

    # Let's predict a number of random items that equal the number of items in the user's test set
    random_aidx = random.sample(artists_idx, no_items)

    # Return list of recommended artist indices
    return random_aidx

# Main program
if __name__ == '__main__':

    # Initialize variables to hold performance measures
    avg_prec = 0;       # mean precision
    avg_rec = 0;        # mean recall

    # Load metadata from provided files into lists
    artists = read_from_file(ARTISTS_FILE)
    users = read_from_file(USERS_FILE)
    # Load UAM
    UAM = np.loadtxt(UAM_FILE, delimiter='\t', dtype=np.float32)

    # For all users in our data (UAM)
    no_users = UAM.shape[0]

    for u in range(0, no_users):
        # Get seed user's artists listened to
        u_aidx = np.nonzero(UAM[u, :])[0]

        # ignore users with less artists than the crossfold validation split maximum | NF
        if NF >= len(u_aidx) or u == no_users - 1:
            continue

        # Split user's artists into train and test set for cross-fold (CV) validation
        fold = 0

        kf = cross_validation.KFold(len(u_aidx), n_folds=NF)  # create folds (splits) for 5-fold CV
        for train_aidx, test_aidx in kf:  # for all folds

            copy_UAM = shortenUam(UAM.copy(), u, train_aidx) # we need to create a copy of the UAM, otherwise modifications within recommend function will effect the variable9
            # Show progress
            print "User: " + str(u) + ", Fold: " + str(fold) + ", Training items: " + str(
                len(train_aidx)) + ", Test items: " + str(len(test_aidx)),      # the comma at the end avoids line break
            # Call recommend function
            #rec_aidx = recommend_CF(copy_UAM, u, u_aidx[train_aidx])

            # K = 1
            # K = 3
            # K = 5
            # K = 10
            # K = 20
            #rec_aidx = recommend_CF_our(copy_UAM, u, u_aidx[train_aidx])

            #rec_aidx = recommend_random_artists_RB(copy_UAM, u, u_aidx[test_aidx])

            # For random recommendation, exclude items that the user already knows, i.e. the ones in the training set
            all_aidx = range(0, UAM.shape[1])
            #rec_aidx = recommend_RB(np.setdiff1d(all_aidx, u_aidx[train_aidx]), 1)       # select the number of recommended items as the number of items in the test set
            #rec_aidx = recommend_RB(np.setdiff1d(all_aidx, u_aidx[train_aidx]), 5)
            #rec_aidx = recommend_RB(np.setdiff1d(all_aidx, u_aidx[train_aidx]), 10)
            #rec_aidx = recommend_RB(np.setdiff1d(all_aidx, u_aidx[train_aidx]), 20)
            rec_aidx = recommend_RB(np.setdiff1d(all_aidx, u_aidx[train_aidx]), 50)
            #rec_aidx = recommend_RB(np.setdiff1d(all_aidx, u_aidx[train_aidx]), 100)

            print "Recommended items: ", len(rec_aidx)

            # Compute performance measures
            correct_aidx = np.intersect1d(u_aidx[test_aidx], rec_aidx)          # correctly predicted artists
#            print 'Recommended artist-ids: ', rec_aidx
#            print 'True artist-ids: ', u_aidx[test_aidx]

            # True Positives is amount of overlap in recommended artists and test artists
            TP = len(correct_aidx)
            # False Positives is recommended artists minus correctly predicted ones
            FP = len(np.setdiff1d(rec_aidx, correct_aidx))

            # Precision is percentage of correctly predicted among predicted
            # Handle special case that not a single artist could be recommended -> by definition, precision = 100%
            if len(rec_aidx) == 0:
                prec = 100.0
            else:
                prec = 100.0 * TP / len(rec_aidx)

            # Recall is percentage of correctly predicted among all listened to
            # Handle special case that there is no single artist in the test set -> by definition, recall = 100%
            if len(test_aidx) == 0:
                rec = 100.0
            else:
                rec = 100.0 * TP / len(test_aidx)


            # add precision and recall for current user and fold to aggregate variables
            avg_prec += prec / (NF * no_users)
            avg_rec += rec / (NF * no_users)

            # Output precision and recall of current fold
            print ("\tPrecision: %.2f, Recall:  %.2f" % (prec, rec))

            # Increase fold counter
            fold += 1

                # Output mean average precision and recall
    f1_score = 2 * ((avg_prec * avg_rec) / (avg_prec + avg_rec))

    print ("\nMAP: %.2f, MAR: %.2f, F1 Score: %.2f" % (avg_prec, avg_rec, f1_score))