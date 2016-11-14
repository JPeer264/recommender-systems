# Load required modules
import csv
import numpy as np
from sklearn import cross_validation  # machine learning & evaluation module
import random
import scipy.spatial.distance as scidist  # import distance computation module from scipy package
import os
import json

# Parameters
TESTFILES = "../test_data/"
TASK2_OUTPUT = "../Task02/output"
# User-artist-matrix (UAM)
UAM_FILE = TESTFILES + "C1ku/C1ku_UAM.txt"
AAM_FILE = TESTFILES + "AAM.txt"
# Artist names for UAM
ARTISTS_FILE = TESTFILES + "C1ku_artists_extended.csv"
# User names for UAM
USERS_FILE = TESTFILES + "C1ku_users_extended.csv"
# Recommendation method
METHOD = "HR_SEB"

MAX_USER = 50
MAX_ARTIST = 1000

VERBOSE = True

NF = 10  # number of folds to perform in cross-validation

MIN_RECOMMENDED_ARTISTS = 5


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


def recommend_CF(UAM, seed_uidx, seed_aidx_train):
    """
    Function that implements a CF recommender

    :param UAM: takes the UAM
    :param seed_uidx: index of the seed user (to make predictions for)
    :param seed_aidx_train: the indices of the seed user's training artists
    :return: a list of recommended artist indices
    """

    # Get playcount vector for seed user
    pc_vec = UAM[seed_uidx, :]

    ### Remove information on test artists from seed's listening vector ###
    # Artists with non-zero listening events
    aidx_nz = np.nonzero(pc_vec)[0]

    # Intersection between:
    # + all artist indices of user and
    # + train indices
    # --> gives test artist indices
    aidx_test = np.setdiff1d(aidx_nz, seed_aidx_train)

    # Set the listening events of seed user to 0
    # (in UAM; pc_vec just points to UAM, is thus automatically updated)
    UAM[seed_uidx, aidx_test] = 0.0

    # Seed user needs to be normalized again
    # Perform sum-to-1 normalization
    UAM[seed_uidx, :] = UAM[seed_uidx, :] / np.sum(UAM[seed_uidx, :])

    # Compute similarities as inverse cosine distance between:
    # + pc_vec of user and
    # + all users via UAM (assuming that UAM is normalized)
    sim_users = np.zeros(shape=(UAM.shape[0]), dtype=np.float32)
    for u in range(0, UAM.shape[0]):
        sim_users[u] = 1.0 - scidist.cosine(pc_vec, UAM[u, :])

    # ######################
    # Compute similarities #
    ########################
    # as inner product betweenpc_vec of user and all users via UAM (assuming that UAM is normalized)

    # Similarities between u and other users
    sim_users = np.inner(pc_vec, UAM)

    # Sort similarities to all others (ascending)
    sort_idx = np.argsort(sim_users)

    # Select the closest neighbor to seed user (which is the last but one; last one is user u herself!)
    neighbor_idx = sort_idx[-2:-1][0]

    # Get all artist indices the seed user and her closest neighbor listened to
    # e.g.: element with non-zero entries in UAM)

    # Indices of artists in training set user
    artist_idx_u = seed_aidx_train

    # Indices of artists user u's neighbor listened to
    artist_idx_n = np.nonzero(UAM[neighbor_idx, :])

    # Compute the set difference between seed user's neighbor and seed user,
    # i.e., artists listened to by the neighbor, but not by seed user.
    # These artists are recommended to seed user.

    # np.nonzero:
    # returns a tuple of arrays -> so we need to take the first element only when computing the set difference
    recommended_artists_idx = np.setdiff1d(artist_idx_n[0], artist_idx_u)

    if len(recommended_artists_idx) <= 5:
        reco_art_RB = recommend_RB(np.setdiff1d(range(0, AAM.shape[1]), seed_aidx_train),
                                   MIN_RECOMMENDED_ARTISTS - len(recommended_artists_idx))

        recommended_artists_idx = np.concatenate([recommended_artists_idx, reco_art_RB])

    print "CF recommended_artists_idx: "
    print "full-size: "
    print recommended_artists_idx

    recommended_artists_idx_shortened = recommended_artists_idx[1:MIN_RECOMMENDED_ARTISTS + 1]
    print "min-size: "
    print recommended_artists_idx_shortened
    return recommended_artists_idx_shortened


def recommend_CB(AAM, seed_aidx_train, K):
    """
    Function that implements a content-based recommender

    :param AAM: artist-artist-matrix containing pair-wise similarities
    :param seed_aidx_train: indices of the seed user's training artists
    :param K: no of neighbours to consider
    :return: number of nearest neighbors (artists) to consider for each seed artist
    """

    # ########################################################
    # Get nearest neighbors of train set artist of seed user #
    ##########################################################

    # Sort AAM column-wise for each row
    sort_idx = np.argsort(AAM[seed_aidx_train, :], axis=1)

    # Select the K closest artists to all artists the seed user listened to
    neighbor_idx = sort_idx[:, -1 - K:-1]

    # Aggregate the artists in neighbor_idx:
    # Count number of appearances of each artist-index in range [0, max(neighbor_idx.flatten())]
    nn_count = np.bincount(neighbor_idx.flatten())

    # Sort this count vector
    # We are interested in the last elements (the artists that appear most frequently as nearest neighbors)
    nn_count_sort_idx = np.argsort(nn_count)

    # Select all artists that appear as nearest neighbors among more than 5% of the user's training artists.
    threshold = np.int(np.round(len(seed_aidx_train) * 0.05))

    selected_artists_idx = np.where(nn_count_sort_idx > threshold)[0]

    recommended_artists_idx = np.setdiff1d(selected_artists_idx, seed_aidx_train)

    if len(recommended_artists_idx) <= MIN_RECOMMENDED_ARTISTS:
        reco_art_RB = recommend_RB(np.setdiff1d(range(0, AAM.shape[1]), seed_aidx_train),
                                   MIN_RECOMMENDED_ARTISTS - len(recommended_artists_idx))

        recommended_artists_idx = np.concatenate([recommended_artists_idx, reco_art_RB])

    return recommended_artists_idx

    print "CB recommended_artists_idx: "
    print "full-size: "
    print recommended_artists_idx

    recommended_artists_idx_shortened = recommended_artists_idx[1:MIN_RECOMMENDED_ARTISTS + 1]
    print "min-size: "
    print recommended_artists_idx_shortened
    return recommended_artists_idx_shortened


def run():
    """
    Function to run an evaluation experiment
    """
    # Initialize variables to hold performance measures
    avg_prec = 0  # mean precision
    avg_rec = 0  # mean recall

    # For all users in our data (UAM)
    no_users = UAM.shape[0]
    no_artists = UAM.shape[1]

    for u in range(0, no_users):

        # Get seed user's artists listened to
        u_aidx = np.nonzero(UAM[u, :])[0]

        if NF >= len(u_aidx) or u == no_users - 1:
            continue

        # Split user's artists into train and test set for cross-fold (CV) validation
        fold = 0

        # create folds (splits) for 10-fold CV
        kf = cross_validation.KFold(len(u_aidx), n_folds=NF)

        # For all folds
        for train_aidx, test_aidx in kf:
            if VERBOSE:
                print "User: " + str(u) + ", Fold: " + str(fold) + ", Training items: " + str(
                    len(train_aidx)) + ", Test items: " + str(len(test_aidx)),

            # Create a copy of the UAM, otherwise modifications within recommend function will effect the variable
            copy_UAM = UAM.copy()

            # Call recommend function
            rec_aidx_CF = recommend_CF(copy_UAM, u, u_aidx[train_aidx])
            rec_aidx_CB = recommend_CB(AAM, u_aidx[train_aidx], K)
            rec_aidx = np.intersect1d(rec_aidx_CB, rec_aidx_CF)

            if VERBOSE:
                print "Recommended items: ", len(rec_aidx)

            ################################
            # Compute performance measures #
            ################################

            # Correctly predicted artists
            correct_aidx = np.intersect1d(u_aidx[test_aidx], rec_aidx)

            # TP - True Positives is amount of overlap in recommended artists and test artists
            # FP - False Positives is recommended artists minus correctly predicted ones
            TP = len(correct_aidx)
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
            if VERBOSE:
                print ("\tPrecision: %.2f, Recall:  %.2f" % (prec, rec))

            # Increase fold counter
            fold += 1

    f1_score = 2 * ((avg_prec * avg_rec) / (avg_prec + avg_rec))

    # Output mean average precision and recall
    if VERBOSE:
        print ("\nMAP: %.2f, MAR  %.2f, F1 Scrore: %.2f" % (avg_prec, avg_rec, f1_score))
    print ("%.3f, %.3f" % (avg_prec, avg_rec))
    print ("K neighbors " + str(K))
    print ("Recommendation: " + str(MIN_RECOMMENDED_ARTISTS))

    data = {'avg_prec': avg_prec, 'avg_rec': avg_rec, 'f1_score': f1_score}

    return data


# Function that implements a dumb random recommender. It predicts a number of randomly chosen items.
# It returns a list of recommended artist indices
def recommend_RB(artists_idx, no_items):
    # artists_idx           list of artist indices to draw random sample from
    # no_items              no of items to predict

    # Let's predict a number of random items that equal the number of items in the user's test set
    random_aidx = random.sample(artists_idx, no_items)

    # Return list of recommended artist indices
    return random_aidx


# Main program
if __name__ == '__main__':

    # Load metadata from provided files into lists
    artists = read_from_file(ARTISTS_FILE)
    users = read_from_file(USERS_FILE)
    # Load UAM
    UAM = np.loadtxt(UAM_FILE, delimiter='\t', dtype=np.float32)[:MAX_USER, :MAX_ARTIST]
    AAM = np.loadtxt(AAM_FILE, delimiter='\t', dtype=np.float32)

    runned_methods = {METHOD: []}

    k_sorted = {}
    r_sorted = {}

    # data
    neighbors = [1, 2, 3, 5, 10, 20, 50]
    recommender_artists = [10, 20, 30, 50, 100, 200, 300]

    output_filedir = TASK2_OUTPUT + '/results/' + METHOD + '/'

    # ensure dir
    if not os.path.exists(output_filedir):
        os.makedirs(output_filedir)

    for neighbor in neighbors:
        k_sorted['K' + str(neighbor)] = []

        K = neighbor

        for recommender_artist in recommender_artists:
            k_sorted['R' + str(recommender_artist)] = []

            MIN_RECOMMENDED_ARTISTS = recommender_artist / 2

            # prepare for appending
            data_to_append = {'neighbors': K, 'recommended_artists': MIN_RECOMMENDED_ARTISTS}

            data = run()

            data_to_append.update(data)
            runned_methods[METHOD].append(data_to_append)

            # write into file
            content = json.dumps(data_to_append, indent=4, sort_keys=True)
            f = open(output_filedir + 'K' + str(K) + '_recommended' + str(MIN_RECOMMENDED_ARTISTS) + '.json', 'w')

            f.write(content)
            f.close()

    content = json.dumps(data_to_append, indent=4, sort_keys=True)
    f = open(output_filedir + 'all.json', 'w')

    f.write(content)
    f.close()
