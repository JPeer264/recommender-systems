import os
import json
import csv
import numpy as np
from sklearn import cross_validation  # machine learning & evaluation module
import random


# Parameters
TESTFILES = "../test_data/"
TASK2_OUTPUT = "../Task02/output"
# User-artist-matrix (UAM)
UAM_FILE = TESTFILES + "C1ku/C1ku_UAM.txt"
# Artist names for UAM
ARTISTS_FILE = TESTFILES + "C1ku_artists_extended.csv"
# User names for UAM
USERS_FILE = TESTFILES + "C1ku_users_extended.csv"
# Recommendation method
METHOD = "RB_Artists"

# Define no of artists that should be recommended
MIN_RECOMMENDED_ARTISTS = 300
MAX_RECOMMENDED_ARTISTS = MIN_RECOMMENDED_ARTISTS

# Number of folds to perform in cross-validation
NF = 10

# Verbose output?
VERBOSE = False


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


def RB_artists(artists_idx, no_items):
    """
    Function that implements a dumb random recommender. It predicts a number of randomly chosen items

    :param artists_idx:  list of artist indices
    :param no_items: no of items to predict
    :return: a dictionary of recommended artist indices (and corresponding scores)
    """

    # Let's predict a number of random items that equal the number of items in the user's test set
    random_aidx = random.sample(artists_idx, no_items)

    # Insert scores into dictionary
    dict_random_aidx = {}
    for aidx in random_aidx:
        dict_random_aidx[aidx] = 1.0  # for random recommendations, all scores are equal

    # Return dict of recommended artist indices as keys (and scores as values)
    return dict_random_aidx


def run():
    """
    Function to run an evaluation experiment

    :return: a dictionary with MAP, MAR and F1-Score for the tested recommender
    """

    # Initialize variables to hold performance measures
    avg_prec = 0;  # mean precision
    avg_rec = 0;  # mean recall

    # For all users in our data (UAM)
    no_users = UAM.shape[0]
    no_artists = UAM.shape[1]

    for u in range(0, no_users):

        # Get seed user's artists listened to
        u_aidx = np.nonzero(UAM[u, :])[0]

        # Ignore users with less artists than the crossfold validation split maximum | NF
        if NF >= len(u_aidx) or u == no_users - 1:
            continue

        # Split user's artists into train and test set for cross-fold (CV) validation
        fold = 0

        # Create folds (splits) for 10-fold CV
        kf = cross_validation.KFold(len(u_aidx), n_folds=NF)

        # For all folds
        for train_aidx, test_aidx in kf:
            # Show progress
            if VERBOSE:
                print "User: " + str(u) + ", Fold: " + str(fold) + ", Training items: " + str(
                    len(train_aidx)) + ", Test items: " + str(len(test_aidx))

            # K_RB = number of randomly selected artists to recommend
            if METHOD == "RB_Artists":  # random baseline
                dict_rec_aidx = RB_artists(np.setdiff1d(range(0, no_artists), u_aidx[train_aidx]),
                                           MIN_RECOMMENDED_ARTISTS)  # len(test_aidx))

        # Distill recommended artist indices from dictionary returned by the recommendation functions
        rec_aidx = dict_rec_aidx.keys()

        if VERBOSE:
            print "Recommended items: ", len(rec_aidx)

        ################################
        # Compute performance measures #
        ################################

        correct_aidx = np.intersect1d(u_aidx[test_aidx], rec_aidx)

        # True Positives is amount of overlap in recommended artists and test artists
        TP = len(correct_aidx)

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

        # Add precision and recall for current user and fold to aggregate variables
        avg_prec += prec / (NF * no_users)
        avg_rec += rec / (NF * no_users)

        # Output precision and recall of current fold
        if VERBOSE:
            print ("\tPrecision: %.2f, Recall:  %.2f" % (prec, rec))

        # Increase fold counter
        fold += 1

    # Output mean average precision and recall
    f1_score = 2 * ((avg_prec * avg_rec) / (avg_prec + avg_rec))

    # Output mean average precision and recall
    if VERBOSE:
        print ("\nMAP: %.2f, MAR:  %.2f, F1-Score: %.2f" % (avg_prec, avg_rec, f1_score))

    data_rb_artists = {'avg_prec': avg_prec, 'avg_rec': avg_rec, 'f1_score': f1_score}
    return data_rb_artists


# Main program, for experimentation.
if __name__ == '__main__':
    # Load metadata from provided files into lists
    artists = read_from_file(ARTISTS_FILE)
    users = read_from_file(USERS_FILE)
    # Load UAM
    UAM = np.loadtxt(UAM_FILE, delimiter='\t', dtype=np.float32)

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

            MIN_RECOMMENDED_ARTISTS = recommender_artist

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
