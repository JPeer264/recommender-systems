###########
# IMPORTS #
###########
import os
import csv
import time
import json
import helper # helper.py
import random
import operator
import numpy as np
import scipy.spatial.distance as scidist
from sklearn import cross_validation
from run_recommender import * # run_recommender.py

####################
# GLOBAL VARIABLES #
####################
TESTFILES    = "../test_data/"
TASK2_OUTPUT = "../Task02/output/"
ARTISTS_FILE = TESTFILES + "C1ku_artists_extended.csv" # artist names for UAM
USERS_FILE   = TESTFILES + "C1ku_artists_extended.csv" # user names for UAM
UAM_FILE     = TESTFILES + "C1ku/C1ku_UAM.txt" # user-artist-matrix (UAM)

NF      = 10
METHOD  = "RB_user_test"
VERBOSE = False
MIN_RECOMMENDED_ARTISTS = 0


# Function that implements a dumb random recommender. It predicts a number of artists from randomly chosen users.
# It returns a dictionary of recommended artist indices (and corresponding scores).
def recommend_random_user_RB(UAM, seed_aidx_train, items=False, K_users = 1):
    global MIN_RECOMMENDED_ARTISTS
    # UAM                   user-artist-matrix
    # artists_idx           list of artist indices to draw random sample from
    # no_items              no of items to predict
    # K_users               no of random users selected

    # Select a random sample of users
    random_uidx = random.sample(range(0,UAM.shape[0]), K_users)
    # Get artits of these
    random_aidx_nz = np.nonzero(UAM[random_uidx,:])[1]   # only interested in artists, hence [1]
    # Remove artists in training set of seed user
    random_aidx = np.setdiff1d(random_aidx_nz, seed_aidx_train)

    random_aidx = np.unique(np.append(random_aidx, items))

    if VERBOSE:
        print str(K_users) + ' user(s) randomly chosen, ' + str(MIN_RECOMMENDED_ARTISTS) + ' recommendations requested, ' + str(len(random_aidx)) + ' found' # restart with K=' + str(K_users+1)

    # Insert scores into dictionary
    dict_random_aidx = {}
    for aidx in random_aidx:
        dict_random_aidx[aidx] = 1.0
        # for random recommendations, all scores are equal

    dict_random_aidx = dict_random_aidx.items()

    if len(dict_random_aidx) < MIN_RECOMMENDED_ARTISTS:
        K_users += 1

        if K_users > UAM.shape[0]:
            K_users = 1

        return recommend_random_user_RB(UAM, seed_aidx_train, random_aidx, K_users)

    new_dict_finish = {}
    for index, key in enumerate(dict_random_aidx, start=0):
        if index < MIN_RECOMMENDED_ARTISTS and index < len(dict_random_aidx):
            new_dict_finish[key[0]] = key[1]

    # Return dict of recommended artist indices as keys (and scores as values)
    return new_dict_finish

# Function to run an evaluation experiment.
def run(_K, _recommended_artists):
    global MIN_RECOMMENDED_ARTISTS

    avg_prec   = 0
    avg_rec    = 0
    no_users   = UAM.shape[0]
    no_artists = UAM.shape[1]
    MIN_RECOMMENDED_ARTISTS = _recommended_artists

    for u in range(0, no_users):
        u_aidx = np.nonzero(UAM[u, :])[0]

        if NF >= len(u_aidx) or u == no_users - 1:
            continue

        fold = 0
        kf   = cross_validation.KFold(len(u_aidx), n_folds=NF)  # create folds (splits) for 5-fold CV

        for train_aidx, test_aidx in kf:
            if VERBOSE:
                print "User: " + str(u) + ", Fold: " + str(fold) + ", Training items: " + str(len(train_aidx)) + ", Test items: " + str(len(test_aidx)),  # the comma at the end avoids line break

            # Call recommend function
            copy_UAM = UAM.copy()  # we need to create a copy of the UAM, otherwise modifications within recommend function will effect the variable

            dict_rec_aidx = recommend_random_user_RB(UAM, u_aidx[train_aidx])

            rec_aidx = dict_rec_aidx.keys()

            if VERBOSE:
                print "Recommended items: ", len(rec_aidx)

            # Compute performance measures
            correct_aidx = np.intersect1d(u_aidx[test_aidx], rec_aidx)  # correctly predicted artists

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
        print ("K neighbors " + str(_K))
        print ("Recommendation: " + str(MIN_RECOMMENDED_ARTISTS))

    data = {}
    data['avg_prec'] = avg_prec
    data['avg_rec'] = avg_rec
    data['f1_score'] = f1_score
    data['recommended'] = dict_rec_aidx

    return data
# /run

# Main program, for experimentation.
if __name__ == '__main__':
    # Load metadata from provided files into lists
    artists = helper.read_csv(ARTISTS_FILE)
    users   = helper.read_csv(USERS_FILE)

    if VERBOSE:
        helper.log_highlight('Loading UAM')

    UAM = np.loadtxt(UAM_FILE, delimiter='\t', dtype=np.float32)

    if VERBOSE:
        print 'Successfully loaded UAM'

    time_start = time.time()

    run_recommender(run, METHOD, [1]) # serial

    time_end = time.time()
    elapsed_time = (time_end - time_start)

    print elapsed_time
