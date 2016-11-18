###########
# IMPORTS #
###########
import os
import csv
import time
import json
import random
import helper # helper.py
import numpy as np
from sklearn import cross_validation
from run_recommender import * # run_recommender.py

####################
# GLOBAL VARIABLES #
####################
TESTFILES    = "../test_data/"
TASK2_OUTPUT = "../Task02/output"
UAM_FILE     = TESTFILES + "C1ku/C1ku_UAM.txt"
ARTISTS_FILE = TESTFILES + "C1ku_artists_extended.csv"
USERS_FILE   = TESTFILES + "C1ku_users_extended.csv"

NF      = 10
METHOD  = "RB_Artists"
VERBOSE = True
MIN_RECOMMENDED_ARTISTS = 0

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

def run(_K, _recommended_artists):
    """
    Function to run an evaluation experiment

    :return: a dictionary with MAP, MAR and F1-Score for the tested recommender
    """
    global MIN_RECOMMENDED_ARTISTS

    avg_prec   = 0
    avg_rec    = 0
    no_users   = UAM.shape[0]
    no_artists = UAM.shape[1]
    MIN_RECOMMENDED_ARTISTS = _recommended_artists

    recommended_artists = {}

    for u in range(0, no_users):
        # Get seed user's artists listened to
        u_aidx = np.nonzero(UAM[u, :])[0]

        recommended_artists[str(u)] = {}

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
            dict_rec_aidx = RB_artists(np.setdiff1d(range(0, no_artists), u_aidx[train_aidx]),
                                       MIN_RECOMMENDED_ARTISTS)  # len(test_aidx))

            recommended_artists[str(u)][str(fold)] = dict_rec_aidx

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

    data = {}
    data['avg_prec'] = avg_prec
    data['avg_rec'] = avg_rec
    data['f1_score'] = f1_score
    data['recommended'] = recommended_artists

    return data


# Main program, for experimentation.
if __name__ == '__main__':
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
