# Implementation of a simple evaluation framework for recommender systems algorithms.
# This script further implements different recommenders: collaborative filtering,
# content-based recommendation, random recommendation, popularity-based recommendation, and
# hybrid methods (score-based and rank-based fusion).
__author__ = 'mms'

###########
# IMPORTS #
###########
import os
import csv
import time
import json
import random
import numpy as np
import helper # helper.py
import scipy.spatial.distance as scidist
from sklearn import cross_validation
from operator import itemgetter
from run_recommender import * # run_recommender.py

####################
# GLOBAL VARIABLES #
####################
ROOT_DIR     = "./"
TESTFILES    = "../test_data/"
UAM_FILE     = TESTFILES + "C1ku_UAM.txt"
ARTISTS_FILE = TESTFILES + "artists.txt"
USERS_FILE   = TESTFILES + "users.txt"
OUTPUT_DIR   = "./output/"

NF      = 10
METHOD  = "PB"
VERBOSE = True
MIN_RECOMMENDED_ARTISTS = 0

# Function that implements a dumb random recommender. It predicts a number of randomly chosen items.
# It returns a dictionary of recommended artist indices (and corresponding scores).
def recommend_RB(artists_idx, no_items):
    # artists_idx           list of artist indices to draw random sample from
    # no_items              no of items to predict

    # Let's predict a number of random items that equal the number of items in the user's test set
    random_aidx = random.sample(artists_idx, no_items)

    # Insert scores into dictionary
    dict_random_aidx = {}
    for aidx in random_aidx:
        dict_random_aidx[aidx] = 1.0  # for random recommendations, all scores are equal

    # Return dict of recommended artist indices as keys (and scores as values)
    return dict_random_aidx

# Function that implements a PB recommender (popularity-based). It takes as input the UAM, computes the most popular
# artists and recommends them to the user, irrespective of their music preferences.
def recommend_PB(UAM, seed_aidx_train, K):
    # UAM               user-artist-matrix
    # seed_aidx_train   indices of training artists for seed user (to exclude corresponding recommendations)
    # K                 number of artists to recommend

    # Remove training set artists from UAM (we do not want to include these in the recommendations)
    UAM[:,seed_aidx_train] = 0.0

    # Ensure that number of available artists is not smaller than number of requested artists (excluding training set artists)
    no_artists = UAM.shape[1]
    if K > no_artists - len(seed_aidx_train):
        #print str(K) + " artists requested, but dataset contains only " + str(no_artists) + " artists! Reducing number of requested artists to " + str(no_artists) + "."
        K = no_artists - len(seed_aidx_train)

    # get K most popular artists, according to UAM
    UAM_sum = np.sum(UAM, axis=0)                                    # sum all (normalized) listening events per artist
    recommended_artists_idx = np.argsort(UAM_sum)[-K:]                        # indices of popularity-sorted artists (K most popular artists)
    recommended_artists_scores = UAM_sum[recommended_artists_idx]             # corresponding popularity scores

    # Normalize popularity scores to range [0,1], to enable fusion with other approaches
    recommended_artists_scores = recommended_artists_scores / np.max(recommended_artists_scores)

    #if len(sorted_dict_reco_aidx) <= MIN_RECOMMENDED_ARTISTS:
    #    print "*"
    #    reco_art_RB = recommend_RB(np.setdiff1d(range(0, AAM.shape[1]), seed_aidx_train),
     #                              MIN_RECOMMENDED_ARTISTS - len(sorted_dict_reco_aidx))
    #    print "Recommended < 10: "
     #   sorted_dict_reco_aidx = sorted_dict_reco_aidx + reco_art_RB.items()

    # Insert indices and scores into dictionary
    dict_recommended_artists_idx = {}
    for i in range(0, len(recommended_artists_idx)):

        dict_recommended_artists_idx[recommended_artists_idx[i]] = recommended_artists_scores[i]
#        print artists[recommended_artists_idx[i]] + ": " + str(recommended_artists_scores[i])

    # Return dictionary of recommended artist indices (and scores)
    return dict_recommended_artists_idx



# Function to run an evaluation experiment.
def run(_K, _recommended_artists):
    global MIN_RECOMMENDED_ARTISTS

    # Initialize variables to hold performance measures
    avg_prec = 0
    avg_rec  = 0
    MIN_RECOMMENDED_ARTISTS = _recommended_artists

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
        kf = cross_validation.KFold(len(u_aidx), n_folds=NF)  # create folds (splits) for 5-fold CV
        for train_aidx, test_aidx in kf:  # for all folds
            # Show progress
            if VERBOSE:
                print "User: " + str(u) + ", Fold: " + str(fold) + ", Training items: " + str(len(train_aidx)) + ", Test items: " + str(len(test_aidx)),      # the comma at the end avoids line break
            # Call recommend function
            copy_UAM = UAM.copy()       # we need to create a copy of the UAM, otherwise modifications within recommend function will effect the variable


            dict_rec_aidx = recommend_PB(copy_UAM, u_aidx[train_aidx], _recommended_artists) # len(test_aidx))

            # Distill recommended artist indices from dictionary returned by the recommendation functions
            rec_aidx = dict_rec_aidx.keys()

            if VERBOSE:
                print "Recommended items: ", len(rec_aidx)

            # Compute performance measures
            correct_aidx = np.intersect1d(u_aidx[test_aidx], rec_aidx)          # correctly predicted artists
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
            if VERBOSE:
                print ("\tPrecision: %.2f, Recall:  %.2f" % (prec, rec))

            # Increase fold counter
            fold += 1

    # Output mean average precision and recall
    if VERBOSE:
        print ("\nMAP: %.2f, MAR  %.2f" % (avg_prec, avg_rec))
        print ("%.3f, %.3f" % (avg_prec, avg_rec))

    f1_score = 2 * ((avg_prec * avg_rec) / (avg_prec + avg_rec))

    data = {}
    data['avg_prec'] = avg_prec
    data['avg_rec'] = avg_rec
    data['f1_score'] = f1_score

    return data


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

    # run_recommender(run, METHOD, [1]) # serial
    run_multithreading(run, METHOD, [1]) # parallel

    time_end = time.time()
    elapsed_time = (time_end - time_start)

    print elapsed_time
