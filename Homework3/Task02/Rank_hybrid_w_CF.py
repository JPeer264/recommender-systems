# Implementation of a simple evaluation framework for recommender systems algorithms.
# This script further implements different baseline recommenders: collaborative filtering,
# content-based recommender, random recommendation, and a simple hybrid method using set-based fusion.
__author__ = 'mms'
__authors_updated_version__ = [
    'Aichbauer Lukas',
    'Leitner Bianca',
    'Stoecklmair Jan Peer',
    'Taferner Mario'
]

###########
# IMPORTS #
###########
import os
import csv
import time
import json
import numpy as np
import random
import helper # helper.py
import operator
import scipy.spatial.distance as scidist
from sklearn import cross_validation
from FileCache import FileCache
from run_recommender import * # run_recommender.py

####################
# GLOBAL VARIABLES #
####################
TESTFILES    = "../test_data/"
TASK2_OUTPUT = "../Task02/output"
UAM_FILE     = TESTFILES + "C1ku/C1ku_UAM.txt"
AAM_FILE     = TESTFILES + "AAM_lyrics_small.txt"
ARTISTS_FILE = TESTFILES + "C1ku_artists_extended.csv"
USERS_FILE   = TESTFILES + "C1ku_users_extended.csv"

NF          = 10
METHOD      = "HR_RB_w_CF"
VERBOSE     = True
MAX_ARTISTS = 3000

# Function to run an evaluation experiment.
def run(_K, _recommended_artists):
    avg_prec   = 0
    avg_rec    = 0
    no_users   = UAM.shape[0]
    no_artists = UAM.shape[1]

    cf_file = FileCache("CF", _K, _recommended_artists)
    cb_file = FileCache("CB_Wiki", 1, _recommended_artists)
    pb_file = FileCache("PB", 1, _recommended_artists)

    recommended_artists = {}

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

            dict_rec_aidx_CB = cb_file.read_for_hybrid(u, fold) #recommend_CB(AAM, u_aidx[train_aidx], _K)
            dict_rec_aidx_PB = pb_file.read_for_hybrid(u, fold) #recommend_PB(copy_UAM, u_aidx[train_aidx], _recommended_artists)
            dict_rec_aidx_CF = cf_file.read_for_hybrid(u, fold) #recommend_PB(copy_UAM, u_aidx[train_aidx], _recommended_artists)

            # @JPEER check in group if that solution is fair enough
            if len(dict_rec_aidx_CB) == 0 or len(dict_rec_aidx_PB) == 0 or len(dict_rec_aidx_CF) == 0:
                continue

            # Fuse scores given by CB and by PB recommenders
            # First, create matrix to hold scores per recommendation method per artist
            scores = np.zeros(shape=(3, no_artists), dtype=np.float32)

            # Add scores from CB and CF recommenders to this matrix
            for aidx in dict_rec_aidx_CB.keys():
                scores[0, aidx] = dict_rec_aidx_CB[aidx]

            for aidx in dict_rec_aidx_PB.keys():
                scores[1, aidx] = dict_rec_aidx_PB[aidx]

            for aidx in dict_rec_aidx_CF.keys():
                scores[2, aidx] = dict_rec_aidx_CF[aidx]

            # Convert scores to ranks
            ranks = np.zeros(shape=(3, no_artists), dtype=np.int16)         # init rank matrix

            for m in range(0, scores.shape[0]):                             # for all methods to fuse
                aidx_nz = np.nonzero(scores[m])[0]                          # identify artists with positive scores
                scores_sorted_idx = np.argsort(scores[m,aidx_nz])           # sort artists with positive scores according to their score
                # Insert votes (i.e., inverse ranks) for each artist and current method

                for a in range(0, len(scores_sorted_idx)):
                    ranks[m, aidx_nz[scores_sorted_idx[a]]] = a + 1

            # Sum ranks over different approaches
            ranks_fused = np.sum(ranks, axis=0)
            # Sort and select top K_HR artists to recommend
            sorted_idx = np.argsort(ranks_fused)
            sorted_idx_top = sorted_idx[-_recommended_artists:]
            # Put (artist index, score) pairs of highest scoring artists in a dictionary
            dict_rec_aidx = {}

            for i in range(0, len(sorted_idx_top)):
                dict_rec_aidx[sorted_idx_top[i]] = ranks_fused[sorted_idx_top[i]]

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
    data['avg_prec']    = avg_prec
    data['avg_rec']     = avg_rec
    data['f1_score']    = f1_score
    data['recommended'] = False

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

    if VERBOSE:
        helper.log_highlight('Loading AAM')

    #AAM = np.loadtxt(AAM_FILE, delimiter='\t', dtype=np.float32)

    if VERBOSE:
        print 'Successfully loaded AAM'

    time_start = time.time()

    run_recommender(run, METHOD)

    time_end     = time.time()
    elapsed_time = (time_end - time_start)

    print elapsed_time
