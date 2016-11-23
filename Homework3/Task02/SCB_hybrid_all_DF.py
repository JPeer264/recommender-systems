# Implementation of a simple evaluation framework for recommender systems algorithms.
# This script further implements different baseline recommenders: collaborative filtering,
# content-based recommender, random recommendation, and a simple hybrid method using set-based fusion.
__author__ = 'mms'
__authors_updated_version__ = [
    'Rudolfson',
    'beelee',
    'jpeer',
    'Mata Mata'
]

###########
# IMPORTS #
###########
import os
import csv
import time
import json
import random
import helper # helper.py
import operator
import numpy as np
import scipy.spatial.distance as scidist        # import distance computation module from scipy package
from sklearn import cross_validation            # machine learning & evaluation module
from FileCache import FileCache
from run_recommender import * # run_recommender.py

####################
# GLOBAL VARIABLES #
####################
TESTFILES    = "../test_data/"
TASK2_OUTPUT = "../Task02/output"
UAM_FILE     = TESTFILES + "C1ku/C1ku_UAM.txt"
AAM_FILE     = TESTFILES + "AAM.txt"
USERS_FILE   = TESTFILES + "C1ku_users_extended.csv"
ARTISTS_FILE = TESTFILES + "C1ku_artists_extended.csv"

NF      = 10
METHOD  = "HR_SCB_all_DF"
VERBOSE = True

# Function to run an evaluation experiment.
def run(_K, _recommended_artists):
    # Initialize variables to hold performance measures
    avg_prec = 0  # mean precision
    avg_rec = 0 # mean recall

    df_a_file = FileCache("DF_age", _K, _recommended_artists)
    df_c_file = FileCache("DF_country", _K, _recommended_artists)
    df_g_file = FileCache("DF_gender", _K, _recommended_artists)

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
            if VERBOSE:
                print "User: " + str(u) + ", Fold: " + str(fold) + ", Training items: " + str(
                    len(train_aidx)) + ", Test items: " + str(len(test_aidx)),  # the comma at the end avoids line break
            # Call recommend function
            copy_UAM = UAM.copy()  # we need to create a copy of the UAM, otherwise modifications within recommend function will effect the variable

            ###############################################
            ## Combine CB and CF together so we get a HF ##
            ###############################################

            dict_rec_aidx_DF_A = df_a_file.read_for_hybrid(u, fold)
            dict_rec_aidx_DF_C = df_c_file.read_for_hybrid(u, fold)
            dict_rec_aidx_DF_G = df_g_file.read_for_hybrid(u, fold)

            # @JPEER check in group if that solution is fair enough
            if len(dict_rec_aidx_DF_A) == 0 or len(dict_rec_aidx_DF_C) == 0 or len(dict_rec_aidx_DF_G) == 0:
                continue

            # Fuse scores given by CF and by CB recommenders
            # First, create matrix to hold scores per recommendation method per artist
            scores = np.zeros(shape=(3, no_artists), dtype=np.float32)

            # Add scores from CB and CF recommenders to this matrix
            for aidx in dict_rec_aidx_DF_A.keys():
                scores[0, aidx] = dict_rec_aidx_DF_A[aidx]

            for aidx in dict_rec_aidx_DF_C.keys():
                scores[1, aidx] = dict_rec_aidx_DF_C[aidx]

            for aidx in dict_rec_aidx_DF_G.keys():
                scores[2, aidx] = dict_rec_aidx_DF_G[aidx]

            # Apply aggregation function (here, just take arithmetic mean of scores)
            scores_fused = np.mean(scores, axis=0)

            # Sort and select top K_HR artists to recommend
            sorted_idx = np.argsort(scores_fused)
            sorted_idx_top = sorted_idx[- _recommended_artists:]

            # Put (artist index, score) pairs of highest scoring artists in a dictionary
            dict_rec_aidx = {}

            for i in range(0, len(sorted_idx_top)):
                dict_rec_aidx[sorted_idx_top[i]] = scores_fused[sorted_idx_top[i]]

            # Distill recommended artist indices from dictionary returned by the recommendation functions
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
        print ("K neighbors " + str(K))
        print ("Recommendation: " + str(_recommended_artists))

    data = {}
    data['f1_score']    = f1_score
    data['avg_prec']    = avg_prec
    data['avg_rec']     = avg_rec
    data['recommended'] = False

    return data
# /run

# Main program, for experimentation.
if __name__ == '__main__':
    artists = helper.read_csv(ARTISTS_FILE)
    users   = helper.read_csv(USERS_FILE)

    if VERBOSE:
        helper.log_highlight('Loading UAM')

    UAM = np.loadtxt(UAM_FILE, delimiter='\t', dtype=np.float32)

    if VERBOSE:
        print 'Successfully loaded UAM'

    if VERBOSE:
        helper.log_highlight('Loading AAM')

    # AAM = np.loadtxt(AAM_FILE, delimiter='\t', dtype=np.float32)

    if VERBOSE:
        print 'Successfully loaded AAM'

    time_start = time.time()

    run_recommender(run, METHOD) # serial

    time_end     = time.time()
    elapsed_time = (time_end - time_start)

    print elapsed_time
