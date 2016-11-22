__authors_updated_version__ = [
    'Aichbauer Lukas',
    'Leitner Bianca',
    'Stoecklmair Jan Peer',
    'Taferner Mario'
]

###########
# IMPORTS #
###########
import csv
import time
import json
import random
import os.path
import numpy as np
import scipy.spatial.distance as scidist
from sklearn import cross_validation
from FileCache import FileCache
from run_recommender import * # run_recommender.py

####################
# GLOBAL VARIABLES #
####################
TESTFILES    = "../test_data/"
UAM_FILE     = TESTFILES + "C1ku/C1ku_UAM.txt"
AAM_FILE     = TESTFILES + "AAM_lyrics_small.txt"
USERS_FILE   = TESTFILES + "C1ku_users_extended.csv"
ARTISTS_FILE = TESTFILES + "C1ku_artists_extended.csv"

NF      = 10
METHOD  = "HR_SEB"
VERBOSE = True

def run(_K, _recommended_artists):
    """
    Function to run an evaluation experiment
    """
    # Initialize variables to hold performance measures
    avg_prec = 0  # mean precision
    avg_rec = 0  # mean recall

    cb_file = FileCache("CB_Wiki", _K, _recommended_artists)
    cf_file = FileCache("CF", _K, _recommended_artists)

    # For all users in our data (UAM)
    no_users = UAM.shape[0]

    for u in range(0, no_users):

        # Get seed user's artists listened to
        # u_aidx = np.nonzero(UAM[u, :])[0]
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
            rec_aidx_CF = cf_file.read_for_hybrid(u, fold) # recommend_CF(copy_UAM, u, u_aidx[train_aidx])
            rec_aidx_CB = cb_file.read_for_hybrid(u, fold) # recommend_CB(AAM, u_aidx[train_aidx], _K)

            # @JPEER check in group if that solution is fair enough
            if len(rec_aidx_CF) == 0 or len(rec_aidx_CB) == 0:
                continue

            # Return the sorted, unique values that are in both of the input arrays.
            rec_aidx = np.intersect1d(rec_aidx_CB, rec_aidx_CF)

            if VERBOSE:
                print "Items CB: " + str(len(rec_aidx_CB))
                print "Items CF: " + str(len(rec_aidx_CF))
                print "Recommended items: " + str(len(rec_aidx))
                print "Predicted to be: " + str(_recommended_artists)

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
        print ("MAP: %.3f, MAR: %.3f, F1 Score: %.3f" % (avg_prec, avg_rec, f1_score))
        print ("K neighbors: " + str(_K))
        print ("Recommendations: " + str(_recommended_artists))

    data = {}
    data['avg_prec']    = avg_prec
    data['avg_rec']     = avg_rec
    data['f1_score']    = f1_score
    data['recommended'] = False

    return data
# /run

# Main program
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

    AAM = np.loadtxt(AAM_FILE, delimiter='\t', dtype=np.float32)

    if VERBOSE:
        print 'Successfully loaded AAM'

    time_start = time.time()

    run_recommender(run, METHOD) # serial

    time_end     = time.time()
    elapsed_time = (time_end - time_start)

    print elapsed_time
