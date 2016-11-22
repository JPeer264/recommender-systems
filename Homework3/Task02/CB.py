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
import json
import time
import random
import helper # helper.py
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
USERS_FILE   = TESTFILES + "C1ku_users_extended.csv" # user names for UAM
UAM_FILE     = TESTFILES + "C1ku/C1ku_UAM.txt" # user-artist-matrix (UAM)
AAM_FILE     = TESTFILES + "AAM_lyrics_grande.txt"

NF      = 10
METHOD  = "CB_Lyrics"
VERBOSE = False
MIN_RECOMMENDED_ARTISTS = 0

# Function that implements a content-based recommender. It takes as input an artist-artist-matrix (AAM) containing pair-wise similarities
# and the indices of the seed user's training artists.
# It returns a dictionary of recommended artist indices (and corresponding scores).
def recommend_CB(AAM, seed_aidx_train, items=[], K=1):
    # AAM               artist-artist-matrix of pairwise similarities
    # seed_aidx_train   indices of training artists for seed user
    # K                 number of nearest neighbors (artists) to consider for each seed artist

    # Get nearest neighbors of train set artist of seed user
    # Sort AAM column-wise for each row
    sort_idx = np.argsort(AAM[seed_aidx_train, :], axis=1)

    # Select the K closest artists to all artists the seed user listened to
    neighbor_idx = sort_idx[:, -1 - K:-1]

    ##### ADDED FOR SCORE-BASED FUSION  #####
    dict_recommended_artists_idx = {}  # dictionary to hold recommended artists and corresponding scores

    # Distill corresponding similarity scores and store in sims_neighbors_idx
    sims_neighbors_idx = np.zeros(shape=(len(seed_aidx_train), K), dtype=np.float32)

    for i in range(0, neighbor_idx.shape[0]):
        sims_neighbors_idx[i] = AAM[seed_aidx_train[i], neighbor_idx[i]]

    # Aggregate the artists in neighbor_idx.
    # To this end, we compute their average similarity to the seed artists
    uniq_neighbor_idx = set(neighbor_idx.flatten())  # First, we obtain a unique set of artists neighboring the seed user's artists.

    # Now, we find the positions of each unique neighbor in neighbor_idx.
    for nidx in uniq_neighbor_idx:
        mask = np.where(neighbor_idx == nidx)

        # print mask
        # Apply this mask to corresponding similarities and compute average similarity
        avg_sim = np.mean(sims_neighbors_idx[mask])
        the_sum = np.sum(sims_neighbors_idx[mask])

        # Store artist index and corresponding aggregated similarity in dictionary of artists to recommend
        dict_recommended_artists_idx[nidx] = the_sum
    #########################################

    for aidx in seed_aidx_train:
        dict_recommended_artists_idx.pop(aidx, None)  # drop (key, value) from dictionary if key (i.e., aidx) exists; otherwise return None

    temp = []
    dictlist = []

    for key, value in dict_recommended_artists_idx.iteritems():
        temp = [key, value]
        dictlist.append(temp)

    sorted_dict_reco_aidx = sorted(dict_recommended_artists_idx.items(), key=operator.itemgetter(1), reverse=True)
    sorted_dict_reco_aidx = sorted_dict_reco_aidx+items
    max = sorted_dict_reco_aidx[0][1]

    new_dict_recommended_artists_idx = {}

    for i in sorted_dict_reco_aidx:
        new_dict_recommended_artists_idx[i[0]] = i[1] / max

    sorted_dict_reco_aidx = list(set(sorted_dict_reco_aidx))

    if len(sorted_dict_reco_aidx) < MIN_RECOMMENDED_ARTISTS:
        K_users = K + 1

        if K_users > AAM.shape[0]:
            K_users = 1

        return recommend_CB(AAM, seed_aidx_train, sorted_dict_reco_aidx, K_users)

    new_return = {}

    for index, key in enumerate(sorted_dict_reco_aidx, start=0):
        if index < MIN_RECOMMENDED_ARTISTS or (index < len(sorted_dict_reco_aidx) and index < MIN_RECOMMENDED_ARTISTS):
            new_return[key[0]] = key[1]

    return new_return
# /recommend_CB

# Function to run an evaluation experiment.
def run(_K, _recommended_artists):
    global MIN_RECOMMENDED_ARTISTS

    avg_prec            = 0
    avg_rec             = 0
    no_users            = UAM.shape[0]
    no_artists          = UAM.shape[1]
    recommended_artists = {}
    MIN_RECOMMENDED_ARTISTS = _recommended_artists

    for u in range(0, no_users):

        # Get seed user's artists listened to
        u_aidx = np.nonzero(UAM[u, :])[0]

        recommended_artists[str(u)] = {}

        if NF >= len(u_aidx) or u == no_users - 1:
            continue

        # Split user's artists into train and test set for cross-fold (CV) validation
        fold = 0
        kf = cross_validation.KFold(len(u_aidx), n_folds=NF)  # create folds (splits) for 5-fold CV

        for train_aidx, test_aidx in kf:  # for all folds

            test_aidx_plus = len(test_aidx) * 1.15

            # Show progress
            if VERBOSE:
                print "User: " + str(u) + ", Fold: " + str(fold) + ", Training items: " + str(
                    len(train_aidx)) + ", Test items: " + str(len(test_aidx)),  # the comma at the end avoids line break

            # Call recommend function
            copy_UAM      = UAM.copy()
            dict_rec_aidx = recommend_CB(AAM, u_aidx[train_aidx], [], _K)
            recommended_artists[str(u)][str(fold)] = dict_rec_aidx

            # Distill recommended artist indices from dictionary returned by the recommendation functions
            rec_aidx = dict_rec_aidx.keys()

            if VERBOSE:
                print "Recommended items: ", len(rec_aidx)

            # Compute performance measures
            correct_aidx = np.intersect1d(u_aidx[test_aidx], rec_aidx)  # correctly predicted artists
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
            avg_rec  += rec / (NF * no_users)

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

    data = {}
    data['f1_score']    = f1_score
    data['avg_prec']    = avg_prec
    data['avg_rec']     = avg_rec
    data['recommended'] = recommended_artists

    return data
# /run

# Main program, for experimentation.
if __name__ == '__main__':
    artists = helper.read_csv(ARTISTS_FILE)
    users   = helper.read_csv(USERS_FILE)

    if VERBOSE:
        helper.log_highlight('Loading UAM')

    UAM = np.loadtxt(UAM_FILE, delimiter='\t', dtype=np.float32)[:500,:10100]

    if VERBOSE:
        print 'Successfully loaded UAM'

    if VERBOSE:
        helper.log_highlight('Loading AAM')

    AAM = np.loadtxt(AAM_FILE, delimiter='\t', dtype=np.float32)[:10100,:10100]

    if VERBOSE:
        print 'Successfully loaded AAM'

    time_start = time.time()

    run_recommender(run, METHOD, [50]) # serial

    time_end     = time.time()
    elapsed_time = (time_end - time_start)

    print elapsed_time
