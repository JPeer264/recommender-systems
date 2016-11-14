# Implementation of a simple evaluation framework for recommender systems algorithms.
# This script further implements different recommenders: collaborative filtering,
# content-based recommendation, random recommendation, popularity-based recommendation, and
# hybrid methods (score-based and rank-based fusion).
__author__ = 'mms'

# Load required modules
import csv
import json
import os
import numpy as np
from sklearn import cross_validation            # machine learning & evaluation module
import random
import scipy.spatial.distance as scidist        # import distance computation module from scipy package
from operator import itemgetter                 # for sorting dictionaries w.r.t. values


# Parameters
ROOT_DIR = "./"
TESTFILES = "../test_data/"
UAM_FILE = TESTFILES + "C1ku_UAM.txt"                # user-artist-matrix (UAM)
#UAM_FILE = "UAM_100u_raw.txt"                # user-artist-matrix (UAM)
ARTISTS_FILE = TESTFILES + "artists.txt"    # artist names for UAM
USERS_FILE = TESTFILES + "users.txt"        # user names for UAM
OUTPUT_DIR = "./output/"
METHOD = "PB"                       # recommendation method
                                    # ["RB", "PB", "CF", "CB", "HR_RB", "HR_SCB"]
K = 10
NF = 10              # number of folds to perform in cross-validation
VERBOSE = True     # verbose output?

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
        print str(K) + " artists requested, but dataset contains only " + str(no_artists) + " artists! Reducing number of requested artists to " + str(no_artists) + "."
        K = no_artists - len(seed_aidx_train)

    # get K most popular artists, according to UAM
    UAM_sum = np.sum(UAM, axis=0)                                    # sum all (normalized) listening events per artist
    recommended_artists_idx = np.argsort(UAM_sum)[-K:]                        # indices of popularity-sorted artists (K most popular artists)
    recommended_artists_scores = UAM_sum[recommended_artists_idx]             # corresponding popularity scores

    # Normalize popularity scores to range [0,1], to enable fusion with other approaches
    recommended_artists_scores = recommended_artists_scores / np.max(recommended_artists_scores)

    # Insert indices and scores into dictionary
    dict_recommended_artists_idx = {}
    for i in range(0, len(recommended_artists_idx)):
        dict_recommended_artists_idx[recommended_artists_idx[i]] = recommended_artists_scores[i]
#        print artists[recommended_artists_idx[i]] + ": " + str(recommended_artists_scores[i])

    # Return dictionary of recommended artist indices (and scores)
    return dict_recommended_artists_idx



# Function to run an evaluation experiment.
def run():
    # Initialize variables to hold performance measures
    avg_prec = 0;       # mean precision
    avg_rec = 0;        # mean recall

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


            # Run recommendation method specified in METHOD
            # NB: u_aidx[train_aidx] gives the indices of training artists

            #K_RB = 10          # for RB: number of randomly selected artists to recommend
            #K_PB = 10          # for PB: number of most frequently played artists to recommend
            #K_CB = 3           # for CB: number of nearest neighbors to consider for each artist in seed user's training set
            #K_CF = 3           # for CF: number of nearest neighbors to consider for each user
            #K_HR = 10          # for hybrid: number of artists to recommend at most
            dict_rec_aidx = recommend_PB(copy_UAM, u_aidx[train_aidx], K) # len(test_aidx))

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


# Main program, for experimentation.
if __name__ == '__main__':

    # Load metadata from provided files into lists
    artists = read_from_file(ARTISTS_FILE)
    users = read_from_file(USERS_FILE)
    # Load UAM
    UAM = np.loadtxt(UAM_FILE, delimiter='\t', dtype=np.float32)

    csv_k_sorted_header = [
        ['Sorted by K values'],
        ['']
    ]

    csv_recommended_sorted_header = [
        ['Sorted by recommended artist values'],
        ['']
    ]

    """
    format
    {
        "cb": [{
            avg_prec: Number,
            avg_rec: Number,
            neighbors: Number,
            f1_score: Number,
            recommended_artists: Number,
        }]
    }
    """
    METHOD_two = METHOD
    runned_methods = {}
    runned_methods[METHOD_two] = []

    k_sorted = {}
    r_sorted = {}

    # data
    neighbors = [ 1, 2, 3, 5, 10, 20, 50 ]
    recommender_artists = [ 10, 20, 30, 50, 100, 200, 300 ]

    output_filedir = OUTPUT_DIR + '/results/' + METHOD_two + '/'

    # ensure dir
    if not os.path.exists(output_filedir):
        os.makedirs(output_filedir)

    for neighbor in neighbors:
        k_sorted['K' + str(neighbor)] = []

        K = neighbor

        for recommender_artist in recommender_artists:
            r_sorted['R' + str(recommender_artist)] = []

            MIN_RECOMMENDED_ARTISTS = recommender_artist
            print MIN_RECOMMENDED_ARTISTS
            # prepare for appending
            data_to_append = {}
            data_to_append['neighbors'] = K
            data_to_append['recommended_artists'] = MIN_RECOMMENDED_ARTISTS

            data = run()

            data_to_append.update(data)
            runned_methods[METHOD_two].append(data_to_append)

            # write into file
            content = json.dumps(data_to_append, indent=4, sort_keys=True)
            f = open(output_filedir + 'K' + str(K) + '_recommended' + str(MIN_RECOMMENDED_ARTISTS) + '.json', 'w')

            f.write(content)
            f.close()

    content = json.dumps(data_to_append, indent=4, sort_keys=True)
    f = open(output_filedir + 'all.json', 'w')

    f.write(content)
    f.close()

    with open(output_filedir + 'all.json') as data_file:
        runned_methods = json.load(data_file)

    for result_obj in runned_methods[METHOD_two]:
        data_neighbors = [
            result_obj['neighbors'],
            result_obj['avg_prec'],
            result_obj['avg_rec'],
            result_obj['f1_score']
        ]

        data_recommended_artists = [
            result_obj['recommended_artists'],
            result_obj['avg_prec'],
            result_obj['avg_rec'],
            result_obj['f1_score']
        ]

        k_sorted['K' + str(result_obj['neighbors'])].append(data_recommended_artists)
        r_sorted['R' + str(result_obj['recommended_artists'])].append(data_neighbors)


    for key, value in r_sorted.items():
        if key[0] == 'R':
            # fill with meta info
            csv_recommended_sorted_header.append([''])
            csv_recommended_sorted_header.append([str(key) + ' recommended artists. '])

            for data in value:
                csv_recommended_sorted_header.append(data)

    for key, value in k_sorted.items():
        if key[0] == 'K':
            # fill with meta info
            csv_k_sorted_header.append([''])
            csv_k_sorted_header.append([str(key) + ' neighbors. '])

            for data in value:
                csv_k_sorted_header.append(data)

    b = open(output_filedir + 'sorted_neighbors.csv', 'w')
    a = csv.writer(b)

    a.writerows(csv_k_sorted_header)
    b.close()

    b = open(output_filedir + 'sorted_recommender.csv', 'w')
    a = csv.writer(b)

    a.writerows(csv_recommended_sorted_header)
    b.close()