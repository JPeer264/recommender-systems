# Load required modules
import csv
import numpy as np
from sklearn import cross_validation            # machine learning & evaluation module
import random
import scipy.spatial.distance as scidist        # import distance computation module from scipy package
import helper # helper.py
import operator
import os
import json

# Parameters
TESTFILES    = "../test_data/"
TASK2_OUTPUT = "../Task02/output/"
ARTISTS_FILE = TESTFILES + "C1ku_artists_extended.csv" # artist names for UAM
USERS_FILE   = TESTFILES + "C1ku_artists_extended.csv" # user names for UAM
UAM_FILE     = TESTFILES + "C1ku/C1ku_UAM.txt" # user-artist-matrix (UAM)

METHOD = "RB_user"

MAX_USERS   = 50

MIN_RECOMMENDED_ARTISTS = 6

K = 1

NF      = 10 # number of folds to perform in cross-validation
VERBOSE = True # verbose output?

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

def recommend_random_user_RB(UAM, u_idx):

    all_idx = range(0, UAM.shape[0])
    random_u_idx = random.sample(np.setdiff1d(all_idx, [u_idx]), 1)[0]

    # cannot generate the own user
    if random_u_idx == u_idx:
        recommend_random_user_RB(UAM, u_idx)

    random_u_aidx = np.nonzero(UAM[random_u_idx, :])[0]

    dict_random_aidx = {}
    for aidx in random_u_aidx:
        dict_random_aidx[aidx] = 1.0

    new_dict_recommended_artists_idx = {}

    sorted_dict_reco_aidx = sorted(dict_random_aidx.items(), key=operator.itemgetter(1), reverse=True)

    if len(dict_random_aidx) <= MIN_RECOMMENDED_ARTISTS:
        print "*"
        reco_art_RB = recommend_random_user_RB(UAM,u_idx)
        print "Recommended < 10: "
        sorted_dict_reco_aidx = sorted_dict_reco_aidx + reco_art_RB.items()

    for index, key in enumerate(sorted_dict_reco_aidx, start=0):
        if index < MIN_RECOMMENDED_ARTISTS and index < len(sorted_dict_reco_aidx):
            new_dict_recommended_artists_idx[key[0]] = key[1]

    return new_dict_recommended_artists_idx

# Function to run an evaluation experiment.
def run():
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
        kf = cross_validation.KFold(len(u_aidx), n_folds=NF)  # create folds (splits) for 5-fold CV

        for train_aidx, test_aidx in kf:  # for all folds
            if VERBOSE:
                print "User: " + str(u) + ", Fold: " + str(fold) + ", Training items: " + str(
                    len(train_aidx)) + ", Test items: " + str(len(test_aidx)),  # the comma at the end avoids line break
            # Call recommend function
            copy_UAM = UAM.copy()  # we need to create a copy of the UAM, otherwise modifications within recommend function will effect the variable

            dict_rec_aidx = recommend_random_user_RB(copy_UAM, u)


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
    print ("Recommendation: " + str(MIN_RECOMMENDED_ARTISTS))

    data = {}
    data['avg_prec'] = avg_prec
    data['avg_rec'] = avg_rec
    data['f1_score'] = f1_score

    return data
# /run

# Main program, for experimentation.
if __name__ == '__main__':

    # Load metadata from provided files into lists
    artists = read_from_file(ARTISTS_FILE)
    users   = read_from_file(USERS_FILE)

    if VERBOSE:
        helper.log_highlight('Read UAM file')

    UAM = np.loadtxt(UAM_FILE, delimiter='\t', dtype=np.float32)

    if VERBOSE:
        print 'Successfully read UAM file\n'

    runned_methods = {}
    runned_methods[METHOD] = []

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

        K2 = neighbor

        for recommender_artist in recommender_artists:
            k_sorted['R' + str(recommender_artist)] = []

            MIN_RECOMMENDED_ARTISTS = recommender_artist

            # prepare for appending
            data_to_append = {}
            data_to_append['neighbors'] = K2
            data_to_append['recommended_artists'] = MIN_RECOMMENDED_ARTISTS

            data = run()

            data_to_append.update(data)
            runned_methods[METHOD].append(data_to_append)

            # write into file
            content = json.dumps(data_to_append, indent=4, sort_keys=True)
            f = open(output_filedir + 'K' + str(K2) + '_recommended' + str(MIN_RECOMMENDED_ARTISTS) + '.json', 'w')

            f.write(content)
            f.close()

    content = json.dumps(data_to_append, indent=4, sort_keys=True)
    f = open(output_filedir + 'all.json', 'w')

    f.write(content)
    f.close()
