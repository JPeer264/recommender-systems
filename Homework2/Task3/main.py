# Implementation of a simple evaluation framework for recommender systems algorithms.
# This script further implements different baseline recommenders: collaborative filtering,
# content-based recommender, random recommendation, and simple hybrid methods.
# It also implements a score-based fusion technique for hybrid recommendation.
__author__ = 'mms'

# Load required modules
import csv
import numpy as np
from sklearn import cross_validation            # machine learning & evaluation module
import random
import scipy.spatial.distance as scidist        # import distance computation module from scipy package
import helper # helper.py
import operator

# Parameters
TESTFILES    = "../testfiles/"
TASK1_OUTPUT = "../Task1/output/"
WIKI         = TASK1_OUTPUT + "wikipedia/"
MUSIXMATCH   = TASK1_OUTPUT + "musixmatch/"
ARTISTS_FILE = TASK1_OUTPUT + "artists.txt" # artist names for UAM
USERS_FILE   = TASK1_OUTPUT + "users.txt" # user names for UAM
UAM_FILE     = TESTFILES + "C1ku_UAM.txt" # user-artist-matrix (UAM)
AAM_FILE     = MUSIXMATCH + "AAM.txt" # artist-artist similarity matrix (AAM)

METHOD = "CB" # recommendation method - ["CF", "CB", "HR_SEB", "HR_SCB"]

MAX_ARTISTS = 3000
MAX_USERS   = 500

MIN_RECOMMENDED_ARTISTS = 300
MAX_RECOMMENDED_ARTISTS = MIN_RECOMMENDED_ARTISTS + 10

K = 10
K_CB = K
K_CF = K
K_HR = MIN_RECOMMENDED_ARTISTS

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


# Function that implements a CF recommender. It takes as input the UAM,
# the index of the seed user (to make predictions for) and the indices of the seed user's training artists.
# It returns a dictionary of recommended artist indices (and corresponding scores).
def recommend_CF(UAM, seed_uidx, seed_aidx_train, K):
    # UAM               user-artist-matrix
    # seed_uidx         user index of seed user
    # seed_aidx_train   indices of training artists for seed user
    # K                 number of nearest neighbors (users) to consider for each seed users

    # Get playcount vector for seed user
    pc_vec = UAM[seed_uidx, :]

    # Remove information on test artists from seed's listening vector
    aidx_nz = np.nonzero(pc_vec)[0]                             # artists with non-zero listening events
    aidx_test = np.intersect1d(aidx_nz, seed_aidx_train)        # intersection between all artist indices of user and train indices gives test artist indices
#    print aidx_test

    # Set to 0 the listening events of seed user user for testing (in UAM; pc_vec just points to UAM, is thus automatically updated)
    UAM[seed_uidx, aidx_test] = 0.0

    # Seed user needs to be normalized again
    # Perform sum-to-1 normalization
    UAM[seed_uidx, :] = UAM[seed_uidx, :] / np.sum(UAM[seed_uidx, :])

    # Compute similarities as inverse cosine distance between pc_vec of user and all users via UAM (assuming that UAM is normalized)
    sim_users = np.zeros(shape=(UAM.shape[0]), dtype=np.float32)
    for u in range(0, UAM.shape[0]):
        sim_users[u] = 1.0 - scidist.cosine(pc_vec, UAM[u,:])

    # Sort similarities to all others
    sort_idx = np.argsort(sim_users)  # sort in ascending order

    # Select the closest neighbor to seed user (which is the last but one; last one is user u herself!)
    neighbor_idx = sort_idx[-1-K:-1]

    # Get all artist indices the seed user and her closest neighbor listened to, i.e., element with non-zero entries in UAM
    artist_idx_u = seed_aidx_train                      # indices of artists in training set user
    # for k=1:
    # artist_idx_n = np.nonzero(UAM[neighbor_idx, :])     # indices of artists user u's neighbor listened to
    # for k>1:
    artist_idx_n = np.nonzero(UAM[neighbor_idx, :])[1]    # [1] because we are only interested in non-zero elements among the artist axis

    # Compute the set difference between seed user's neighbor and seed user,
    # i.e., artists listened to by the neighbor, but not by seed user.
    # These artists are recommended to seed user.
    recommended_artists_idx = np.setdiff1d(artist_idx_n, artist_idx_u)


    ##### ADDED FOR SCORE-BASED FUSION  #####
    dict_recommended_artists_idx = {}           # dictionary to hold recommended artists and corresponding scores
    # Compute artist scores. Here, just derived from max-to-1-normalized play count vector of nearest neighbor (neighbor_idx)
    # for k=1:
    # scores = UAM[neighbor_idx, recommended_artists_idx] / np.max(UAM[neighbor_idx, recommended_artists_idx])
    # for k>1:
    scores = np.mean(UAM[neighbor_idx][:, recommended_artists_idx], axis=0)
    sum = np.sum(UAM[neighbor_idx][:, recommended_artists_idx], axis=0)

    # Write (artist index, score) pairs to dictionary of recommended artists
    for i in range(0, len(recommended_artists_idx)):
        dict_recommended_artists_idx[recommended_artists_idx[i]] = sum[i]
    #########################################

    dictlist = []

    for key, value in dict_recommended_artists_idx.iteritems():
        temp = [key, value]
        dictlist.append(temp)

    sorted_dict_reco_aidx = sorted(dict_recommended_artists_idx.items(), key=operator.itemgetter(1), reverse=True)

    max_value = sorted_dict_reco_aidx[0][1]



    new_dict_recommended_artists_idx = {}

    for i in sorted_dict_reco_aidx:
        new_dict_recommended_artists_idx[i[0]] = i[1] / max_value

    if len(sorted_dict_reco_aidx) <= MIN_RECOMMENDED_ARTISTS:
        print "*"
        reco_art_RB = recommend_RB(np.setdiff1d(range(0, AAM.shape[1]), seed_aidx_train), MIN_RECOMMENDED_ARTISTS-len(sorted_dict_reco_aidx))
        print "Recommended < 10: "
        sorted_dict_reco_aidx =  sorted_dict_reco_aidx+reco_art_RB.items()

    for index, key in enumerate(sorted_dict_reco_aidx, start=0):
        if index < MAX_RECOMMENDED_ARTISTS and index < len(sorted_dict_reco_aidx):
            new_dict_recommended_artists_idx[key[0]] = key[1]


    # Return dictionary of recommended artist indices (and scores)
    return new_dict_recommended_artists_idx
# /recommend_CF


# Function that implements a content-based recommender. It takes as input an artist-artist-matrix (AAM) containing pair-wise similarities
# and the indices of the seed user's training artists.
# It returns a dictionary of recommended artist indices (and corresponding scores).
def recommend_CB(AAM, seed_aidx_train, K):
    # AAM               artist-artist-matrix of pairwise similarities
    # seed_aidx_train   indices of training artists for seed user
    # K                 number of nearest neighbors (artists) to consider for each seed artist

    # Get nearest neighbors of train set artist of seed user
    # Sort AAM column-wise for each row
    sort_idx = np.argsort(AAM[seed_aidx_train, :], axis=1)
    print '\n'
    print sort_idx
    print '\n'
    # print "AAAAMMMM###"
    # print AAM[seed_aidx_train, :MAX_ARTISTS]
    # print "###"
    # print sort_idx
    # print "###"

    # Select the K closest artists to all artists the seed user listened to
    neighbor_idx = sort_idx[:, -1 - K:-1]

    ##### ADDED FOR SCORE-BASED FUSION  #####
    dict_recommended_artists_idx = {}  # dictionary to hold recommended artists and corresponding scores

    # Distill corresponding similarity scores and store in sims_neighbors_idx
    sims_neighbors_idx = np.zeros(shape=(len(seed_aidx_train), K), dtype=np.float32)
    print "###"
    print neighbor_idx.shape[0]
    print "###"
    for i in range(0, neighbor_idx.shape[0]):
        sims_neighbors_idx[i] = AAM[seed_aidx_train[i], neighbor_idx[i]]

    # Aggregate the artists in neighbor_idx.
    # To this end, we compute their average similarity to the seed artists
    uniq_neighbor_idx = set(
        neighbor_idx.flatten())  # First, we obtain a unique set of artists neighboring the seed user's artists.

    # Now, we find the positions of each unique neighbor in neighbor_idx.
    for nidx in uniq_neighbor_idx:
        mask = np.where(neighbor_idx == nidx)

        # print mask
        # Apply this mask to corresponding similarities and compute average similarity
        avg_sim = np.mean(sims_neighbors_idx[mask])
        sum = np.sum(sims_neighbors_idx[mask]);

        # Store artist index and corresponding aggregated similarity in dictionary of artists to recommend
        dict_recommended_artists_idx[nidx] = sum
    #########################################
    # print "###"
    # print "###"
    # print dict_recommended_artists_idx
    # print "###"
    # print "###"




    for aidx in seed_aidx_train:
        dict_recommended_artists_idx.pop(aidx,
                                         None)  # drop (key, value) from dictionary if key (i.e., aidx) exists; otherwise return None

    temp = []
    dictlist = []

    for key, value in dict_recommended_artists_idx.iteritems():
        temp = [key, value]
        dictlist.append(temp)

    sorted_dict_reco_aidx = sorted(dict_recommended_artists_idx.items(), key=operator.itemgetter(1), reverse=True)

    #print sorted_dict_reco_aidx

    max = sorted_dict_reco_aidx[0][1]

    new_dict_recommended_artists_idx = {}

    for i in sorted_dict_reco_aidx:
        new_dict_recommended_artists_idx[i[0]] = i[1] / max


    if len(sorted_dict_reco_aidx) <= MIN_RECOMMENDED_ARTISTS:
        #print "*"
        reco_art_RB = recommend_RB(np.setdiff1d(range(0, AAM.shape[1]), seed_aidx_train), MIN_RECOMMENDED_ARTISTS-len(sorted_dict_reco_aidx))
        #print "Recommended < 10: "
        sorted_dict_reco_aidx =  sorted_dict_reco_aidx+reco_art_RB.items()

    for index, key in enumerate(sorted_dict_reco_aidx, start=0):
        if index < MAX_RECOMMENDED_ARTISTS and index < len(sorted_dict_reco_aidx):
            new_dict_recommended_artists_idx[key[0]] = key[1]

    print 'test'
    #print "###"
    #print "###"
    #print new_dict_recommended_artists_idx
    #print "###"
    #print "###"

    # print "###"
    # print dict_recommended_artists_idx
    # print "###"



    # print "###"
    # print recommended_artists_idx
    # print "###"

    # dict_recommended_artists_idx = dict((k, v) for k, v in dict_recommended_artists_idx.items() if v >= 0.15)

    # print '-----------'
    # print '-----------'
    # print '-----------'
    # print '-----------'
    # print '-----------'
    # print dict_recommended_artists_idx
    # print '-----------'
    # print '-----------'
    # print '-----------'
    # print '-----------'
    # print '-----------'
    # Return dictionary of recommended artist indices (and scores)
    return new_dict_recommended_artists_idx

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

        if NF >= len(u_aidx) or u == no_users -1:
            continue

        # Split user's artists into train and test set for cross-fold (CV) validation
        fold = 0
        kf   = cross_validation.KFold(len(u_aidx), n_folds=NF)  # create folds (splits) for 5-fold CV

        for train_aidx, test_aidx in kf:  # for all folds
            if VERBOSE:
                print "User: " + str(u) + ", Fold: " + str(fold) + ", Training items: " + str(len(train_aidx)) + ", Test items: " + str(len(test_aidx)),      # the comma at the end avoids line break
            # Call recommend function
            copy_UAM = UAM.copy()       # we need to create a copy of the UAM, otherwise modifications within recommend function will effect the variable

            ###############################################
            ## Combine CB and CF together so we get a HF ##
            ###############################################

            dict_rec_aidx_CB = recommend_CB(AAM, u_aidx[train_aidx], K_CB)
            dict_rec_aidx_CF = recommend_CF(copy_UAM, u, u_aidx[train_aidx], K_CF)

            # Fuse scores given by CF and by CB recommenders
            # First, create matrix to hold scores per recommendation method per artist
            scores = np.zeros(shape=(2, no_artists), dtype=np.float32)

            # Add scores from CB and CF recommenders to this matrix
            for aidx in dict_rec_aidx_CB.keys():
                scores[0, aidx] = dict_rec_aidx_CB[aidx]

            for aidx in dict_rec_aidx_CF.keys():
                scores[1, aidx] = dict_rec_aidx_CF[aidx]

            # Apply aggregation function (here, just take arithmetic mean of scores)
            scores_fused = np.mean(scores, axis=0)

            # Sort and select top K_HR artists to recommend
            sorted_idx     = np.argsort(scores_fused)
            sorted_idx_top = sorted_idx[-1-K_HR:]
            print "###"
            print len(scores_fused)
            print "###"
            # Put (artist index, score) pairs of highest scoring artists in a dictionary
            dict_rec_aidx = {}

            for i in range(0, len(sorted_idx_top)):
                dict_rec_aidx[sorted_idx_top[i]] = scores_fused[sorted_idx_top[i]]


            # Distill recommended artist indices from dictionary returned by the recommendation functions
            rec_aidx = dict_rec_aidx.keys()
            
            if VERBOSE:
                print "Recommended items: ", len(rec_aidx)

            # Compute performance measures
            correct_aidx = np.intersect1d(u_aidx[test_aidx], rec_aidx)          # correctly predicted artists

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
# /run

# Main program, for experimentation.
if __name__ == '__main__':

    # Load metadata from provided files into lists
    artists = read_from_file(ARTISTS_FILE)
    users   = read_from_file(USERS_FILE)

    if VERBOSE:
        helper.log_highlight('Read UAM file')

    UAM = np.loadtxt(UAM_FILE, delimiter='\t', dtype=np.float32)[:, :MAX_ARTISTS]

    if VERBOSE:
        print 'Successfully read UAM file\n'

    if VERBOSE:
        helper.log_highlight('Read AAM file')

    AAM = np.loadtxt(AAM_FILE, delimiter='\t', dtype=np.float32)[:MAX_ARTISTS, :MAX_ARTISTS]

    if VERBOSE:
        print 'Successfully read AAM file\n'

    run()
