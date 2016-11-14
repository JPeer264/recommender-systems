# Implementation of a simple evaluation framework for recommender systems algorithms.
# This script further implements different recommenders: collaborative filtering,
# content-based recommendation, random recommendation, popularity-based recommendation, and
# hybrid methods (score-based and rank-based fusion).
__author__ = 'mms'

# Load required modules
import csv
import numpy as np
from sklearn import cross_validation            # machine learning & evaluation module
import random
import scipy.spatial.distance as scidist        # import distance computation module from scipy package
from operator import itemgetter                 # for sorting dictionaries w.r.t. values


# Parameters
ROOT_DIR = "./"
UAM_FILE = ROOT_DIR + "UAM_100u.txt"                # user-artist-matrix (UAM)
#UAM_FILE = "UAM_100u_raw.txt"                # user-artist-matrix (UAM)
ARTISTS_FILE = ROOT_DIR + "UAM_100u_artists.txt"    # artist names for UAM
USERS_FILE = ROOT_DIR + "UAM_100u_users.txt"        # user names for UAM
AAM_FILE = ROOT_DIR + "AAM_100u.txt"                # artist-artist similarity matrix (AAM)
METHOD = "PB"                       # recommendation method
                                    # ["RB", "PB", "CF", "CB", "HR_RB", "HR_SCB"]

NF = 2              # number of folds to perform in cross-validation
VERBOSE = False     # verbose output?

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
    aidx_test = np.setdiff1d(aidx_nz, seed_aidx_train)          # set difference between all artist indices of user and train indices gives test artist indices
#    print aidx_test

    # Set to 0 the listening events of seed user user for testing (in UAM; pc_vec just points to UAM, is thus automatically updated)
    UAM[seed_uidx, aidx_test] = 0.0

    # Seed user needs to be normalized again
    # Perform sum-to-1 normalization
    sim_users = np.zeros(shape=(UAM.shape[0]), dtype=np.float32)
    UAM[seed_uidx,:] = UAM[seed_uidx,:] / np.sum(UAM[seed_uidx,:])

    # Compute similarities as inverse cosine distance between pc_vec of user and all users via UAM (assuming that UAM is normalized)
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
    artist_idx_n = np.nonzero(UAM[neighbor_idx, :])[1]

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
    scores = np.mean(UAM[neighbor_idx][:,recommended_artists_idx], axis=0)

    # Write (artist index, score) pairs to dictionary of recommended artists
    for i in range(0, len(recommended_artists_idx)):
        dict_recommended_artists_idx[recommended_artists_idx[i]] = scores[i]
    #########################################


    # Return dictionary of recommended artist indices (and scores)
    return dict_recommended_artists_idx


# Function that implements a content-based recommender. It takes as input an artist-artist-matrix (AAM) containing pair-wise similarities
# and the indices of the seed user's training artists.
# It returns a dictionary of recommended artist indices (and corresponding scores).
def recommend_CB(AAM, seed_aidx_train, K):
    # AAM               artist-artist-matrix of pairwise similarities
    # seed_aidx_train   indices of training artists for seed user
    # K                 number of nearest neighbors (artists) to consider for each seed artist


    # Get nearest neighbors of train set artist of seed user
    # Sort AAM column-wise for each row
    sort_idx = np.argsort(AAM[seed_aidx_train,:], axis=1)

    # Select the K closest artists to all artists the seed user listened to
    neighbor_idx = sort_idx[:,-1-K:-1]


    ##### ADDED FOR SCORE-BASED FUSION  #####
    dict_recommended_artists_idx = {}           # dictionary to hold recommended artists and corresponding scores

    # Distill corresponding similarity scores and store in sims_neighbors_idx
    sims_neighbors_idx = np.zeros(shape=(len(seed_aidx_train), K), dtype=np.float32)
    for i in range(0, neighbor_idx.shape[0]):
        sims_neighbors_idx[i] = AAM[seed_aidx_train[i], neighbor_idx[i]]

    # Aggregate the artists in neighbor_idx.
    # To this end, we compute their average similarity to the seed artists
    uniq_neighbor_idx = set(neighbor_idx.flatten())     # First, we obtain a unique set of artists neighboring the seed user's artists.
    # Now, we find the positions of each unique neighbor in neighbor_idx.
    for nidx in uniq_neighbor_idx:
        mask = np.where(neighbor_idx == nidx)
        # Apply this mask to corresponding similarities and compute average similarity
        avg_sim = np.mean(sims_neighbors_idx[mask])
        # Store artist index and corresponding aggregated similarity in dictionary of arists to recommend
        dict_recommended_artists_idx[nidx] = avg_sim
    #########################################

    # Remove all artists that are in the training set of seed user
    for aidx in seed_aidx_train:
        dict_recommended_artists_idx.pop(aidx, None)            # drop (key, value) from dictionary if key (i.e., aidx) exists; otherwise return None

    # Return dictionary of recommended artist indices (and scores)
    return dict_recommended_artists_idx


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
        dict_random_aidx[aidx] = 1.0            # for random recommendations, all scores are equal

    # Return dict of recommended artist indices as keys (and scores as values)
    return dict_random_aidx

# Function that implements a dumb random recommender. It predicts a number of artists from randomly chosen users.
# It returns a dictionary of recommended artist indices (and corresponding scores).
def recommend_RB_user(UAM, seed_aidx_train, no_items, K_users = 1):
    # UAM                   user-artist-matrix
    # artists_idx           list of artist indices to draw random sample from
    # no_items              no of items to predict
    # K_users               no of random users selected

    # Select a random sample of users
    random_uidx = random.sample(range(0,UAM.shape[0]), K_users)
    # Get artits of these
    random_aidx_nz = np.nonzero(UAM[random_uidx,:])[1]      # only interested in artists, hence [1]
    # Remove artists in training set of seed user
    random_aidx = np.setdiff1d(set(random_aidx_nz), seed_aidx_train)

    if VERBOSE:
        print str(K_users) + ' user(s) randomly chosen, ' + str(no_items) + ' recommendations requested, ' + str(len(random_aidx)) + ' found' # restart with K=' + str(K_users+1)

    # Start over with increased number of users to consider, if recommended artists smaller than requested
    if len(random_aidx) < no_items:
        K_users += 1
        return recommend_RB_user(UAM, seed_aidx_train, no_items, K_users)

    # Insert scores into dictionary
    dict_random_aidx = {}
    for aidx in random_aidx:
        dict_random_aidx[aidx] = 1.0            # for random recommendations, all scores are equal

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
            if METHOD == "RB":          # random baseline
#                dict_rec_aidx = recommend_RB(np.setdiff1d(range(0, no_artists), u_aidx[train_aidx]), K_RB) # len(test_aidx))
                N = 100
                dict_rec_aidx = recommend_RB_user(UAM, u_aidx[train_aidx], N, K_RB) # len(test_aidx))
            elif METHOD == "PB":        # popularity-based recommender
                dict_rec_aidx = recommend_PB(copy_UAM, u_aidx[train_aidx], K_PB) # len(test_aidx))
            elif METHOD == "CF":        # collaborative filtering
                dict_rec_aidx = recommend_CF(copy_UAM, u, u_aidx[train_aidx], K_CF)
            elif METHOD == "CB":        # content-based recommender
                dict_rec_aidx = recommend_CB(AAM, u_aidx[train_aidx], K_CB)
            elif METHOD == "HR_SCB":    # hybrid of CF and CB, using score-based fusion (SCB)
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
                sorted_idx = np.argsort(scores_fused)
                sorted_idx_top = sorted_idx[-K_HR:]
                # Put (artist index, score) pairs of highest scoring artists in a dictionary
                dict_rec_aidx = {}
                for i in range(0, len(sorted_idx_top)):
                    dict_rec_aidx[sorted_idx_top[i]] = scores_fused[sorted_idx_top[i]]
            elif METHOD == "HR_RB":     # hybrid of CB and PB, using rank-based fusion (RB), Borda rank aggregation
                dict_rec_aidx_CB = recommend_CB(AAM, u_aidx[train_aidx], K_CB)
                dict_rec_aidx_PB = recommend_PB(copy_UAM, u_aidx[train_aidx], K_PB)
                # Fuse scores given by CB and by PB recommenders
                # First, create matrix to hold scores per recommendation method per artist
                scores = np.zeros(shape=(2, no_artists), dtype=np.float32)
                # Add scores from CB and CF recommenders to this matrix
                for aidx in dict_rec_aidx_CB.keys():
                    scores[0, aidx] = dict_rec_aidx_CB[aidx]
                for aidx in dict_rec_aidx_PB.keys():
                    scores[1, aidx] = dict_rec_aidx_PB[aidx]
                # Convert scores to ranks
                ranks = np.zeros(shape=(2, no_artists), dtype=np.int16)         # init rank matrix
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
                sorted_idx_top = sorted_idx[-K_HR:]
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


# Main program, for experimentation.
if __name__ == '__main__':

    # Load metadata from provided files into lists
    artists = read_from_file(ARTISTS_FILE)
    users = read_from_file(USERS_FILE)
    # Load UAM
    UAM = np.loadtxt(UAM_FILE, delimiter='\t', dtype=np.float32)
    # Load AAM
    AAM = np.loadtxt(AAM_FILE, delimiter='\t', dtype=np.float32)

    # Number of neighbors / items to consider
    K_CB = 10
    K_CF = 10
    K_PB = 10

    # Run different experiments
    if False:
        METHOD = "CB"
        print METHOD
        for K_CB in range(1, 11):
            print (str(K_CB) + ","),
            run()

    if False:
        METHOD = "CF"
        print METHOD
        for K_CF in range(1, 11):
            print (str(K_CF) + ","),
            run()

    if False:
        METHOD = "RB"
        print METHOD
        for K_RB in range(1, 11):
            print (str(K_RB) + ","),
            run()

    if True:
        METHOD = "PB"
        print METHOD
        for K_PB in range(1, 11):
            print (str(K_PB) + ","),
            run()

    # For hyrbid methods, set K_CB, K_CF, and K_PB to reasonable values.
    if False:
        METHOD = "HR_SCB"
        print METHOD
        for K_HR in range(1, 11):
#            K_CB = K_CF = K_PB = K_HR
            K_CB = K_CF = K_PB = 3
            print (str(K_HR) + ","),
            run()

    if True:
        METHOD = "HR_RB"
        print METHOD
        for K_HR in range(1, 11):
#            K_CB = K_CF = K_PB = K_HR
            K_CB = K_CF = K_PB = K_HR
            print (str(K_HR) + ","),
            run()
