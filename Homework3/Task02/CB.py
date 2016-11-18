# Load required modules
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

# Parameters
TESTFILES    = "../test_data/"
TASK2_OUTPUT = "../Task02/output/"
ARTISTS_FILE = TESTFILES + "C1ku_artists_extended.csv" # artist names for UAM
USERS_FILE   = TESTFILES + "C1ku_users_extended.csv" # user names for UAM
UAM_FILE     = TESTFILES + "C1ku/C1ku_UAM.txt" # user-artist-matrix (UAM)
AAM_FILE     = TESTFILES + "AAM_lyrics_small.txt"

NF          = 10
METHOD      = "CB"
VERBOSE     = True
MAX_ARTISTS = 500
MAX_USERS   = 50
MIN_RECOMMENDED_ARTISTS = 0
K = 1


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



# Function that implements a content-based recommender. It takes as input an artist-artist-matrix (AAM) containing pair-wise similarities
# and the indices of the seed user's training artists.
# It returns a dictionary of recommended artist indices (and corresponding scores).
def recommend_CB(AAM, seed_aidx_train, items=False, K=1):
    # AAM               artist-artist-matrix of pairwise similarities
    # seed_aidx_train   indices of training artists for seed user
    # K                 number of nearest neighbors (artists) to consider for each seed artist

    # Get nearest neighbors of train set artist of seed user
    # Sort AAM column-wise for each row
    sort_idx = np.argsort(AAM[seed_aidx_train, :], axis=1)

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


    max = sorted_dict_reco_aidx[0][1]

    new_dict_recommended_artists_idx = {}

    for i in sorted_dict_reco_aidx:
        new_dict_recommended_artists_idx[i[0]] = i[1] / max

    sorted_dict_reco_aidx = np.unique(np.append(sorted_dict_reco_aidx, items))
    print type(sorted_dict_reco_aidx)

    #sorted_dict_reco_aidx = list(set(sorted_dict_reco_aidx))

    if len(sorted_dict_reco_aidx) < MIN_RECOMMENDED_ARTISTS:
        return recommend_CB(UAM, seed_aidx_train, sorted_dict_reco_aidx, K + 1)
        #reco_art_CB = reco_art_CB.items()
        #sorted_dict_reco_aidx = sorted_dict_reco_aidx + reco_art_CB
        #sorted_dict_reco_aidx = list(set(sorted_dict_reco_aidx))

    new_return = {}

    for index, key in enumerate(sorted_dict_reco_aidx, start=0):
        if index < MIN_RECOMMENDED_ARTISTS and index < len(sorted_dict_reco_aidx):
            new_return[key[0]] = key[1]


    return new_return

# Function to run an evaluation experiment.
def run():
    # Initialize variables to hold performance measures
    avg_prec = 0;  # mean precision
    avg_rec = 0;  # mean recall

    # For all users in our data (UAM)
    no_users = UAM.shape[0]
    no_artists = UAM.shape[1]
    for u in range(0, no_users):

        # Get seed user's artists listened to
        u_aidx = np.nonzero(UAM[u, :MAX_ARTISTS])[0]

        if NF >= len(u_aidx) or u == no_users - 1:
            continue

        # Split user's artists into train and test set for cross-fold (CV) validation
        fold = 0
        kf = cross_validation.KFold(len(u_aidx), n_folds=NF)  # create folds (splits) for 5-fold CV

        for train_aidx, test_aidx in kf:  # for all folds

            test_aidx_plus = len(test_aidx) * 1.15

            # print train_aidx
            # print "###"
            # print "###"
            # print test_aidx
            # Show progress
            if VERBOSE:
                print "User: " + str(u) + ", Fold: " + str(fold) + ", Training items: " + str(
                    len(train_aidx)) + ", Test items: " + str(len(test_aidx)),  # the comma at the end avoids line break
            # Call recommend function
            copy_UAM = UAM.copy()  # we need to create a copy of the UAM, otherwise modifications within recommend function will effect the variable

            dict_rec_aidx = recommend_CB(AAM, u_aidx[train_aidx], False, K)


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
        print ("K neighbors " + str(K2))
        print AAM_FILE

    data = {}
    data['f1_score'] = f1_score
    data['avg_prec'] = avg_prec
    data['avg_rec'] = avg_rec

    return data


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


# Main program, for experimentation.
if __name__ == '__main__':

    # Load metadata from provided files into lists
    artists = read_from_file(ARTISTS_FILE)
    users = read_from_file(USERS_FILE)
    # Load UAM
    UAM = np.loadtxt(UAM_FILE, delimiter='\t', dtype=np.float32)[:, :MAX_ARTISTS]
    # Load AAM
    AAM = np.loadtxt(AAM_FILE, delimiter='\t', dtype=np.float32)

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
    METHOD_two = "CB_wiki"
    runned_methods = {}
    runned_methods[METHOD_two] = []

    k_sorted = {}
    r_sorted = {}

    # data
    neighbors = [ 1, 2, 3, 5, 10, 20, 50 ]
    recommender_artists = [100, 200, 300 ]

    output_filedir = TASK2_OUTPUT + '/results/' + METHOD_two + '/'

    # ensure dir
    if not os.path.exists(output_filedir):
        os.makedirs(output_filedir)

    for neighbor in neighbors:
        k_sorted['K' + str(neighbor)] = []

        K2 = neighbor

        for recommender_artist in recommender_artists:
            r_sorted['R' + str(recommender_artist)] = []

            MIN_RECOMMENDED_ARTISTS = recommender_artist
            print MIN_RECOMMENDED_ARTISTS
            # prepare for appending
            data_to_append = {}
            data_to_append['neighbors'] = K2
            data_to_append['recommended_artists'] = MIN_RECOMMENDED_ARTISTS

            data = run()

            data_to_append.update(data)
            runned_methods[METHOD_two].append(data_to_append)

            # write into file
            content = json.dumps(data_to_append, indent=4, sort_keys=True)
            f = open(output_filedir + 'K' + str(K2) + '_recommended' + str(MIN_RECOMMENDED_ARTISTS) + '.json', 'w')

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

    # data = [
    #     ["Test"],
    #     ["K", "MAP", "MAR"],
    #     [1, 0.12, 0.12]
    # ]




    # if METHOD == "HR_SCB":
    #     print METHOD
    #     K_CB = 3  # number of nearest neighbors to consider in CB (= artists)
    #     K_CF = 3  # number of nearest neighbors to consider in CF (= users)
    #     for K_HR in range(10, 100):
    #         print (str(K_HR) + ","),
    #         run()

    # if METHOD == "CB":
    #     print METHOD
    #     run()

    # if METHOD == "CF":
    #     print METHOD
    #     for K_CF in range(1, 100):
    #         print (str(K_CF) + ","),
    #         run()
