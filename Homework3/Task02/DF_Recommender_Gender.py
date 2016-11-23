__author__ = 'Mata Mata'

###########
# IMPORTS #
###########
import csv
import time
import math
import json
import numpy as np
import operator
import scipy.spatial.distance as scidist
from sklearn import cross_validation
from run_recommender import *  # run_recommender.py


####################
# GLOBAL VARIABLES #
####################
TESTFILES    = "../test_data/"
TASK2_OUTPUT = "../Task02/output/"
ARTISTS_FILE = TESTFILES + "C1ku_artists_extended.csv"  # artist names for UAM
USERS_FILE   = TESTFILES + "C1ku_users_extended.csv"  # user names for UAM
UAM_FILE     = TESTFILES + "C1ku/C1ku_UAM.txt"  # user-artist-matrix (UAM)

NF      = 10
VERBOSE = True
METHOD  = "DF_gender"
MIN_RECOMMENDED_ARTISTS = 0

def read_users_file(filename, colnr):
    """
    Function to read the users file

    :param filename: the path of the file to load
    :param colnr: the column no to load content from
    :return: a list of data
    """

    data = []
    idx_count = 0

    # open file for reading
    with open(filename, 'r') as f:
        # create reader
        reader = csv.reader(f, delimiter='\t')
        # skip header
        headers = reader.next()
        for row in reader:
            content = row[colnr]
            item = [idx_count] + [content]
            data.append(item)
            idx_count += 1
    f.close()
    return data

def clean_list_from_empty_value(old_list, column_no, value):
    """
    Cleans a given list from all rows that contain a given value.

    :param old_list: the list to clean
    :param column_no: the column number of the old_list in which should be searched for the value
    :param value: the value to clean from
    :return: a cleaned list
    """

    cleaned_list = []

    if VERBOSE:
        print""
        print "################################"
        print "# CLEAN LIST FROM EMPTY VALUES #"
        print "################################"
        print "Length of old_list: " + str(len(old_list))

    for row in old_list:
        row_content = row[column_no]

        if row_content == value or row_content == '':
            continue
        else:
            item = [row[0]] + [row_content]
            cleaned_list.append(item)

    if VERBOSE:
        print "Length of cleaned-list: " + str(len(cleaned_list))

    return cleaned_list



def generate_gender_lists(users):
    """
    Function generates a dictionary for each gender

    :param users: xxx
    :return: xxx
    """
    users_gender_clean = clean_list_from_empty_value(users, 1, '')
    gender_m = {}
    gender_f = {}
    gender_n = {}

    # Checks if the user is male or female and adds them into the correct array
    for user in users_gender_clean:
        if user[1] == 'm':
            gender_m[user[0]] = 'm'
        elif user[1] == 'f':
            gender_f[user[0]] = 'f'
        else:
            gender_n[user[0]] = 'n'

    if VERBOSE:
        helper.log_highlight("GENDER STATISTICS")
        print "Male: " + str(len(gender_m))
        print "Female: " + str(len(gender_f))
        print "Neutral: " + str(len(gender_n))

    gender_lists = [gender_m] + [gender_f] + [gender_n]

    return gender_lists



def generate_gender_UAM(UAM, gender):
    # alle user mit gender = 'm' durchgehen
    # und diese werte 0 setzen



    return True


def recommend_gender_DF(UAM, seed_uidx, seed_aidx_train, K):
    """
    Function that implements a Demographic Filtering Recommender

    :param UAM: user-artist-matrix
    :param seed_uidx: user index of seed user
    :param seed_aidx_train: indices of training artists for seed user
    :param K: number of nearest neighbors (users) to consider for each seed users
    :param df_list: list containing all users that have defined attribute
    :return: a dictionary of recommended artists
    """
    # Get playcount vector for seed user
    pc_vec = UAM[seed_uidx, :]

    # Remove information on test artists from seed's listening vector
    aidx_nz = np.nonzero(pc_vec)[0]                             # artists with non-zero listening events
    aidx_test = np.intersect1d(aidx_nz, seed_aidx_train)        # intersection between all artist indices of user and train indices gives test artist indices
    # print aidx_test

    # Set to 0 the listening events of seed user user for testing (in UAM; pc_vec just points to UAM, is thus automatically updated)
    UAM[seed_uidx, aidx_test] = 0.0

    # Seed user needs to be normalized again
    # Perform sum-to-1 normalizationUAM[seed_uidx, :]
    user_sum = np.sum(UAM[seed_uidx, :])

    if user_sum == 0.0:
        UAM[seed_uidx, :] = UAM[seed_uidx, :]
    else:
        UAM[seed_uidx, :] = UAM[seed_uidx, :] / user_sum

    # Compute similarities as inverse cosine distance between pc_vec of user and all users via UAM (assuming that UAM is normalized)
    sim_users = np.zeros(shape=(UAM.shape[0]), dtype=np.float32)
    for u in range(0, UAM.shape[0]):

        user_gender_list = False
        gender = False

        if u in genderLists[0]:
            user_gender_list = genderLists[0]
            gender = 'm'
        elif u in genderLists[1]:
            user_gender_list = genderLists[1]
            gender = 'f'

        number = 1.0 - scidist.cosine(pc_vec, UAM[u,:])

        if math.isnan(number):
            number = 0.0

        sim_users[u] = number

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
    sum_value = np.sum(UAM[neighbor_idx][:, recommended_artists_idx], axis=0)

    # Write (artist index, score) pairs to dictionary of recommended artists
    for i in range(0, len(recommended_artists_idx)):
        dict_recommended_artists_idx[recommended_artists_idx[i]] = sum_value[i]
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

    if len(new_dict_recommended_artists_idx) < MIN_RECOMMENDED_ARTISTS:
        reco_art_CF = recommend_gender_DF(UAM, seed_uidx, seed_aidx_train, K + 1)
        new_dict_recommended_artists_idx.update(reco_art_CF)

    sorted_dict_reco_aidx = sorted(new_dict_recommended_artists_idx.items(), key=operator.itemgetter(1), reverse=True)

    new_dict_finish = {}
    for index, key in enumerate(sorted_dict_reco_aidx, start=0):
        if index < MIN_RECOMMENDED_ARTISTS and index < len(sorted_dict_reco_aidx):
            new_dict_finish[key[0]] = key[1]


    # Return dictionary of recommended artist indices (and scores)
    return new_dict_finish
# /recommend_gender_DF



def run(_K, _recommended_artists):
    """
    Function to run an evaluation experiment

    :param _K:
    :param _recommended_artists:
    :return: a dictionary of data
    """

    global MIN_RECOMMENDED_ARTISTS, UAM_MALE, UAM_FEMALE, UAM_NEUTRAL

    # Initialize variables to hold performance measures
    avg_prec = 0
    avg_rec = 0
    no_users = UAM.shape[0]
    MIN_RECOMMENDED_ARTISTS = _recommended_artists

    recommended_artists = {}

    user_with_attr_counter = 0

    for u in range(0, no_users):
        user_gender_list = False
        gender = False

        if u in genderLists[0]:
            user_gender_list = genderLists[0]
            gender = 'm'
        elif u in genderLists[1]:
            user_gender_list = genderLists[1]
            gender = 'f'
        elif u in genderLists[2]:
            user_gender_list = genderLists[2]
            gender = 'n'

        # Only perform test for seed_user who has attribute
        if user_gender_list:

            user_with_attr_counter += 1


            if gender == 'm':
               u_aidx = np.nonzero(UAM_MALE[u, :])[0]
            elif gender == 'f':
               u_aidx = np.nonzero(UAM_FEMALE[u, :])[0]
            else:
               u_aidx = np.nonzero(UAM_NEUTRAL[u, :])[0]

            if NF >= len(u_aidx) or u == no_users - 1:
                continue
            # Get seed user's artists listened to
            #u_aidx = np.nonzero(UAM[u, :])[0]

            recommended_artists[str(u)] = {}


            # Split user's artists into train and test set for cross-fold (CV) validation
            fold = 0
            kf = cross_validation.KFold(len(u_aidx), n_folds=NF)  # create folds (splits) for 5-fold CV

            for train_aidx, test_aidx in kf:  # for all folds
                if VERBOSE:
                    print "User: " + str(u) + ", Fold: " + str(fold) + ", Training items: " + str(
                        len(train_aidx)) + ", Test items: " + str(
                        len(test_aidx)),  # the comma at the end avoids line break

                copy_UAM = UAM.copy()

                if gender == 'm':
                    copy_UAM = UAM_MALE.copy()
                elif gender == 'f':
                    copy_UAM = UAM_FEMALE.copy()
                else:
                    copy_UAM = UAM_NEUTRAL.copy()

                dict_rec_aidx = recommend_gender_DF(copy_UAM, u, u_aidx[train_aidx], _K)

                if not dict_rec_aidx:
                    continue

                recommended_artists[str(u)][str(fold)] = dict_rec_aidx

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
        print ("K neighbors " + str(_K))
        print ("Recommendation: " + str(_recommended_artists))
        print "____________________________________________________________________________"

    data = {}
    data['avg_prec'] = avg_prec
    data['avg_rec'] = avg_rec
    data['f1_score'] = f1_score
    data['recommended'] = recommended_artists

    return data


# Main program, for experimentation.
if __name__ == '__main__':
    global genderLists, UAM_FILE, UAM_FEMALE, UAM_NEUTRAL

    # Load metadata from provided files into lists
    artists      = helper.read_csv(ARTISTS_FILE)
    users_gender = read_users_file(USERS_FILE, 5)
    genderLists  = generate_gender_lists(users_gender)

    if VERBOSE:
        print genderLists[0]
        print genderLists[1]
        print genderLists[2]

    if VERBOSE:
        helper.log_highlight('Loading UAM')

    UAM = np.loadtxt(UAM_FILE, delimiter='\t', dtype=np.float32)

    # copy UAM for every gender/neutrals
    UAM_MALE    = UAM.copy()
    UAM_FEMALE  = UAM.copy()
    UAM_NEUTRAL = UAM.copy()

    # generate UAM file for males
    # neutralize neutrals and females
    UAM_MALE[genderLists[1].keys()] = 0.0
    UAM_MALE[genderLists[2].keys()] = 0.0

    # generate UAM file for females
    # neutralize neutrals and males
    UAM_FEMALE[genderLists[0].keys()] = 0.0
    UAM_FEMALE[genderLists[2].keys()] = 0.0

    # generate UAM file for neutrals
    # neutralize female and male
    UAM_NEUTRAL[genderLists[0].keys()] = 0.0
    UAM_NEUTRAL[genderLists[1].keys()] = 0.0

    if VERBOSE:
        print 'Successfully loaded UAM'

    time_start = time.time()

    run_recommender(run, METHOD)  # serial

    time_end = time.time()
    elapsed_time = (time_end - time_start)

    print ""
    print "Elapsed time: " + str(elapsed_time)
