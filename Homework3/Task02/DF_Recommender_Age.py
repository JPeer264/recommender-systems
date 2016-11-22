import collections

__author__ = 'beelee'

# Load required modules
import csv
import time
import operator
import numpy as np
import scipy.spatial.distance as scidist
from sklearn import cross_validation
from run_recommender import *  # run_recommender.py
import json
import helper  # helper.py

# Parameters
TESTFILES = "../test_data/"
TASK2_OUTPUT = "../Task02/output/"
ARTISTS_FILE = TESTFILES + "C1ku_artists_extended.csv"  # artist names for UAM
USERS_FILE = TESTFILES + "C1ku_users_extended.csv"  # user names for UAM
UAM_FILE = TESTFILES + "C1ku/C1ku_UAM.txt"  # user-artist-matrix (UAM)

# Define test-parameters here:
# -----------------------------
VERBOSE = True
NF = 10
MAX_ARTISTS = 1000
MAX_USERS = 10
MIN_RECOMMENDED_ARTISTS = 0
METHOD = "DF_age"
# -----------------------------


def read_artists_file(filename):
    """
    Function to read the artists file

    :param filename: the path of the file to load
    :return: a list of data
    """

    data = []
    # open file for reading

    with open(filename, 'r') as f:
        # create reader
        reader = csv.reader(f, delimiter='\t')
        # skip header
        headers = reader.next()
        for row in reader:
            item = row[0]
            data.append(item)
    f.close()
    return data


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
        helper.log_highlight("CLEAN LIST FROM EMPTY VALUES")
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


def check_if_list_contains_user(user_idx, list_to_check):
    """
    Function checks if given user_idx is in a given list

    :param user_idx: the user-index to look for
    :param list_to_check: the list to check
    :return: True if value is in the list, False if value is not in the list
    """

    if VERBOSE:
        helper.log_highlight("CHECK IF LIST CONTAINS USER")

    user_has_df_attr = False

    for i in range(0, len(list_to_check)):

        # If user_idx is in list, return true and stop looking
        if user_idx == list_to_check[i][0]:
            if VERBOSE:
                print "YEAHHH... List contains user " + str(user_idx) + "!"
            user_has_df_attr = True
            break

    if VERBOSE:
        if not user_has_df_attr:
            print "OHHOHHH... List does NOT contain user " + str(user_idx) + "!"

    # When whole list was checked, return whether user_idx was found or not (True|False)
    return user_has_df_attr


def check_if_member_of_age_group(user_idx, json_file_age_group, age_group):
    """
    Function checks if a given user is member of a specific age group

    :param user_idx: user index to check
    :param json_file_age_group: json-file containing age_group information
    :param age_group: string-name of age group (e.g. group_teenager)
    :return: True if user is member of age-group, False if user is not a member of age group
    """
    if VERBOSE:
        helper.log_highlight("CHECK IF USER IS MEMBER OF AGE-GROUP")

    user_in_age_group = False

    with open(TESTFILES + json_file_age_group) as json_file:
        age_group_json = json.load(json_file)

    # Get users age from users_age list
    user_age = int(users_age[user_idx][1])
    if VERBOSE:
        print("User " + str(user_idx) + " has age: " + str(user_age))

    # Loop through ages in age_group - if user_age in age-group break and return True
    for key, age in age_group_json[age_group].items():
        if age == user_age:
            if VERBOSE:
                print "YEAHHH... User is member of " + age_group + "!"
            user_in_age_group = True
            break

    if VERBOSE:
        if not user_in_age_group:
            print "OHHOHHH... User is NOT a member of " + age_group + "!"

    # When whole age-group was checked, return whether user_idx was found or not (True|False)
    return user_in_age_group


def get_all_users_within_age_group(user_idx, json_file_age_group, nearby, lower, index):
    """
    Function collects all available users that are a member of the seed users age group.
    If the amount of available users within the age group is smaller then the amount of neighbours (K):
    --> alternately fills the list with additional users from age groups above and below.

    :param user_idx: the user index of the seed user
    :param json_file_age_group: the json-file containing the age-group definition
    :param nearby: TRUE if additional users from age-groups nearby are needed
    :param lower: TRUE if the age group below the users age group should be taken
    :param: index: value for which the index of the users age_group_dict should be changed
    :return: list of all users within seed users age group
    """

    global user_age_group
    user_age_group = []

    global all_users_in_age_group
    all_users_in_age_group = []

    global all_users_in_age_group_lower
    all_users_in_age_group_lower = []

    global all_users_in_age_group_higher
    all_users_in_age_group_higher = []

    if VERBOSE:
        helper.log_highlight("GET NEIGHBOURS FROM USERS AGE GROUP")

    # Load Json-File with age group information in dict
    with open(TESTFILES + json_file_age_group) as json_file:
        age_group_dict = json.load(json_file)

    # Get users age
    user_age = int(users_age[user_idx][1])

    if VERBOSE:
        print("User " + str(user_idx) + " has age: " + str(user_age))

    index_counter = -1

    # Find users age group and save index of that age group
    for ages in age_group_dict:
        index_counter += 1
        for i in ages:
            if ages[i] == user_age:
                user_age_group = ages
                index_user_age_group = index_counter

    if VERBOSE:
        print("User " + str(user_idx) + " is member of age group: " + str(user_age_group))

    # Get list from all users that are member of the users age group
    for row in users_age_clean:
        row_content = int(row[1])

        for i in user_age_group:
            if row_content == user_age_group[i]:
                user = row[0]
                all_users_in_age_group.append(user)

    if VERBOSE:
        print("Amount of users within the same age group: " + str(len(all_users_in_age_group)))

    lenght_age_group_dict = len(age_group_dict)

    if nearby:
        if lower:
            if index_user_age_group - index > 1:
                # ...get the next higher and next lower age-group from the users age-group
                age_group_lower = age_group_dict[index_user_age_group - index]

                if VERBOSE:
                    print "Age group lower: " + str(age_group_lower)

                # Get list from all users that are member of the next lower age group of
                # the seed user
                for row in users_age_clean:
                    row_content = int(row[1])

                    for i in age_group_lower:
                        if row_content == age_group_lower[i]:
                            user = row[0]
                            all_users_in_age_group.append(user)

        if not lower:
            if index + index_user_age_group < lenght_age_group_dict:
                # ...get the next higher age-group from the users age-group
                age_group_higher = age_group_dict[index_user_age_group + index]

                if VERBOSE:
                    print "Age group higher: " + str(age_group_higher)

                # Get list from all users that are member of the next higher age group of
                # the seed user
                for row in users_age_clean:
                    row_content = int(row[1])

                    for i in age_group_higher:
                        if row_content == age_group_higher[i]:
                            user = row[0]
                            all_users_in_age_group.append(user)

    all_users_in_age_group_idx = list(set(all_users_in_age_group))

    return all_users_in_age_group_idx


def recommend_age_DF(UAM, seed_uidx, seed_aidx_train, K):
    """
    Function that implements a Demographic Filtering Recommender

    :param UAM: user-artist-matrix
    :param seed_uidx: user index of seed user
    :param seed_aidx_train: indices of training artists for seed user
    :param K: number of nearest neighbors (users) to consider for each seed users
    :return: a dictionary of recommended artists
    """

    # Get playcount vector for seed user
    pc_vec = UAM[seed_uidx, :]

    # Remove information on test artists from seed's playcount vector
    # Artists with non-zero listening events
    aidx_nz = np.nonzero(pc_vec)[0]
    # Test artist indices: intersection between all artist indices of user and train indices
    aidx_test = np.intersect1d(aidx_nz, seed_aidx_train)

    # Set listening events of seed user to 0 (in UAM; pc_vec just points to UAM, is thus automatically updated)
    UAM[seed_uidx, aidx_test] = 0.0

    # Seed user needs to be normalized again
    # Perform sum-to-1 normalization
    UAM[seed_uidx, :] = UAM[seed_uidx, :] / np.sum(UAM[seed_uidx, :])


    # Get all users that are in same age-group as seed user
    age_group_users_idx = get_all_users_within_age_group(seed_uidx, "age_range.json", False, False, 0)

    # Compute similarities as inverse cosine distance between pc_vec of user and all users via UAM (assuming that UAM is normalized)
    sim_users = np.zeros(shape=(UAM.shape[0]), dtype=np.float32)
    for u in range(0, UAM.shape[0]):
        sim_users[u] = 1.0 - scidist.cosine(pc_vec, UAM[u, :])

    # Sort similarities to all others
    sort_idx = np.argsort(sim_users)  # sort in ascending order

    # Check if sort_idx contains users of age_group_users_idx
    sort_idx = list(set(sort_idx).intersection(age_group_users_idx))

    loopcounter = 0

    while len(sort_idx) == 1 and K < MIN_RECOMMENDED_ARTISTS:
        loopcounter += 1

        if loopcounter % 2 == 0:
            add_users = get_all_users_within_age_group(seed_uidx, "age_range.json", True, True, loopcounter)
        else:
            add_users = get_all_users_within_age_group(seed_uidx, "age_range.json", True, False, loopcounter)

        new_users_idx = age_group_users_idx + add_users
        sort_idx = list(set(sort_idx).intersection(new_users_idx))

        if loopcounter > 11:
            sort_idx = np.argsort(sim_users)

    if len(sort_idx) <= 1:
        return False

    # Select the closest neighbor to seed user (which is the last but one; last one is user u herself!)
    neighbor_idx = sort_idx[-1 - K:-1]


    # Get all artist indices the seed user and her closest neighbor listened to, i.e., element with non-zero entries in UAM
    artist_idx_u = seed_aidx_train  # indices of artists in training set user
    # for k=1:
    # artist_idx_n = np.nonzero(UAM[neighbor_idx, :])     # indices of artists user u's neighbor listened to
    # for k>1:
    artist_idx_n = np.nonzero(UAM[neighbor_idx, :])[
        1]  # [1] because we are only interested in non-zero elements among the artist axis

    # Compute the set difference between seed user's neighbor and seed user,
    # i.e., artists listened to by the neighbor, but not by seed user.
    # These artists are recommended to seed user.
    recommended_artists_idx = np.setdiff1d(artist_idx_n, artist_idx_u)

    ##### ADDED FOR SCORE-BASED FUSION  #####
    dict_recommended_artists_idx = {}  # dictionary to hold recommended artists and corresponding scores
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

    sorted_dict_reco_aidx = list(set(sorted_dict_reco_aidx))

    if len(new_dict_recommended_artists_idx) < MIN_RECOMMENDED_ARTISTS:
        K_users = K + 1

        if K_users > UAM.shape[0]:
            K_users = 1

        return recommend_age_DF(UAM, seed_aidx_train, seed_aidx_train, K_users)

    if len(sorted_dict_reco_aidx) < MIN_RECOMMENDED_ARTISTS:
        reco_art_CF = recommend_age_DF(UAM, seed_uidx, seed_aidx_train, K + 1)
        reco_art_CF = reco_art_CF.items()
        sorted_dict_reco_aidx = sorted_dict_reco_aidx + reco_art_CF
        sorted_dict_reco_aidx = list(set(sorted_dict_reco_aidx))

    new_dict_finish = {}
    for index, key in enumerate(sorted_dict_reco_aidx, start=0):
        if index < MIN_RECOMMENDED_ARTISTS and index < len(sorted_dict_reco_aidx):
            new_dict_finish[key[0]] = key[1]

    # Return dictionary of recommended artist indices (and scores)
    return new_dict_finish


def run(_K, _recommended_artists):
    """
    Function to run an evaluation experiment

    :param _K:
    :param _recommended_artists:
    :return: a dictionary of data
    """

    global MIN_RECOMMENDED_ARTISTS

    # Initialize variables to hold performance measures
    avg_prec = 0
    avg_rec = 0
    no_users = UAM.shape[0]
    MIN_RECOMMENDED_ARTISTS = _recommended_artists

    recommended_artists = {}

    user_with_attr_counter = 0

    for u in range(0, no_users):

        # Check if user has age attribute
        user_has_attr = check_if_list_contains_user(u, users_age_clean)

        # Only perform test for seed_user who has attribute
        if user_has_attr:

            user_with_attr_counter += 1

            # Get seed user's artists listened to
            u_aidx = np.nonzero(UAM[u, :])[0]

            recommended_artists[str(u)] = {}

            if NF >= len(u_aidx) or u == no_users - 1:
                continue

            # Split user's artists into train and test set for cross-fold (CV) validation
            fold = 0
            kf = cross_validation.KFold(len(u_aidx), n_folds=NF)  # create folds (splits) for 5-fold CV

            for train_aidx, test_aidx in kf:  # for all folds
                if VERBOSE:
                    print "User: " + str(u) + ", Fold: " + str(fold) + ", Training items: " + str(
                        len(train_aidx)) + ", Test items: " + str(
                        len(test_aidx)),  # the comma at the end avoids line break
                # Call recommend function
                copy_UAM = UAM.copy()  # we need to create a copy of the UAM, otherwise modifications within recommend function will effect the variable

                dict_rec_aidx = recommend_age_DF(copy_UAM, u, u_aidx[train_aidx], _K)

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
    # if VERBOSE:
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
    # Load metadata from provided files into lists
    artists = read_artists_file(ARTISTS_FILE)

    users_age = read_users_file(USERS_FILE, 1)
    users_country = read_users_file(USERS_FILE, 2)
    users_gender = read_users_file(USERS_FILE, 5)

    global users_age_clean
    users_age_clean = clean_list_from_empty_value(users_age, 1, '-1')

    # loopcounter = 0
    # sort_idx = get_all_users_within_age_group(0, "age_range.json", False, False, 0)
    # print "sort_idx: "
    # print sort_idx
    # print ""
    #
    #
    # for i in range(1, 15):
    #     loopcounter += 1
    #     print "i: " + str(i)
    #
    #     if loopcounter % 2 == 0:
    #         add_users = get_all_users_within_age_group(18, "age_range.json", True, True, i)
    #         print "add_users Lower: " + str(add_users)
    #     else:
    #         add_users = get_all_users_within_age_group(18, "age_range.json", True, False, i)
    #         print "add_users Higher: " + str(add_users)
    #
    #     sort_idx + add_users
    #     print "sort_idx: "
    #     print len(sort_idx)
    #
    #     print "new_idx: "
    #     print len(list(set(sort_idx)))




    if VERBOSE:
        helper.log_highlight('Loading UAM')

    # UAM = np.loadtxt(UAM_FILE, delimiter='\t', dtype=np.float32)[:MAX_USERS, :]
    UAM = np.loadtxt(UAM_FILE, delimiter='\t', dtype=np.float32)[:1000, :10100]

    if VERBOSE:
        print 'Successfully loaded UAM'

    time_start = time.time()

    run_recommender(run, METHOD)  # serial

    time_end = time.time()
    elapsed_time = (time_end - time_start)

    print ""
    print "Elapsed time: " + str(elapsed_time)


    # no_users = UAM.shape[0]
    #
    # counter = 0
    #
    # for u in range(0, no_users):
    #     print "_______________________________________________"
    #     print ""
    #     print "User: " + str(u) + ":"
    #     test = check_if_list_contains_item(u, users_age_clean)
    #     print test
    #
    #     if test:
    #         counter += 1
    # print ""
    # print "###########################"
    # print " Users with attribute: " + str(counter)
    # print "###########################"
    #
    # (check_if_member_of_age_group(0, "age_range.json", "group_late_twenties"))
