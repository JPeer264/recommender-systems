__author__ = 'beelee, mariooooo, luuuuuuuukas'

# Load required modules
import csv
import time
import operator
import numpy as np
import scipy.spatial.distance as scidist
from sklearn import cross_validation
from run_recommender import *  # run_recommender.py
from geopy.distance import great_circle

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
MAX_USERS = 50
MIN_RECOMMENDED_ARTISTS = 0
METHOD = "DF_Country_test"
# -----------------------------

LANGUAGES_COUNTRIES = [['af', 'ZA'],
                       ['am', 'ET'],
                       ['ar', 'AE'],
                       ['ar', 'BH'],
                       ['ar', 'DZ'],
                       ['ar', 'EG'],
                       ['ar', 'IQ'],
                       ['ar', 'JO'],
                       ['ar', 'KW'],
                       ['ar', 'LB'],
                       ['ar', 'LY'],
                       ['ar', 'MA'],
                       ['arn', 'CL'],
                       ['ar', 'OM'],
                       ['ar', 'QA'],
                       ['ar', 'SA'],
                       ['ar', 'SY'],
                       ['ar', 'TN'],
                       ['ar', 'YE'],
                       ['as', 'IN'],
                       ['ba', 'RU'],
                       ['be', 'BY'],
                       ['bg', 'BG'],
                       ['bn', 'BD'],
                       ['bn', 'IN'],
                       ['bo', 'CN'],
                       ['br', 'FR'],
                       ['ca', 'ES'],
                       ['co', 'FR'],
                       ['cs', 'CZ'],
                       ['cy', 'GB'],
                       ['da', 'DK'],
                       ['de', 'AT'],
                       ['de', 'CH'],
                       ['de', 'DE'],
                       ['de', 'LI'],
                       ['de', 'LU'],
                       ['dsb', 'DE'],
                       ['dv', 'MV'],
                       ['el', 'GR'],
                       ['en', '029'],
                       ['en', 'AU'],
                       ['en', 'BZ'],
                       ['en', 'CA'],
                       ['en', 'GB'],
                       ['en', 'IE'],
                       ['en', 'IN'],
                       ['en', 'JM'],
                       ['en', 'MY'],
                       ['en', 'NZ'],
                       ['en', 'PH'],
                       ['en', 'SG'],
                       ['en', 'TT'],
                       ['en', 'US'],
                       ['en', 'ZA'],
                       ['en', 'ZW'],
                       ['es', 'AR'],
                       ['es', 'BO'],
                       ['es', 'CL'],
                       ['es', 'CO'],
                       ['es', 'CR'],
                       ['es', 'DO'],
                       ['es', 'EC'],
                       ['es', 'ES'],
                       ['es', 'GT'],
                       ['es', 'HN'],
                       ['es', 'MX'],
                       ['es', 'NI'],
                       ['es', 'PA'],
                       ['es', 'PE'],
                       ['es', 'PR'],
                       ['es', 'PY'],
                       ['es', 'SV'],
                       ['es', 'US'],
                       ['es', 'UY'],
                       ['es', 'VE'],
                       ['et', 'EE'],
                       ['eu', 'ES'],
                       ['fa', 'IR'],
                       ['fi', 'FI'],
                       ['fil', 'PH'],
                       ['fo', 'FO'],
                       ['fr', 'BE'],
                       ['fr', 'CA'],
                       ['fr', 'CH'],
                       ['fr', 'FR'],
                       ['fr', 'LU'],
                       ['fr', 'MC'],
                       ['fy', 'NL'],
                       ['ga', 'IE'],
                       ['gd', 'GB'],
                       ['gl', 'ES'],
                       ['gsw', 'FR'],
                       ['gu', 'IN'],
                       ['he', 'IL'],
                       ['hi', 'IN'],
                       ['hr', 'BA'],
                       ['hr', 'HR'],
                       ['hsb', 'DE'],
                       ['hu', 'HU'],
                       ['hy', 'AM'],
                       ['id', 'ID'],
                       ['ig', 'NG'],
                       ['ii', 'CN'],
                       ['is', 'IS'],
                       ['it', 'CH'],
                       ['it', 'IT'],
                       ['ja', 'JP'],
                       ['ka', 'GE'],
                       ['kk', 'KZ'],
                       ['kl', 'GL'],
                       ['km', 'KH'],
                       ['kn', 'IN'],
                       ['kok', 'IN'],
                       ['ko', 'KR'],
                       ['ky', 'KG'],
                       ['lb', 'LU'],
                       ['lo', 'LA'],
                       ['lt', 'LT'],
                       ['lv', 'LV'],
                       ['mi', 'NZ'],
                       ['mk', 'MK'],
                       ['ml', 'IN'],
                       ['mn', 'MN'],
                       ['moh', 'CA'],
                       ['mr', 'IN'],
                       ['ms', 'BN'],
                       ['ms', 'MY'],
                       ['mt', 'MT'],
                       ['nb', 'NO'],
                       ['ne', 'NP'],
                       ['nl', 'BE'],
                       ['nl', 'NL'],
                       ['nn', 'NO'],
                       ['nso', 'ZA'],
                       ['oc', 'FR'],
                       ['or', 'IN'],
                       ['pa', 'IN'],
                       ['pl', 'PL'],
                       ['prs', 'AF'],
                       ['ps', 'AF'],
                       ['pt', 'BR'],
                       ['pt', 'PT'],
                       ['qut', 'GT'],
                       ['quz', 'BO'],
                       ['quz', 'EC'],
                       ['quz', 'PE'],
                       ['rm', 'CH'],
                       ['ro', 'RO'],
                       ['ru', 'RU'],
                       ['rw', 'RW'],
                       ['sah', 'RU'],
                       ['sa', 'IN'],
                       ['se', 'FI'],
                       ['se', 'NO'],
                       ['se', 'SE'],
                       ['si', 'LK'],
                       ['sk', 'SK'],
                       ['sl', 'SI'],
                       ['sma', 'NO'],
                       ['sma', 'SE'],
                       ['smj', 'NO'],
                       ['smj', 'SE'],
                       ['smn', 'FI'],
                       ['sms', 'FI'],
                       ['sq', 'AL'],
                       ['sv', 'FI'],
                       ['sv', 'SE'],
                       ['sw', 'KE'],
                       ['syr', 'SY'],
                       ['ta', 'IN'],
                       ['te', 'IN'],
                       ['th', 'TH'],
                       ['tk', 'TM'],
                       ['tn', 'ZA'],
                       ['tr', 'TR'],
                       ['tt', 'RU'],
                       ['ug', 'CN'],
                       ['uk', 'UA'],
                       ['ur', 'PK'],
                       ['vi', 'VN'],
                       ['wo', 'SN'],
                       ['xh', 'ZA'],
                       ['yo', 'NG'],
                       ['zh', 'CN'],
                       ['zh', 'HK'],
                       ['zh', 'MO'],
                       ['zh', 'SG'],
                       ['zh', 'TW'],
                       ['zu', 'ZA']]


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
        print""
        print "################################"
        print "# CLEAN LIST FROM EMPTY VALUES #"
        print "################################"
        print "Length of old_list: "
        print str(len(old_list))

    for row in old_list:
        row_content = row[column_no]

        if row_content == value or row_content == '':
            continue
        else:
            item = [row[0]] + [row_content]
            cleaned_list.append(item)

    if VERBOSE:
        print "Length of cleaned-list: "
        print str(len(cleaned_list))
    return cleaned_list


def check_if_list_contains_user(user_idx, list_to_check):
    """
    Function checks if given user_idx is in a given list

    :param user_idx: the user-index to look for
    :param list_to_check: the list to check
    :return: True if value is in the list, False if value is not in the list
    """

    user_has_df_attr = False

    for i in range(0, len(list_to_check)):

        # If user_idx is in list, return true and stop looking
        if user_idx == list_to_check[i][0]:
            user_has_df_attr = True
            break

    # When whole list was checked, return whether user_idx was found or not (True|False)
    return user_has_df_attr

def get_neighbor_contries(seed_uidx, nearby, distance):
    seed_lat_long = (users_long_clean[seed_uidx][1], users_lat_clean[seed_uidx][1])
    seed_country = [item for item in users_country_clean if item[0] == seed_uidx]
    seed_country = seed_country[0][1]
    countries = list(set([item[1] for item in users_country_clean]))
    country_neighbor_idx = [item for item in users_country_clean if item[1] == seed_country]

    seed_country_lang = [item for item in LANGUAGES_COUNTRIES if item[1] == seed_country]

    same_lang = []

    for country in countries:
        country_lang = [item for item in LANGUAGES_COUNTRIES if item[1] == country]
        for lang in seed_country_lang:
            for lang2 in country_lang:
                if lang[0] == lang2[0]:
                    same_lang.append(country)

    same_lang = list(set(same_lang))

    for country in same_lang:
        country_neighbor_idx = country_neighbor_idx + [item for item in users_country_clean if item[1] == country]

    country_neighbor_idx = [int(i[0]) for i in country_neighbor_idx]

    #print country_neighbor_idx

    if nearby:
        for i in range(0, len(users_country_clean)):
            if seed_uidx != i:
                user_lat_long = (users_long_clean[i][1],users_lat_clean[i][1])
                if great_circle(seed_lat_long, user_lat_long).miles < distance:
                    print distance
                    country_neighbor_idx.append(i)

    country_neighbor_idx = list(set(country_neighbor_idx))

    #print len(country_neighbor_idx)


    return country_neighbor_idx


def recommend_country_DF(UAM, seed_uidx, seed_aidx_train, K):
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

    country_neighbor_idx = get_neighbor_contries(seed_uidx, False, 0)

    # Compute similarities as inverse cosine distance between pc_vec of user and all users via UAM (assuming that UAM is normalized)
    sim_users = np.zeros(shape=(UAM.shape[0]), dtype=np.float32)
    for u in range(0, UAM.shape[0]):
        sim_users[u] = 1.0 - scidist.cosine(pc_vec, UAM[u, :])

    # Sort similarities to all others
    sort_idx = np.argsort(sim_users)  # sort in ascending order

    print sort_idx

    sort_idx = list(set(sort_idx).intersection(country_neighbor_idx))

    distance = 800
    while len(sort_idx) == 1 and K < MIN_RECOMMENDED_ARTISTS:
        sort_idx = list(set(sort_idx).intersection(get_neighbor_contries(seed_uidx, True, distance)))
        distance += 200
        if distance > 5000:
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
    artist_idx_n = np.nonzero(UAM[neighbor_idx, :])[1]  # [1] because we are only interested in non-zero elements among the artist axis

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

    if len(new_dict_recommended_artists_idx) < MIN_RECOMMENDED_ARTISTS:
        reco_art_CF = recommend_country_DF(UAM, seed_uidx, seed_aidx_train, K + 1)
        new_dict_recommended_artists_idx.update(reco_art_CF)

    sorted_dict_reco_aidx = sorted(new_dict_recommended_artists_idx.items(), key=operator.itemgetter(1), reverse=True)

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
    _K = MIN_RECOMMENDED_ARTISTS
    print _K
    print MIN_RECOMMENDED_ARTISTS

    recommended_artists = {}

    user_with_attr_counter = 0

    for u in range(0, no_users):

        print ""
        print "User: " + str(u) + ":"
        user_has_attr = check_if_list_contains_user(u, users_country_clean)
        print "User has attribute: " + str(user_has_attr)

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

                dict_rec_aidx = recommend_country_DF(copy_UAM, u, u_aidx[train_aidx], _K)

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

    print ""
    print "###########################"
    print " Users with attribute: " + str(user_with_attr_counter)
    print "###########################"

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

    # Load metadata from provided files into lists
    artists = read_artists_file(ARTISTS_FILE)

    users_country = read_users_file(USERS_FILE, 2)
    users_long = read_users_file(USERS_FILE, 3)
    users_lat = read_users_file(USERS_FILE, 4)

    global users_country_clean
    global users_long_clean
    global users_lat_clean
    users_country_clean = clean_list_from_empty_value(users_country, 1, '')
    users_long_clean = clean_list_from_empty_value(users_long, 1, '')
    users_lat_clean = clean_list_from_empty_value(users_lat, 1, '')


    if VERBOSE:
        helper.log_highlight('Loading UAM')

    UAM = np.loadtxt(UAM_FILE, delimiter='\t', dtype=np.float32)[:250, :10100]

    if VERBOSE:
        print 'Successfully loaded UAM'

    time_start = time.time()

    run_recommender(run, METHOD, [6], [10])  # serial

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
