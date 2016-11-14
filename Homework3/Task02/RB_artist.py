import numpy as np
import random
import csv
# Machine learning & evaluation module
from sklearn import cross_validation

# User-Artist-Matrix (UAM)
UAM_FILE = "/Users/Bee/Documents/__FH/FH Salzburg/02_Recommender_Systems/02_Homework/recommender-systems/Homework3/test_data/C1ku/C1ku_UAM.txt"
# Artist names for UAM
ARTISTS_FILE = "/Users/Bee/Documents/__FH/FH Salzburg/02_Recommender_Systems/02_Homework/recommender-systems/Homework3/test_data/C1ku_artists_extended.csv"
# User names for UAM
USERS_FILE = "/Users/Bee/Documents/__FH/FH Salzburg/02_Recommender_Systems/02_Homework/recommender-systems/Homework3/test_data/C1ku_users_extended.csv"
# Recommendation method
METHOD = "RB"

MAX_USER = 50

# Define number of folds to perform in cross-validation
NO_FOLDS = 10

K = 1

# Define number of neighboures that should be recommended
NO_REC_ARTISTS = 100

# Verbose output?
VERBOSE = True


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


# Function that implements a dumb random recommender. It predicts a number of randomly chosen items.
# It returns a dictionary of recommended artist indices (and corresponding scores).
def RB_artists(artists_idx, no_items):
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


def run():
    """
    Function to run an evaluation experiment
    """

    # Initialize variables to hold performance measures
    avg_prec = 0;  # mean precision
    avg_rec = 0;  # mean recall

    # For all users in our data (UAM)
    #no_users = UAM.shape[0]
    no_users = MAX_USER+2
    no_artists = UAM.shape[1]

    for u in range(0, no_users):

        # Get seed user's artists listened to
        u_aidx = np.nonzero(UAM[u, :])[0]

        # ignore users with less artists than the crossfold validation split maximum | NF
        if NO_FOLDS >= len(u_aidx) or u == no_users - 1:
            continue

        # Split user's artists into train and test set for cross-fold (CV) validation
        fold = 0

        # Create folds (splits) for 5-fold CV
        kf = cross_validation.KFold(len(u_aidx), n_folds=NO_FOLDS)

        # for all folds
        for train_aidx, test_aidx in kf:
            # Show progress
            if VERBOSE:
                print "User: " + str(u) + ", Fold: " + str(fold) + ", Training items: " + str(
                    len(train_aidx)) + ", Test items: " + str(len(test_aidx))

            # K_RB = number of randomly selected artists to recommend
            if METHOD == "RB":  # random baseline
                dict_rec_aidx = RB_artists(np.setdiff1d(range(0, no_artists), u_aidx[train_aidx]),
                                           NO_REC_ARTISTS)  # len(test_aidx))

        # Distill recommended artist indices from dictionary returned by the recommendation functions
        rec_aidx = dict_rec_aidx.keys()

        if VERBOSE:
            print "Recommended items: ", len(rec_aidx)

        ################################
        # Compute performance measures #
        ################################

        correct_aidx = np.intersect1d(u_aidx[test_aidx], rec_aidx)

        # True Positives is amount of overlap in recommended artists and test artists
        TP = len(correct_aidx)

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
        avg_prec += prec / (NO_FOLDS * no_users)
        avg_rec += rec / (NO_FOLDS * no_users)

        # Output precision and recall of current fold
        if VERBOSE:
            print ("\tPrecision: %.2f, Recall:  %.2f" % (prec, rec))

        # Increase fold counter
        fold += 1

    # Output mean average precision and recall
    f1_score = 2 * ((avg_prec * avg_rec) / (avg_prec + avg_rec))

    # Output mean average precision and recall
    if VERBOSE:
        print ("\nMAP: %.2f, MAR:  %.2f, F1-Score: %.2f" % (avg_prec, avg_rec, f1_score))

    data = {'avg_prec': avg_prec, 'avg_rec': avg_rec, 'f1_score': f1_score}
    return data


if __name__ == '__main__':
    # Load metadata from provided files into lists
    artists = read_from_file(ARTISTS_FILE)
    users = read_from_file(USERS_FILE)
    # Load UAM
    UAM = np.loadtxt(UAM_FILE, delimiter='\t', dtype=np.float32)

    if True:
        METHOD = "RB"
        print METHOD
        run()
