# Load required modules
import csv
import numpy as np
from sklearn import cross_validation  # machine learning & evaluation module
import random
import traceback

# Parameters
UAM_FILE = "./data/C1ku_UAM.txt"  # user-artist-matrix (UAM)
ARTISTS_FILE = "./data/C1ku_idx_artists.txt"  # artist names for UAM
USERS_FILE = "./data/C1ku_idx_users.txt"  # user names for UAM

NF = 10  # number of folds to perform in cross-validation


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


# Function that implements a CF recommender. It takes as input the UAM, metadata (artists and users),
# the index of the seed user (to make predictions for) and the indices of the seed user's training artists.
# It returns a list of recommended artist indices
def recommend_CF(UAM, seed_uidx, seed_aidx_train):
    # UAM               user-artist-matrix
    # seed_uidx         user index of seed user
    # seed_aidx_train   indices in UAM of training artists for seed user

    # Get playcount vector for seed user
    pc_vec = UAM[seed_uidx, :]

    # Remove information on test artists from seed's listening vector
    aidx_nz = np.nonzero(pc_vec)[0]  # artists with non-zero listening events
    aidx_test = np.setdiff1d(aidx_nz,
                             seed_aidx_train)  # compute set difference between all artist indices of user and train indices gives test artist indices
    #    print aidx_test

    # Set to 0 the listening events of seed user for testing (in UAM; pc_vec just points to UAM, is thus automatically updated)
    UAM[seed_uidx, aidx_test] = 0.0

    # Seed user needs to be normalized again
    # Perform sum-to-1 normalization
    UAM[seed_uidx, :] = UAM[seed_uidx, :] / np.sum(UAM[seed_uidx, :])

    # Compute similarities as inner product between pc_vec of user and all users via UAM (assuming that UAM is normalized)
    sim_users = np.inner(pc_vec, UAM)  # similarities between u and other users

    # Alternatively, compute cosine similarities as inverse cosine distance between pc_vec of user and all users via UAM (assuming that UAM is normalized)
    #    sim_users = np.zeros(shape=(UAM.shape[0]), dtype=np.float32)
    #    for u in range(0, UAM.shape[0]):
    #        sim_users[u] = 1.0 - scidist.cosine(pc_vec, UAM[u,:])

    # Sort similarities to all others
    sort_idx = np.argsort(sim_users)  # sort in ascending order

    # Select the closest neighbor to seed user (which is the last but one; last one is user u herself!)
    neighbor_idx = sort_idx[-2:-1][0]
    #    print "The closest user to user " + str(seed_uidx) + " is " + str(neighbor_idx) + "."
    #    print "The closest user to user " + users[seed_uidx] + " is user " + users[neighbor_idx] + "."

    # Get all artist indices the seed user and her closest neighbor listened to, i.e., element with non-zero entries in UAM
    artist_idx_u = seed_aidx_train  # indices of artists in training set user
    artist_idx_n = np.nonzero(UAM[neighbor_idx, :])  # indices of artists user u's neighbor listened to

    # Compute the set difference between seed user's neighbor and seed user,
    # i.e., artists listened to by the neighbor, but not by seed user.
    # These artists are recommended to seed user.

    # np.nonzero returns a tuple of arrays, so we need to take the first element only when computing the set difference
    recommended_artists_idx = np.setdiff1d(artist_idx_n[0], artist_idx_u)
    # or alternatively, convert to a numpy array by ...
    # artist_idx_np.setdiff1d(np.array(artist_idx_n), np.array(artist_idx_u))

    # Return list of recommended artist indices
    return recommended_artists_idx


# Function that implements dumb random recommender. It predicts a number of randomly chosen items.
# It returns a list of recommended artist indices.
def recommend_RB(artists_idx, no_items):
    # artists_idx           list of artist indices to draw random sample from
    # no_items              no of items to predict

    # Let's predict a number of random items that equal the number of items in the user's test set
    random_aidx = random.sample(artists_idx, no_items)

    # Return list of recommended artist indices
    return random_aidx


# Evaluate F1-Score
def f1_measure(precision, recall):
    return 2 * ((precision * recall) / (precision + recall))


# Main program
if __name__ == '__main__':
    # Initialize variables to hold performance measures
    avg_prec = 0;  # mean precision
    avg_rec = 0;  # mean recall

    # Load metadata from provided files into lists
    artists = read_from_file(ARTISTS_FILE)
    users = read_from_file(USERS_FILE)

    # Load UAM
    UAM = np.loadtxt(UAM_FILE, delimiter='\t', dtype=np.float32)  # For all users in our data (UAM)

    # For all users in our data (UAM)
    no_users = UAM.shape[0]
    AMOUNT_OF_USER = [1,5,10,20,50,100]

    for idx in AMOUNT_OF_USER:
        for u in range(0, idx):
            try:

                # Get seed user's artists listened to
                u_aidx = np.nonzero(UAM[u, :])[0]

                # Split user's artists into train and test set for cross-fold (CV) validation
                fold = 0
                kf = cross_validation.KFold(len(u_aidx), n_folds=NF)  # create folds (splits) for 5-fold CV
                for train_aidx, test_aidx in kf:  # for all folds
                    # Show progress
                    # print "User: " + str(u) + ", Fold: " + str(fold) + ", Training items: " + str(
                        # len(train_aidx)) + ", Test items: " + str(len(test_aidx)),  # the comma at the end avoids line break
                    # Call recommend function
                    copy_UAM = UAM.copy()  # we need to create a copy of the UAM, otherwise modifications within recommend function will effect the variable
                    rec_aidx = recommend_CF(copy_UAM, u, u_aidx[train_aidx])
                    print rec_aidx
                    #            print "Recommended items: ", len(rec_aidx)

                    # For random recommendation, exclude items that the user already knows, i.e. the ones in the training set
                    #            all_aidx = range(0, UAM.shape[1])
                    #            rec_aidx = recommend_RB(np.setdiff1d(all_aidx, u_aidx[train_aidx]), len(test_aidx))       # select the number of recommended items as the number of items in the test set

                    # print "Recommended items: ", len(rec_aidx)

                    # Compute performance measures
                    correct_aidx = np.intersect1d(u_aidx[test_aidx], rec_aidx)  # correctly predicted artists
                    #            print 'Recommended artist-ids: ', rec_aidx
                    #            print 'True artist-ids: ', u_aidx[test_aidx]

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
                    # print ("\tPrecision: %.2f, Recall:  %.2f" % (prec, rec))

                    # Increase fold counter
                    fold += 1

            except ValueError:
                print ""
                print "USER SKIPPED!"
                print ""
                continue

            except Exception:
                print ""
                print "ERROR:"
                print(traceback.format_exc())

    # Output mean average precision and recall
    print ""
    print ("\nMAP: %.2f, MAR: %.2f" % (avg_prec, avg_rec))

    print ""
    print f1_measure(avg_prec, avg_rec)
