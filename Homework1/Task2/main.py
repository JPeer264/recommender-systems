# Implementation of a simple user-based CF recommender
__author__ = 'mms'

# Load required modules
import csv
import numpy as np


# Parameters
UAM_FILE = "UAM_5.txt"                    # user-artist-matrix (UAM)
ARTISTS_FILE = "UAM_artists_5.txt"        # artist names for UAM
USERS_FILE = "UAM_users_5.txt"            # user names for UAM


# Function to read metadata (users or artists)
def read_from_file(filename):
    data = []
    with open(filename, 'r') as f:                  # open file for reading
        reader = csv.reader(f, delimiter='\t')      # create reader
        headers = reader.next()                     # skip header
        for row in reader:
            item = row[0]
            data.append(item)
    return data


# Main program
if __name__ == '__main__':

    # Initialize variables
#    artists = []            # artists
#    users = []              # users
#    UAM = []                # user-artist-matrix

    # Load metadata from provided files into lists
    artists = read_from_file(ARTISTS_FILE)
    users = read_from_file(USERS_FILE)
#    print users
#    print artists

    # Load UAM - Konstruiert Matrix aus einem File
    UAM = np.loadtxt(UAM_FILE, delimiter='\t', dtype=np.float32)

    # For all users
    for u in range(0, UAM.shape[0]):
        # get playcount vector for current user u
        pc_vec = UAM[u,:]

        # Compute similarities as inner product between pc_vec of user and all users via UAM (assuming that UAM is already normalized)
        sim_users = np.inner(pc_vec, UAM)     # similarities between u and other users
        print sim_users

        # Sort similarities to all others
        sort_idx = np.argsort(sim_users)        # sort in ascending order
#        print sort_idx

        # Select the closest neighbor to seed user u (which is the last but one; last one is user u herself!)
        # neighbor_idx = sort_idx[k:-1][0] k definiert anzahl der nachbarn
        neighbor_idx = sort_idx[-2:-1][0]
        print "The closest user to user " + str(u) + " is " + str(neighbor_idx) + "."
        print "The closest user to user " + users[u] + " is user " + users[neighbor_idx] + "."

        # Get np.argsort(sim_users)l artist indices user u and her closest neighbor listened to, i.e., element with non-zero entries in UAM
        artist_idx_u = np.nonzero(UAM[u,:])                 # indices of artists user u listened to
        artist_idx_n = np.nonzero(UAM[neighbor_idx,:])      # indices of artists user u's neighbor listened to

        # Compute the set difference between u's neighbor and u,
        # i.e., artists listened to by the neighbor, but not by u.
        # These artists can be recommended to u.

        # np.nonzero returns a tuple of arrays, so we need to take the first element only when computing the set difference
        recommended_artists_idx = np.setdiff1d(artist_idx_n[0], artist_idx_u[0])
        # or alternatively, convert to a numpy array by ...
#        recommended_artists_idx = np.setdiff1d(np.array(artist_idx_n), np.array(artist_idx_u))


        # Output recommendations
        # artist indices
#        print "Indices of the " + str(len(recommended_artists_idx)) + " recommended artists: ", recommended_artists_idx

        # artist names
        artists_array = np.asarray(artists)     # convert list of artists to array of artists (for convenient indexing)
        print "Names of the " + str(len(recommended_artists_idx)) + " recommended artists: ", artists_array[recommended_artists_idx]
