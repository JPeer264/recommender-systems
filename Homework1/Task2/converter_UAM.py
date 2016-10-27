# Converts crawled listening events data from Last.fm to user-artist-matrix
__author__ = 'mms'

# Load required modules
import csv
import numpy as np

# Parameters
OUTPUT_DIR = './output'
LE_FILE = "../Task1/output/user_info/listening_history.txt"
UAM_FILE = OUTPUT_DIR + "/UAM.txt"               # user-artist-matrix (UAM)
ARTISTS_FILE = OUTPUT_DIR + "/UAM_artists_5.txt" # artist names for UAM
USERS_FILE = OUTPUT_DIR + "/UAM_users_5.txt"     # user names for UAM


# Main program
if __name__ == '__main__':

    artists = {}            # dictionary, (mis)used as ordered list of artists without duplicates
    users = {}              # dictionary, (mis)used as ordered list of users without duplicates
    listening_events = {}   # dictionary to store assignments between user and artist

    # Read listening events from provided file
    with open(LE_FILE, 'r') as f:
        reader = csv.reader(f, delimiter='\t')      # create reader
        headers = reader.next()                     # skip header
        for row in reader:
            user = row[0]
            artist = row[2]
            track = row[4]
            time = row[5]

            # create ordered set (list) of unique elements (for artists / tracks)
            artists[artist] = None
            users[user] = None

            # initialize listening event counter, access by tuple (user, artist) in dictionary
            listening_events[(user, artist)] = 0


    # Read listening events from provided file (to fill user-artist matrix)
    with open(LE_FILE, 'r') as f:
        reader = csv.reader(f, delimiter='\t')      # create reader
        headers = reader.next()                     # skip header
        for row in reader:
            user = row[0]
            artist = row[2]
            track = row[4]
            time = row[5]
            # increase listening counter for (user, artist) pair/tuple
            listening_events[(user, artist)] += 1


    # Assign a unique index to all artists and users in dictionary (we need these to create the UAM)
    # Artists
    counter = 0
    for artist in artists.keys():
        artists[artist] = counter
        counter += 1
    # Users
    counter = 0
    for user in users.keys():
        users[user] = counter
        counter += 1

    # Now we use numpy to create the UAM
    UAM = np.zeros(shape=(len(users.keys()), len(artists.keys())), dtype=np.float32)
    # dtype='uint32')        # first, create an empty matrix
    # iterate through all (user, artist) tuples in listening_events
    for u in users.keys():
        for a in artists.keys():
            try:
                # get correct index for user u and artist a
                idx_u = users[u]
                idx_a = artists.get(a)

                # insert number of listening events of user u to artist a in UAM
                UAM[idx_u, idx_a] = listening_events[(u, a)]
                print "Inserted into UAM the triple (", u, ", ", a, ", ", listening_events[(u,a)], ")"

            except KeyError:        # if user u did not listen to artist a, we continue
                continue

    # Get sum of play events per user and per artist
    sum_pc_user = np.sum(UAM, axis=1)
    sum_pc_artist = np.sum(UAM, axis=0)

    # Normalize the UAM (simply by computing the fraction of listening events per artist for each user)
    no_users = UAM.shape[0]
    no_artists = UAM.shape[1]
    # np.tile: take sum_pc_user no_artists times (results in an array of length no_artists*no_users)
    # np.reshape: reshape the array to a matrix
    # np.transpose: transpose the reshaped matrix
    artist_sum_copy = np.tile(sum_pc_user, no_artists).reshape(no_artists, no_users).transpose()
    # Perform sum-to-1 normalization
    UAM = UAM / artist_sum_copy

    # Inform user
    print "UAM created. Users: " + str(UAM.shape[0]) + ", Artists: " + str(UAM.shape[1])

    helper.ensure_dir(OUTPUT_DIR)

    # Write everything to text file (artist names, user names, UAM)
    # Write artists to text file
    with open(ARTISTS_FILE, 'w') as outfile:             # "a" to append
        outfile.write('artist\n')
        for key in artists.keys():          # for all artists listened to by any user
            outfile.write(key + "\n")

    # Write users to text file
    with open(USERS_FILE, 'w') as outfile:
        outfile.write('user\n')
        for key in users.keys():            # for all users
            outfile.write(key + "\n")

    # Write UAM
    np.savetxt(UAM_FILE, UAM, fmt='%0.6f', delimiter='\t', newline='\n')

