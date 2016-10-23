import csv
import helper  # helper.py
import json
import numpy as np

USER_FILE = './Testfiles/C1ku_users_extended.csv'
USER_LIST = './user_list.csv'
OUTPUT_DIR = './user_info/'


def read_user_file(uf):
    """
    Read csv-File
    TODO delete this - discuss in team
    """
    users = []

    # Open file - read only
    with open(uf, 'r') as f:
        # Read csv - Files and interprete tabs as new item
        reader = csv.reader(f, delimiter='\t')

        # Loop through reader
        for r in reader:
            # Append user-names only
            users.append(r[0])

        return users

def get_user_friends(all_users):

    all_user_friends = []
    all_users_and_friends = []
    all_unique_users =  []
    user_list = iter(all_users)
    list = all_users
    counter = 0

    for user in user_list:
        counter += 1
        user_get_friends = helper.api_user_call("getfriends", user)

        try:
            user_friends = user_get_friends['friends']['user']

            for friend in user_friends:
                friend_name = friend['name']
                all_user_friends.append(friend_name)

                print ""
                print "Counter: " + str(counter)
                print "all_user_friends: " +  str(len(all_user_friends))
        except:
            next(user_list)

    all_users_and_friends = list + all_user_friends
    all_unique_users = helper.get_unique_items(all_users_and_friends)
    np.savetxt("user_list.csv", all_unique_users, delimiter=",", fmt='%s')
    # print len(all_users_and_friends)
    # print len(all_unique_users)

    return


def limit_user(all_users, min_amount_of_users, play_count, min_amount_of_artists_user,
               min_amount_of_unique_artists_all_users):
    """
    Get a minimum of 5 users and 50 unique artists and return users

    Data cleansing:
    Only add user if user if it is equal to min_amount_of_artists and
    if the atrists playcount id equal to play_count

    :param all_users: a list of all users
    :param min_amount_of_users: how many users should get saved
    :param play_count: playcount of artist
    :param min_amount_of_artists_user: min. amount of unique artists per user
    :param min_amount_of_unique_artists_all_users: min. amount of unique artists for all users

    :return: returns limited_users
    """
    limited_users = []
    all_artist_names = []
    user_count = 0

    user_list = iter(all_users)

    # Loop through list of all users
    for user in user_list:

        user_count += 1
        print "Usercount: " + str(user_count)

        # Get artist-history from users via LastFM-API call
        top_artists = helper.api_user_call("gettopartists", user)

        # Error Handling: error = User not found
        try:
            artists = top_artists['topartists']['artist']
            artist_counter = 0
        except:
            print "EXCEPTION"
            next(user_list)

        # Loop through artists-list and evaluate playcount
        for artist in artists:

            # Save playcount for artist
            artist_playcount = int(artist['playcount'])

            # Check if playcount of artist is equal or greater than defined play_count and
            # if true add artist to all_artist_names list
            if artist_playcount >= play_count:
                artist_counter += 1
                artist_name = artist['name']
                all_artist_names.append(artist_name)

        print "Artist names (not unique): " + str(len(all_artist_names))

        # Data cleansing: only add users with more than 10 unique artists
        if artist_counter > min_amount_of_artists_user:

            # Fill list with users
            limited_users.append(user)
            print "ADDED USER: " +  str(user_count)

            # Delete duplicates from all_artist_names and save to new list all_artist_names
            all_artist_names = helper.get_unique_items(all_artist_names)

            print "Artist names (unique): " + str(len(all_artist_names))

            # Limit amount of unique artists for all users to a defined minimum (min_amount_of_unique_artists_all_users)
            # and limit amount of all users to a defined minimum (min_amount_of_users)
            # If true - stop for loop and return users (limited_users)
            if len(all_artist_names) >= min_amount_of_unique_artists_all_users and len(
                    limited_users) >= min_amount_of_users:
                return limited_users


# /limit_user
def lfm_prepare_history_string(track, user):
    """
    prepares a string for the txt file

    :param track: an object with tracks
    :param user: an user object with id and name

    :return: creates a string with metadata combined with tabs
    """
    user_id = "USER_ID"
    user_name = "USER_NAME"

    for t in track:
        artist_id = t['artist']['mbid']
        artist_name = t['artist']['#text']
        track_id = t['mbid']
        track_name = t['name']
        timestamp = t['date']['uts']

    return user_id + "\t" + user_name + "\t" + artist_id + "\t" + artist_name + "\t" + track_id + "\t" + track_name + "\t" + timestamp


# /lfm_prepare_history_string
def lfm_save_history_of_users(users):
    """
    TODO no printing
    TODO save into file
    saves the history of users

    :param users: an array of users
    """
    for user in users:
        recent_tracks = helper.api_user_call("getrecenttracks", user)
        one_track = recent_tracks['recenttracks']['track']
        print lfm_prepare_history_string(one_track, user)

    return


# /lfm_save_history_of_users

# Main
if __name__ == "__main__":
    users = read_user_file(USER_FILE)
    get_user_friends(users)
    user_list = read_user_file(USER_LIST)

    # :param all_users: a list of all users
    # :param min_amount_of_users: how many users should get saved min
    # :param play_count: playcount of artist
    # :param min_amount_of_artists_user: min. amount of unique artists per user
    # :param min_amount_of_unique_artists_all_users: min. amount of unique artists for all users
    limited_users = limit_user(user_list, 500, 500, 10, 50)

    #print limited_users
    #lfm_save_history_of_users(limited_users)



# CTRL + Shift + E for Debug
# CTRL + Shift + R for Run
