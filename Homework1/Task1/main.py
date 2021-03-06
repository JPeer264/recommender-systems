import helper  # helper.py
import os

import json
import numpy as np

## TODO set in txt files
#       first row should describe the dataset

USER_FILE  = '../Testfiles/C1ku_users_extended.csv'
BASE_DIR   = './output'
USER_LIST_FILE  = BASE_DIR + '/user_list.csv'
OUTPUT_DIR = BASE_DIR + '/user_info/'
USER_CHARACTERISTICS_FILE = OUTPUT_DIR + "users_characteristics.txt"
USER_LISTENING_HISTORY = OUTPUT_DIR + "listening_history.txt"

VERBOSE = True # set to True prints information about the current state into the console
VERBOSE_DEPTH = 2 # describes how deep the verbose mode goes - maximum 2

MAX_RECENT_TRACK_PER_PAGE = 200 # maximum is 200
MAX_RECENT_TRACK_PAGES = 6

def get_user_friends(all_users, limit_user):
    all_user_friends      = []
    all_users_and_friends = []
    all_unique_users      = []
    user_list             = iter(all_users)
    list                  = all_users

    if VERBOSE:
        helper.log_highlight("Fetching friends of user")

    for index, user in enumerate(user_list, start = 1):
        user_get_friends = helper.api_user_call("getfriends", user, "")

        try:
            user_friends = user_get_friends['friends']['user']

            if VERBOSE:
                print "Fetching friends of " + user

            for friend in user_friends:
                friend_name = friend['name']
                all_user_friends.append(friend_name)

                if index > limit_user:
                    all_users_and_friends = list + all_user_friends
                    all_unique_users = helper.get_unique_items(all_users_and_friends)
                    np.savetxt(USER_LIST_FILE, all_unique_users, delimiter=",", fmt='%s')
                    # print len(all_users_and_friends)
                    # print len(all_unique_users)

                    if VERBOSE:
                        print "\nSuccessfully created " + USER_LIST_FILE
                        print "Successfully fetched friends\n"

                    return

        except KeyError:
            print ""
            print "SKIP: User has no friends"
            next(user_list)

        except Exception:
            print ""
            print "ERROR: "
            print(traceback.format_exc())

    all_users_and_friends = list + all_user_friends
    all_unique_users      = helper.get_unique_items(all_users_and_friends)

    np.savetxt(USER_LIST_FILE, all_unique_users, delimiter=",", fmt='%s')

    return
# /get_user_friends

def limit_user(all_users, min_amount_of_users, play_count, min_amount_of_artists_user, min_amount_of_unique_artists_all_users):
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
    limited_users    = []
    all_artist_names = []
    user_list        = iter(all_users)
    counter          = 1

    if VERBOSE:
        helper.log_highlight("Limit users - data cleansing")

    # Loop through list of all users
    for index, user in enumerate(user_list, start = 1):

        # Get artist-history from users via LastFM-API call
        top_artists = helper.api_user_call("gettopartists", user, "")

        # Error Handling: error = User not found
        try:
            artists        = top_artists['topartists']['artist']
            artist_counter = 0

        except KeyError:
            print ""
            print "SKIP: User has no artists"
            next(user_list)

        except Exception:
            print ""
            print "ERROR:"
            print(traceback.format_exc())

        # Loop through artists-list and evaluate playcount
        for artist in artists:

            # Save playcount for artist
            artist_playcount = int(artist['playcount'])

            # Check if playcount of artist is equal or greater than defined play_count and
            # if true add artist to all_artist_names list
            if artist_playcount >= play_count:
                artist_counter += 1
                artist_name    = artist['name']

                all_artist_names.append(artist_name.encode('utf-8'))

        if VERBOSE and VERBOSE_DEPTH == 2:
            print "    Artists (not unique): " + str(len(all_artist_names))

        # Data cleansing: only add users with more than 10 unique artists
        if artist_counter > min_amount_of_artists_user:
            if VERBOSE:
                print "Fetched satisfying user [" + str(counter) + " of " + str(min_amount_of_users) + "]"

            # Fill list with users
            limited_users.append(user)
            counter += 1

            # Delete duplicates from all_artist_names and save to new list all_artist_names
            all_artist_names = helper.get_unique_items(all_artist_names)

            if VERBOSE and VERBOSE_DEPTH == 2:
                print "    Artists (unique):     " + str(len(all_artist_names))

            # Limit amount of unique artists for all users to a defined minimum (min_amount_of_unique_artists_all_users)
            # and limit amount of all users to a defined minimum (min_amount_of_users)
            # If true - stop for loop and return users (limited_users)
            if  len(all_artist_names) >= min_amount_of_unique_artists_all_users \
                and len(limited_users) >= min_amount_of_users:

                np.savetxt(OUTPUT_DIR + "/limited_user_list.csv", limited_users, delimiter=",", fmt='%s')
                np.savetxt(OUTPUT_DIR + "/all_artist_names.csv", all_artist_names, delimiter=",", fmt='%s')

                if VERBOSE:
                    print "\nData cleansing successful\n"

                return limited_users
# /limit_user

def lfm_prepare_history_string(track, user_name):
    """
    prepares a string for the txt file

    :param track: an object with tracks
    :param user: an user object with id and name

    :return: creates a string with metadata combined with tabs
    """
    artist_id   = track['artist']['mbid']
    artist_name = track['artist']['#text']
    track_id    = track['mbid']
    track_name  = track['name']
    timestamp   = track['date']['uts']

    user_history = user_name + "\t" + artist_id + "\t" + artist_name + "\t" + track_id + "\t" + track_name + "\t" + timestamp

    return user_history.encode('utf-8')
# /lfm_prepare_history_string

def lfm_prepare_user_characteristics_string(user):
    """
    prepares a string for the txt file

    :param track: an object with tracks
    :param user: an user object with id and name

    :return: creates a string with metadata combined with tabs
    """
    user_name       = user['name']
    user_country    = user['country']
    user_age        = user['age']
    user_gender     = user['gender']
    user_playcount  = user['playcount']
    user_playlists  = user['playlists']
    user_registered = user['registered']['unixtime']

    user_characteristics = user_name + "\t" + user_country + "\t" + user_age + "\t" + user_gender + "\t" + user_playcount + "\t" + user_playlists + "\t" + user_registered

    return user_characteristics.encode('utf-8')

# /lfm_prepare_history_string

def lfm_save_history_of_users(users):
    """
    saves the history of users

    :param users: an array of users
    """
    content = ""
    user_iter = iter(users)

    if VERBOSE:
        helper.log_highlight("Saving listening history")

    # mkdir in py
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    for index, user in enumerate(user_iter, start = 1):

        try:
            if VERBOSE:
                print "Fetch recent tracks from user [" + str(index) + " of " + str(len(users)) + "]"

            for page_number in range(1, (MAX_RECENT_TRACK_PAGES + 1)):
                recent_tracks = helper.api_user_call("getrecenttracks", user, "&limit=" + str(MAX_RECENT_TRACK_PER_PAGE) + "&page=" + str(page_number))['recenttracks']['track']

                for index, recent_track in enumerate(recent_tracks, start = 1):
                    if VERBOSE and VERBOSE_DEPTH == 2:
                        print "    Fetch recent track [" + str(index * page_number) + " of " + str(len(recent_tracks) * MAX_RECENT_TRACK_PAGES) + "]"

                    listening_history = lfm_prepare_history_string(recent_track, user)
                    content           += listening_history + "\n"
        except Exception:
            print "EXCEPTION lfm_save_history_of_users"
            next(user_iter)

    text_file   = open(USER_LISTENING_HISTORY, 'w')

    text_file.write(content)
    text_file.close()

    if VERBOSE:
        print "\nSuccessfully created " + USER_LISTENING_HISTORY + "\n"

    return

# /lfm_save_history_of_users

def lfm_save_user_characteristics(users):
    content = ""
    users_iter = iter(users)

    if VERBOSE:
        helper.log_highlight("Saving user characteristics")

    # mkdir in py
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    for index, user in enumerate(users_iter, start = 1):
        try:
            if VERBOSE:
                print "Fetch user characteristics [" + str(index) + " of " + str(len(users)) + "]"

            user_info   = helper.api_user_call("getinfo", user, "")
            user        = user_info['user']
            user_string = lfm_prepare_user_characteristics_string(user)
            content     += user_string + "\n"

        except:
            print "EXCEPTION lfm_save_user_characteristics"
            next(users_iter)

    text_file   = open(USER_CHARACTERISTICS_FILE, 'w')

    text_file.write(content)
    text_file.close()

    if VERBOSE:
        print "\nSuccessfully created " + USER_CHARACTERISTICS_FILE + "\n"

    return
# /lfm_save_user_characteristics

# Main
if __name__ == "__main__":
    users = helper.read_csv(USER_FILE)

    get_user_friends(users, 10)

    user_list     = helper.read_csv(USER_LIST_FILE)
    limited_users = limit_user(user_list, 5, 5, 5, 10)
    # limited_users = limit_user(user_list, 500, 500, 10, 50)

    lfm_save_history_of_users(limited_users)
    lfm_save_user_characteristics(limited_users)
# /Main
