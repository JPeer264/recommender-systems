import urllib
import csv
import json
import os

USER_FILE = './C1ku_users_extended.csv'
OUTPUT_DIR = './user_info/'

LASTFM_API_KEY = "8aa5abf299b1aaf6e4758f6ce3dc2fcf"
LASTFM_API_URL = "http://ws.audioscrobbler.com/2.0/"


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

def lfm_api_user_call(method, username):
    """
    triggers an api to the user api

    :param method: the method of the user api, e.g. gettopartists
    :param username: the data from this user

    :return: returns a json decoded object
    """
    url = LASTFM_API_URL + \
          "?method=user." + method + \
          "&user=" + username + \
          "&format=json" + \
          "&api_key=" + LASTFM_API_KEY

    # Perform API-call and save (comes as String formatted as JSON)
    json_string = urllib.urlopen(url).read()

    return json.loads(json_string)
# /lfm_api_user_call

def lfm_api_call_get_listening_history(username):
    """
    calls internally lfm_api_user_call with the method getrecenttracks

    :param username: the data from this user
    """
    return lfm_api_user_call("getrecenttracks", username)
# /lfm_api_call_get_listening_history

def limit_user(all_users, maxAmountOfUsers):
    """
    TODO do not print before return
    Get a minimum of 5 users and 50 unique artists and return users

    :param all_users: a list of all users
    :param maxAmountOfUsers: how many users should get saved

    :return: returns limite
    """
    limited_users    = []
    all_artist_names = []

    # Loop through list of all users
    for user in all_users:
        # Get artist-history from users via LastFM-API call
        # urllib.quote = Replace special characters in string
        top_artists = lfm_api_user_call("gettopartists", urllib.quote(user))

        # Get artists
        artists = top_artists['topartists']['artist']

        # Create list of artist-names (all_artists) by iterating
        # through list with all artist information (artists)
        for artist in artists:
            artist_name = artist['name']
            all_artist_names.append(artist_name)

        # Fill list with users
        limited_users.append(user)

        # Delete duplicates from all_artist_names and save to new list all_artist_names
        all_artist_names = get_unique_items(all_artist_names)

        # Limit amount of unique artists to a minimum of 50 and limit amount of users to minimum of 5
        # If true - stop for loop and return users (limited_users)
        if len(all_artist_names) >= 50 and len(limited_users) >= maxAmountOfUsers:
            print len(all_artist_names)
            print len(limited_users)
            return limited_users
# /limit_user

def get_unique_items(iterable):
    """
    Deletes duplicates in array
    https://stackoverflow.com/questions/32664180/why-does-removing-duplicates-from-a-list-produce-none-none-output

    :param iterable: an array to remove duplicates

    :return: an array with no duplicates
    """
    seen = set()
    result = []
    for item in iterable:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result
# /get_unique_items

def lfm_prepare_history_string(track, user):
    """
    prepares a string for the txt file

    :param track: an object with tracks
    :param user: an user object with id and name

    :return: creates a string with metadata combined with tabs
    """
    user_id   = "USER_ID"
    user_name = "USER_NAME"

    for t in track:
        artist_id   = t['artist']['mbid']
        artist_name = t['artist']['#text']
        track_id    = t['mbid']
        track_name  = t['name']
        timestamp   = t['date']['uts']

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
        one_track = lfm_api_call_get_listening_history(user)['recenttracks']['track']
        print lfm_prepare_history_string(one_track, user)

    return
# /lfm_save_history_of_users

# Main
if __name__ == "__main__":
    users = read_user_file(USER_FILE)

    limited_users = limit_user(users, 5)
    lfm_save_history_of_users(limited_users)

# CTRL + Shift + E for Debug
# CTRL + Shift + R for Run