import urllib
import csv
import json
import os

USER_FILE = './C1ku_users_extended.csv'
OUTPUT_DIR = './user_info/'

LASTFM_API_KEY = "8aa5abf299b1aaf6e4758f6ce3dc2fcf"
LASTFM_API_URL = "http://ws.audioscrobbler.com/2.0/"

test_track = {"artist":{"#text":"Les Savy Fav","mbid":"c507bf8f-2ac0-47e5-b7c6-4afd61a977e2"},"name":"Scout's Honor","streamable":"0","mbid":"df324b6c-5899-4391-9468-f3017a198832","album":{"#text":"3/5","mbid":"b49dc70e-424d-42c3-b844-ea513d5474d7"},"url":"https://www.last.fm/music/Les+Savy+Fav/_/Scout%27s+Honor","image":[{"#text":"https://lastfm-img2.akamaized.net/i/u/34s/2552a65281d84644b99c2976e1ff5f10.png","size":"small"},{"#text":"https://lastfm-img2.akamaized.net/i/u/64s/2552a65281d84644b99c2976e1ff5f10.png","size":"medium"},{"#text":"https://lastfm-img2.akamaized.net/i/u/174s/2552a65281d84644b99c2976e1ff5f10.png","size":"large"},{"#text":"https://lastfm-img2.akamaized.net/i/u/300x300/2552a65281d84644b99c2976e1ff5f10.png","size":"extralarge"}],"date":{"uts":"1221598396","#text":"16 Sep 2008, 20:53"}}

# Read csv-File
def read_user_file(uf):

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


def lfm_api_user_call(method, user):
    url = LASTFM_API_URL + \
          "?method=user." + method + \
          "&user=" + user + \
          "&format=json" + \
          "&api_key=" + LASTFM_API_KEY

    return urllib.urlopen(url).read()

# Get a minimum of 5 users and 50 unique artists and return users
def limit_user(all_users):

    limited_users = []
    all_artist_names = []

    # Loop through list of all users
    for user in all_users:

        # Get artist-history from users via LastFM-API call
        # urllib.quote = Replace special characters in string
        url = LASTFM_API_URL + "?method=user.gettopartists&user=" + urllib.quote(user) + \
              "&format=json" + "&api_key=" + LASTFM_API_KEY

        # Perform API-call and save (comes as String formatted as JSON)
        top_artists = urllib.urlopen(url).read()

        # Decode JSON-File
        artists = json.loads(top_artists)

        # Get artists from decoded JSON File
        all_artists_user = artists['topartists']['artist']

        # Create list of artist-names (all_artists) by iterating
        # through list with all artist information (all_artists_user)
        for artist in all_artists_user:
            artist_name = artist['name']
            all_artist_names.append(artist_name)

        # Fill list with users
        limited_users.append(user)

        # Delete duplicates from all_artist_names and save to new list all_artist_names
        all_artist_names = getUniqueItems(all_artist_names)

        # Limit amount of unique artists to a minimum of 50 and limit amount of users to minimum of 5
        # If true - stop for loop and return users (limited_users)
        if len(all_artist_names) >= 50 and len(limited_users) >= 5:
            print len(all_artist_names)
            print len(limited_users)
            return limited_users


# Deletes duplicates in array
def getUniqueItems(iterable):
    seen = set()
    result = []
    for item in iterable:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def lfm_api_call_get_listening_history(user):
    return lfm_api_user_call("getrecenttracks", user)


def lfm_prepare_history_string(history, user):
    user_id = "USER_ID"
    user_name = "USER_NAME"

    for h in history:
        artist_id = h['artist']['mbid']
        artist_name = h['artist']['#text']
        track_id = h['mbid']
        track_name = h['name']
        timestamp = h['date']['uts']

    return user_id + "\t" + user_name + "\t" + artist_id + "\t" + artist_name + "\t" + track_id + "\t" + track_name + "\t" + timestamp

def lfm_save_history_of_users(users):

    for user in users:
        one_track = json.loads(lfm_api_call_get_listening_history(user))['recenttracks']['track']
        print lfm_prepare_history_string(one_track, user)
    return


# Main
if __name__ == "__main__":
    users = read_user_file(USER_FILE)
    # print lfm_prepare_history_string(test_track)
    limited_users = limit_user(users)
    lfm_save_history_of_users(limited_users)

# CTRL + Shift + E for Debug
# CTRL + Shift + R for Run