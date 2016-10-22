#
# Collect listening information for at least 500 users
# and at least 5,000 unique artist
#

import urllib
import csv
import json
import os

USER_FILE = './C1ku_users_extended.csv'
OUTPUT_DIR = './user_info'

LASTFM_API_KEY = "1619f1bb56a7d077193577b2abcec351"
LASTFM_API_URL = "https://ws.audioscrobbler.com/2.0/"


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


# Main
if __name__ == "__main__":
    users = read_user_file(USER_FILE)
    print limit_user(users)
