import urllib
import json

USER_FILE = './C1ku_users_extended.csv'
OUTPUT_DIR = './user_info/'

LASTFM_API_KEY = "8aa5abf299b1aaf6e4758f6ce3dc2fcf"
LASTFM_API_URL = "http://ws.audioscrobbler.com/2.0/"

def api_user_call(method, username):
    """
    triggers an api to the user api

    :param method: the method of the user api, e.g. gettopartists
    :param username: the data from this user

    :return: returns a json decoded object
    """
    # urllib.quote = Replace special characters in string
    url = LASTFM_API_URL + \
          "?method=user." + method + \
          "&user=" + urllib.quote(username) + \
          "&format=json" + \
          "&api_key=" + LASTFM_API_KEY

    # Perform API-call and save (comes as String formatted as JSON)
    json_string = urllib.urlopen(url).read()

    return json.loads(json_string)
# /lfm_api_user_call

def get_unique_items(iterable):
    """
    Deletes duplicates in array
    https://stackoverflow.com/questions/32664180/why-does-removing-duplicates-from-a-list-produce-none-none-output

    :param iterable: an array to remove duplicates

    :return: an array with no duplicates
    """
    seen   = set()
    result = []

    for item in iterable:
        if item not in seen:
            seen.add(item)
            result.append(item)

    return result
# /get_unique_items
