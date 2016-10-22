import urllib
import csv
import json
import os

USER_FILE = './lastfm_users_5.csv'
OUTPUT_DIR = './user_info/'

LASTFM_API_KEY = "8aa5abf299b1aaf6e4758f6ce3dc2fcf"
LASTFM_API_URL = "http://ws.audioscrobbler.com/2.0/"

test_track = {"artist":{"#text":"Les Savy Fav","mbid":"c507bf8f-2ac0-47e5-b7c6-4afd61a977e2"},"name":"Scout's Honor","streamable":"0","mbid":"df324b6c-5899-4391-9468-f3017a198832","album":{"#text":"3/5","mbid":"b49dc70e-424d-42c3-b844-ea513d5474d7"},"url":"https://www.last.fm/music/Les+Savy+Fav/_/Scout%27s+Honor","image":[{"#text":"https://lastfm-img2.akamaized.net/i/u/34s/2552a65281d84644b99c2976e1ff5f10.png","size":"small"},{"#text":"https://lastfm-img2.akamaized.net/i/u/64s/2552a65281d84644b99c2976e1ff5f10.png","size":"medium"},{"#text":"https://lastfm-img2.akamaized.net/i/u/174s/2552a65281d84644b99c2976e1ff5f10.png","size":"large"},{"#text":"https://lastfm-img2.akamaized.net/i/u/300x300/2552a65281d84644b99c2976e1ff5f10.png","size":"extralarge"}],"date":{"uts":"1221598396","#text":"16 Sep 2008, 20:53"}}

def read_userfile(uf):
    users = []
    with open(uf, 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        for r in reader:
            users.append(r[0])
    return users

def lfm_api_user_call(method, user):
    url = LASTFM_API_URL + \
          "?method=user." + method + \
          "&user=" + user + \
          "&format=json" + \
          "&api_key=" + LASTFM_API_KEY

    return urllib.urlopen(url).read()


def lfm_api_call_get_listening_history(user):
    return lfm_api_user_call("getrecenttracks", "badmusic")


def lfm_prepare_history_string(history):
    user_id = "USER_ID"
    user_name = "USER_NAME"

    artist_id = history['artist']['mbid']
    artist_name = history['artist']['#text']
    track_id = history['mbid']
    track_name = history['name']
    timestamp = history['date']['uts']

    return user_id + "\t" + user_name + "\t" + artist_id + "\t" + artist_name + "\t" + track_id + "\t" + track_name + "\t" + timestamp


# Main
if __name__ == "__main__":
    users = read_userfile(USER_FILE)
    # print lfm_api_call_getListeningHistory(1)
    print lfm_prepare_history_string(test_track)
#    print users

# CTRL + Shift + E for Debug
# CTRL + Shift + R for Run