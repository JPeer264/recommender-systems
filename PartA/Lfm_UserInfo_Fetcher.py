import urllib
import csv
import json
import os

USER_FILE = './lastfm_users_20.csv'
OUTPUT_DIR = './user_info/'

# generated on last.fm/api
LASTFM_API_KEY = '9bd61f2b7c2931520f8409006d5014a3'
LASTFM_API_URL = 'http://ws.audioscrobbler.com/2.0/'

# read_user_file def -> function
def read_user_file(uf):
    users = []

    with open(uf, 'r') as f:
        reader = csv.reader(f, delimiter='\t')

        for r in reader:
            users.append(r[0])

        return users
# /read_user_file

# lfm_api_call
def lfm_api_call(user):
    url  = LASTFM_API_URL + '?method=user.getinfo&user=' + urllib.quote(user) + '&format=json&api_key=' + LASTFM_API_KEY
    cont = urllib.urlopen(url).read()

    return cont
# /lfm_api_call

# Main
if __name__ == '__main__':
    users = read_user_file(USER_FILE)
    print users

    # mkdir in py
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # Loop over all users
    for u in users:
        content   = lfm_api_call(u)
        ofile     = OUTPUT_DIR + '/' + u + '.json'

        f = open(ofile, 'w')
        f.write(content)
        f.close()

        json_user = json.loads(content)

        print json_user['user']['name']
        print json_user['user']['age']
        print json_user['user']['gender']
        print json_user['user']['playlists']
# /Main