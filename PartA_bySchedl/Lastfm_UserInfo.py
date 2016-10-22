import urllib
import os
import csv
import json


OUTPUT_DIR = "./user_info/"
USER_FILE = "./lastfm_users_20.csv"

LASTFM_API_KEY = "4cb074e4b8ec4ee9ad3eb37d6f7eb240"
LASTFM_API_URL = "http://ws.audioscrobbler.com/2.0/"

def read_user_file(uf):
    users = []       # init list
    with open(uf, 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        for r in reader:
#            print r[0]
            users.append(r[0])
    return users

def lfm_api_call_getinfo(user):
    url = LASTFM_API_URL + "?method=user.getinfo&user=" + urllib.quote(user) + \
          "&format=json" + "&api_key=" + LASTFM_API_KEY
    content = urllib.urlopen(url).read()
    return content

# Main program
if __name__ == '__main__':
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    users = read_user_file(USER_FILE)

    for u in users:
        # call Last.fm API: user.getInfo
        c = lfm_api_call_getinfo(u)
#        print c

        output_file = OUTPUT_DIR + "/" + u + '.json'
        fh = open(output_file, 'w')
        fh.write(c)
        fh.close()

        # parse json
        user_data = json.loads(c)
        print user_data
        print 'User name: ' + user_data["user"]["name"]
        print 'PC: ' + user_data["user"]["playcount"]
        print 'Country: ' + user_data["user"]["country"]
        print 'Age: ' + user_data["user"]["age"]
        print 'Gender: ' +  user_data["user"]["gender"]
        print 'Registration: ' + user_data["user"]["registered"]["unixtime"]
