import urllib
import csv
import json

USER_FILE = './lastfm_users_5.csv'
OUTPUT_DIR = './user_info/'

# generated on last.fm/api
LASTFM_API_KEY = '9bd61f2b7c2931520f8409006d5014a3'
LASTFM_API_URL = 'http://ws.audioscrobler.com/2.0/'

# read_user_file def -> function
def read_user_file(uf):
    users = []

    with open(uf, 'r') as f:
        reader = csv.reader(f, delimiter='\t')

        for r in reader:
            print r[0]
# /read_user_file

if __name__ == '__main__':
    read_user_file(USER_FILE)