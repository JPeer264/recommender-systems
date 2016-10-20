import urllib
import csv
import json

USER_FILE = './lastfm_user_5.csv'
OUTPUT_DIR = './user_info/'

# generated on last.fm/api
LASTFM_API_KEY = 'fc2148c763e24a6b0d93f37ae1c7a00c'
LASTFM_API_URL = 'http://ws.audioscrobler.com/2.0/'

# def -> function
def read_user_file(uf):
    users = []

    with open(uf, 'r')