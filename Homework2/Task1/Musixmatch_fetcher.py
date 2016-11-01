import helper # helper.py
import numpy as np
import urllib
import json
import csv
import re

# apicalls limited to 2000 a day
APIKEY_1 = 'a12b038ac3676f2943c701c4c1758f55'
APIKEY_2 = '96cc93f38f26db6650e40916f4380270'
APIKEY_3 = '192df0c0daaac4722a6f1e2c00b30f5d'
APIKEY_4 = '2b659ad7e0c7a554fc58294d93e68c06'

OUTPUT_DIR = './output/'
OUTPUT_DIR_MUSIXMATCH = OUTPUT_DIR + 'musixmatch/'
ARTIST_FILE = OUTPUT_DIR + 'artists.txt'
GENERATED_ARTISTS_FILE   = OUTPUT_DIR_MUSIXMATCH + 'artist_ids.txt'
GENERATED_ALBUM_IDS_FILE = OUTPUT_DIR_MUSIXMATCH + 'album_ids.txt'
GENERATED_TRACKS_FILE    = OUTPUT_DIR_MUSIXMATCH + 'album_tracks.txt'
GENERATED_LYRICS_FILE    = OUTPUT_DIR_MUSIXMATCH + 'lyrics.json'

MUSIXMATCH_URL = 'http://api.musixmatch.com/ws/1.1/'
FORMAT = 'json'

VERBOSE = True

NUMBER_OF_MAX_ARTISTS = 1000
NUMBER_OF_ALBUMS      = 3
NUMBER_OF_MAX_TRACKS  = 10
MAX_API_QUERIES = 2000

API_COUNTER = 600
MAX = 8000

def fetch_musixmatch_basic(method, additionalstring):
    global API_COUNTER

    # change APIKEY
    if API_COUNTER < MAX_API_QUERIES:
        key = APIKEY_1
    elif API_COUNTER < MAX_API_QUERIES * 2:
        key = APIKEY_2
    elif API_COUNTER < MAX_API_QUERIES * 3:
        key = APIKEY_3
    else:
        key = APIKEY_4

    API_COUNTER += 1

    url =   MUSIXMATCH_URL + \
            method + \
            '?apikey=' + key + \
            '&format=' + FORMAT + \
            '&' + additionalstring

    json_string = urllib.urlopen(url).read()

    return json.loads(json_string)
# /fetch_musixmatch_basic

def fetch_artist_by_term(artist_name):
    return fetch_musixmatch_basic('artist.search', 'q_artist=' + artist_name)
# /fetch_artist_by_term

def fetch_artist_albums(artist_id):
    return fetch_musixmatch_basic('artist.albums.get', 'artist_id=' + str(artist_id) + '&s_release_date=asc')
# /fetch_artist_albums

def fetch_artist_album_tracks(album_id):
    return fetch_musixmatch_basic('album.tracks.get', 'album_id=' + str(album_id) + '&f_has_lyrics')
# /fetch_artist_album_tracks

def fetch_lyrics_by_track_id(track_id):
    return fetch_musixmatch_basic('track.lyrics.get', 'track_id=' + str(track_id))
# /fetch_lyrics_by_track_id

def get_artist_ids(artist_name_array):
    artists_with_id = {}

    if VERBOSE:
        helper.log_highlight('Fetching Artist IDs')

    for index, artist_name in enumerate(artist_name_array, start = 1):
        response    = fetch_artist_by_term(artist_name)
        header      = response['message']['header']
        status_code = header['status_code']

        if VERBOSE:
            print 'Fetching ' + artist_name + ' [' + str(index) + ' of ' + str(NUMBER_OF_MAX_ARTISTS) + ']'

        if status_code is 200 and len(response['message']['body']['artist_list']) > 0:
            # always get the first artist
            chosen_artist    = response['message']['body']['artist_list'][0]['artist']
            chosen_artist_id = chosen_artist['artist_id']
            artists_with_id[chosen_artist_id] = artist_name
        else:
            if VERBOSE:
                print artist_name + ' not found'

    return artists_with_id
# /get_artist_ids

def get_artist_albums(artist_name_object, number_of_albums):
    artist_album_ids = {}
    counter = 1

    if VERBOSE:
        helper.log_highlight('Fetching albums of artists')

    for artist_id, artist_name in artist_name_object.items():
        response    = fetch_artist_albums(artist_id)
        header      = response['message']['header']
        status_code = header['status_code']

        if VERBOSE:
            print 'Fetching albums of ' + str(artist_id) + ' [' + str(counter) + ' of ' + str(len(artist_name_object)) + ']'

        if status_code is 200 and len(response['message']['body']['album_list']) > 0:
            albums = response['message']['body']['album_list']

            for index, album in enumerate(albums, start = 0):
                if index < number_of_albums:
                    album_id = album['album']['album_id']

                    try:
                        artist_album_ids[artist_id].append(album_id)
                    except:
                        artist_album_ids[artist_id] = []
                        artist_album_ids[artist_id].append(album_id)
        else:
            if VERBOSE:
                'Album ' + str(counter) + ' of ' + str(artist_id) + ' not found'

        counter += 1

    return artist_album_ids
# /get_artist_albums

def get_artist_album_tracks(artist_album_object, number_of_tracks_per_album):
    artist_album_tracks = {}
    counter = 1

    if VERBOSE:
        helper.log_highlight('Fetching tracks of albums')

    for artist_id, album_array in artist_album_object.items():
        if VERBOSE:
            print 'Fetching albums of artist ' + str(artist_id) + ' [' + str(counter) + ' of ' + str(len(artist_album_object)) + ']'

        for index, album_id in enumerate(album_array, start = 1):
            response    = fetch_artist_album_tracks(album_id)
            header      = response['message']['header']
            status_code = header['status_code']

            if VERBOSE:
                print '    Fetching tracks of album ' + str(album_id) + ' [' + str(index) + ' of ' + str(len(album_array)) + ']'

            if status_code is 200 and len(response['message']['body']['track_list']) > 0:
                tracks = response['message']['body']['track_list']

                for index, track in enumerate(tracks, start = 0):
                    if index < number_of_tracks_per_album:
                        track_id = track['track']['track_id']

                        try:
                            artist_album_tracks[artist_id].append(track_id)
                        except:
                            artist_album_tracks[artist_id] = []
                            artist_album_tracks[artist_id].append(track_id)
            else:
                if VERBOSE:
                    print '    Tracks of album ' + str(album_id) + ' not found'

        counter +=1

    return artist_album_tracks
# /get_artist_album_tracks

def get_lyrics_by_tracks(artist_tracks_id_object):
    artist_tracks_object = {}
    counter = 1
    musixmatch_regex = re.compile(r'\*.*\*\s*$') # this will delete **** This Lyrics is NOT... *** at the end of the string

    if VERBOSE:
        helper.log_highlight('Fetching lyrics of tracks')

    for artist_id, tracks in artist_tracks_id_object.items():
        if VERBOSE:
            print 'Fetching tracks of artist ' + str(artist_id) + ' [' + str(counter) + ' of ' + str(len(artist_tracks_id_object)) + ']'


        for index, track_id in enumerate(tracks, start = 1):
            response    = fetch_lyrics_by_track_id(track_id)
            header      = response['message']['header']
            status_code = header['status_code']

            if VERBOSE:
                print '    Fetching lyrics of track ' + str(track_id) + ' [' + str(index) + ' of ' + str(len(tracks)) + ']'

            if status_code is 200:
                lyrics = response['message']['body']['lyrics']['lyrics_body']

                lyrics_replaced = re.sub(r'\*.*\*\s*$', '', lyrics)

                try:
                    artist_tracks_object[artist_id] += lyrics_replaced
                except:
                    artist_tracks_object[artist_id] = ''
                    artist_tracks_object[artist_id] += lyrics_replaced

    return artist_tracks_object
# /get_lyrics_by_tracks

def save_txt(objects, filename):
    text = ''

    for key, value in objects.items():
        if type(value) is list:
            for val in value:
                text += str(key) + '\t' + str(val) + '\n'

        if type(value) is str:
            text += str(key) + '\t' + str(value) + '\n'


    text_file = open(OUTPUT_DIR_MUSIXMATCH + filename, 'w')

    text_file.write(text)
    text_file.close()
# /save_txt

def save_json(objects, filename):
    content   = json.dumps(objects, indent=4, sort_keys=True)
    json_file = open(OUTPUT_DIR_MUSIXMATCH + filename, 'w')

    json_file.write(content)
    json_file.close()
# /save_json

def read_txt(filename, multiple_values=False):
    file_contents = {}

    with open(filename, 'r') as f:
        reader  = csv.reader(f, delimiter='\t')      # create reader
        headers = reader.next()                     # skip header

        for row in reader:
            key   = row[0]
            value = row[1]

            if multiple_values:
                try:
                    file_contents[key].append(value)
                except:
                    file_contents[key] = []
                    file_contents[key].append(value)
            else:
                file_contents[key] = value

    return file_contents
# /read_file

# Main program
if __name__ == '__main__':
    artists = helper.read_csv(ARTIST_FILE)

    if type(NUMBER_OF_MAX_ARTISTS) is bool and NUMBER_OF_MAX_ARTISTS is True:
        NUMBER_OF_MAX_ARTISTS = len(artists)

    artists           = artists[:NUMBER_OF_MAX_ARTISTS]
    number_of_fetches = NUMBER_OF_MAX_ARTISTS * 2 + (NUMBER_OF_MAX_ARTISTS * NUMBER_OF_ALBUMS) * (1 + NUMBER_OF_MAX_TRACKS)

    if VERBOSE:
        helper.log_highlight('You will have ' + str(number_of_fetches) + ' queries to the musixmatch api')
        print ''
        print 'Artist queries: ' + str(NUMBER_OF_MAX_ARTISTS)
        print 'Album queries:  ' +  str(NUMBER_OF_MAX_ARTISTS)
        print 'Track queries:  ' + str(NUMBER_OF_MAX_ARTISTS * NUMBER_OF_ALBUMS)
        print 'Lyrics queries: ' + str((NUMBER_OF_MAX_ARTISTS * NUMBER_OF_ALBUMS) * NUMBER_OF_MAX_TRACKS)
        print ''
        print 'These numbers can vary if an artists has less albums, tracks or tracks with lyrics'
        print ''

    helper.ensure_dir(OUTPUT_DIR_MUSIXMATCH)

    # live fetching
    # fetched_artist_ids          = get_artist_ids(artists)
    # fetched_artist_album_ids    = get_artist_albums(fetched_artist_ids, NUMBER_OF_ALBUMS)
    # fetched_artist_album_tracks = get_artist_album_tracks(fetched_artist_album_ids, NUMBER_OF_MAX_TRACKS)
    # fetched_lyrics              = get_lyrics_by_tracks(fetched_artist_album_tracks)

    # fetching with stored data
    fetched_artist_ids          = read_txt(GENERATED_ARTISTS_FILE)
    fetched_artist_album_ids    = read_txt(GENERATED_ALBUM_IDS_FILE, True)
    fetched_artist_album_tracks = read_txt(GENERATED_TRACKS_FILE, True)
    # fetched_lyrics              = get_lyrics_by_tracks(fetched_artist_album_tracks)

    # save_txt(fetched_artist_ids, 'artist_ids.txt')
    # save_txt(fetched_artist_album_ids, 'album_ids.txt')
    # save_txt(fetched_artist_album_tracks, 'album_tracks.txt')
    # save_json(fetched_lyrics, 'lyrics.json')