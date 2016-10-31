import helper # helper.py
import numpy as np
import urllib
import json
import re

# apicalls limited to 2000 a day
APIKEY_1 = 'a12b038ac3676f2943c701c4c1758f55'
APIKEY_2 = '96cc93f38f26db6650e40916f4380270'

OUTPUT_DIR = './output/musixmatch/'

MUSIXMATCH_URL = 'http://api.musixmatch.com/ws/1.1/'
FORMAT = 'json'

VERBOSE = True

def fetch_musixmatch_basic(method, additionalstring):
    API_COUNTER = 0
    # change APIKEY
    if API_COUNTER < 2000:
        key = APIKEY_1
    else:
        key = APIKEY_2

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

    for artist_name in artist_name_array:
        response    = fetch_artist_by_term(artist_name)
        header      = response["message"]["header"]
        status_code = header["status_code"]

        if VERBOSE:
            print "Fetching"

        if status_code is 200:
            # always get the first artist
            chosen_artist    = response['message']['body']['artist_list'][0]['artist']
            chosen_artist_id = chosen_artist['artist_id']
            artists_with_id[chosen_artist_id] = artist_name

    return artists_with_id
# /get_artist_ids

def get_artist_albums(artist_name_object, number_of_albums):
    artist_album_ids = {}

    for artist_id, artist_name in artist_name_object.items():
        response    = fetch_artist_albums(artist_id)
        header      = response["message"]["header"]
        status_code = header["status_code"]

        if status_code is 200:
            albums = response['message']['body']['album_list']

            for index, album in enumerate(albums, start = 0):
                if index < number_of_albums:
                    album_id = album['album']['album_id']

                    try:
                        artist_album_ids[artist_id].append(album_id)
                    except:
                        artist_album_ids[artist_id] = []
                        artist_album_ids[artist_id].append(album_id)

    return artist_album_ids
# /get_artist_albums

def get_artist_album_tracks(artist_album_object, number_of_tracks_per_album):
    artist_album_tracks = {}

    for artist_id, album_array in artist_album_object.items():
        for album_id in album_array:
            response    = fetch_artist_album_tracks(album_id)
            header      = response["message"]["header"]
            status_code = header["status_code"]

            if status_code is 200:
                tracks = response['message']['body']['track_list']

                for index, track in enumerate(tracks, start = 0):
                    if index < number_of_tracks_per_album:
                        track_id = track['track']['track_id']

                        try:
                            artist_album_tracks[artist_id].append(track_id)
                        except:
                            artist_album_tracks[artist_id] = []
                            artist_album_tracks[artist_id].append(track_id)

    return artist_album_tracks
# /get_artist_album_tracks

def get_lyrics_by_tracks(artist_tracks_id_object):
    artist_tracks_object = {}
    musixmatch_regex = re.compile(r'\*.*\*\s*$') # this will delete **** This Lyrics is NOT... *** at the end of the string

    for artist_id, tracks in artist_tracks_id_object.items():
        for track_id in tracks:
            response    = fetch_lyrics_by_track_id(track_id)
            header      = response["message"]["header"]
            status_code = header["status_code"]

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
    text = ""

    for key, value in objects.items():
        if type(value) is list:
            for val in value:
                text += str(key) + '\t' + str(val) + '\n'

        if type(value) is str:
            text += str(key) + '\t' + str(value) + '\n'


    text_file   = open(OUTPUT_DIR + filename, 'w')

    text_file.write(text)
    text_file.close()
# /save_txt

def save_json(objects, filename):
    content   = json.dumps(objects, indent=4, sort_keys=True)
    json_file = open(OUTPUT_DIR + filename, 'w')

    json_file.write(content)
    json_file.close()
# /save_json

# Main program
if __name__ == '__main__':
    artists = [
        'eminem',
        'metallica',
        'prodigy',
        'sdp'
    ]

    helper.ensure_dir(OUTPUT_DIR)

    fetched_artist_ids          = get_artist_ids(artists)
    fetched_artist_album_ids    = get_artist_albums(fetched_artist_ids, 2)
    fetched_artist_album_tracks = get_artist_album_tracks(fetched_artist_album_ids, 1)
    fetched_lyrics              = get_lyrics_by_tracks(fetched_artist_album_tracks)

    # mock data
    # fetched_artist_ids          = {64: 'metallica', 13816: 'prodigy', 426: 'eminem'} #get_artist_ids(artists)
    # fetched_artist_album_ids    = {64: [20484539, 10380391], 426: [13169182, 10485240]} #get_artist_albums(fetched_artist_ids, 2)
    # fetched_artist_album_tracks = {64: [80519275, 16237402], 426: [17638929, 3253605]} #get_artist_album_tracks(fetched_artist_album_ids, 1)
    # fetched_lyrics              = {64: u"My life suffocates\nPlanting seeds of hate\nI've loved, turned to hate\nTrapped far beyond my fate\n\nI give, you take\nThis life that I forsake\nBeen cheated of my youth\nYou turned this lie to truth\nAnger, misery\nYou'll suffer unto me\n\nHarvester of sorrow\nLanguage of the mad\n\nHarvester of sorrow\n...\n\nI've got somethin' to say\nI killed your baby today and it\nDoesn't matter much to me\nAs long as it's dead\n\nI've got somethin' to say\nI raped your mother today and it\nDoesn't matter much to me\nAs long as she spread\n...\n\n", 426: u"Now I don't really care what you call me\nYou can even call me cold\nThese bitches knew as soon as they saw me\nIt's never me they'll get the privilege to know\nI roll like a desperado, now I never know where I'm gonna go\nStill I ball like there's no tomorrow\nUntil it's over and that's all she wrote\nYou're starin' straight into a barrel of hate, terrible fate\nNot even a slim chance to make a narrow escape\n\nCupid shot his arrow and missed, wait Sarah you're late\nYour train left; mascara and eggs smeared on your face\n\nNight's over, good bye ho I thought that I told ya\n\nThat spilled nut ain't nothing to cry over\nNever should've came within range of my Rover\nShould've known I was trouble soon as I rolled up\nAny chick who's dumb enough after I blindfold her\nTo still come back to the crib\nMust want me to mess with her mind, hold up\nShe mistook me for some high roller, well I won't buy her soda\nUnless it's Rock & Rye Cola (Faygo's cheaper)\nBuy you a bag of Fritos?\nI wouldn't let you eat the fuckin' chip on my shoulder\n...\n\nStep by step, heart to heart\nLeft right left, we all fall down\n\nStep by step, heart to heart, left right left\nWe all fall down like toy soldiers\nBit by bit, torn apart, we never win\nBut the battle wages on for toy soldiers\n\nI'm supposed to be the soldier who never blows his composure\nEven though I hold the weight of the whole world on my shoulders\nI am never supposed to show it, my crew ain't supposed to know it\nEven if it means goin' toe to toe with a Benzino it don't matter\n\nI'd never drag them in battles that I can handle unless\nI absolutely have to I'm supposed to set an example\nI need to be the leader, my crew looks for me to guide 'em\nIf some shit ever just pop off, I'm supposed to be beside 'em\n\nThat Ja shit I tried to squash it, it was too late to stop it\nThere's a certain line you just don't cross and he crossed it\nI heard him say Hailie`s name on a song and I just lost it\nIt was crazy, this shit went way beyond some Jay-Z and nas shit\n\nAnd even though the battle was won, I feel like we lost it\nI spent too much energy on it, honestly I'm exhausted\nAnd I'm so caught in it I almost feel I'm the one who caused it\nThis ain't what I'm in hip-hop for, it's not why I got in it\n\nThat was never my object for someone to get killed\nWhy would I wanna destroy something I help build\nIt wasn't my intentions, my intentions was good\nI went through my whole career without ever mentionin'\n...\n\n"} #get_lyrics_by_tracks(fetched_artist_album_tracks)


    save_txt(fetched_artist_ids, 'artist_ids.txt')
    save_txt(fetched_artist_album_ids, 'album_ids.txt')
    save_txt(fetched_artist_album_tracks, 'album_tracks.txt')
    save_json(fetched_lyrics, 'lyrics.json')