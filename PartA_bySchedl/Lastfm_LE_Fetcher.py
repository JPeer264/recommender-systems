# Simple Last.fm API crawler to download listening events.
__author__ = 'mms'

# Load required modules
import os
import urllib
import csv
import json


# Parameters
LASTFM_API_KEY = "4cb074e4b8ec4ee9ad3eb37d6f7eb240"
LASTFM_API_URL = "http://ws.audioscrobbler.com/2.0/"
LASTFM_OUTPUT_FORMAT = "json"

MAX_PAGES = 5                   # maximum number of pages per user
MAX_EVENTS_PER_PAGE = 200       # maximum number of listening events to retrieve per page

USERS_FILE = "./lastfm_users_5.csv"       # text file containing Last.fm user names
OUTPUT_DIRECTORY = "./lastfm_crawls_5"    # directory to write output to
LE_FILE = "LE_5.txt"                      # aggregated listening events


# Simple function to read content of a text file into a list
def read_users(users_file):
    users = []                                      # list to hold user names
    with open(users_file, 'r') as f:
        reader = csv.reader(f, delimiter='\t')      # create reader
#        headers = reader.next()                    # in case we have a header
        for row in reader:
            users.append(row[0])
    return users


# Function to call Last.fm API: Users.getRecentTrack
def lastfm_api_call_getLEs(user, output_dir):
    content_merged = []        # empty list

    # ensure that output directory structure exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # retrieve content from URL
    query_quoted = urllib.quote(user)

    # loop over number of pages to retrieve
    for p in range(0, MAX_PAGES):
        url = LASTFM_API_URL + "?method=user.getrecenttracks&user=" + query_quoted + \
              "&format=" + LASTFM_OUTPUT_FORMAT + \
              "&api_key=" + LASTFM_API_KEY + \
              "&limit=" + str(MAX_EVENTS_PER_PAGE) + \
              "&page=" + str(p+1)

        print "Retrieving page #" + str(p+1)
        content = urllib.urlopen(url).read()

        content_merged.append(content)      # add retrieved content of current page to merged content variable

        # write content to local file
        output_file = output_dir + "/" + user + "_" + str(p) + "." + LASTFM_OUTPUT_FORMAT
        file_out = open(output_file, 'w')
        file_out.write(content)
        file_out.close()

    return content_merged          # return all content retrieved


# Main program
if __name__ == '__main__':

    # Create output directory if non-existent
    if not os.path.exists(OUTPUT_DIRECTORY):
        os.makedirs(OUTPUT_DIRECTORY)

    # Read users from provided file
    users = read_users(USERS_FILE)
    #print users

    # Create list to hold listening events
    LEs = []

    # For all users, retrieve user info
    for u in range(0, len(users)):
        print 'Fetching user data for user #' + str(u) + ': ' + users[u] + ' ...'
        content = lastfm_api_call_getLEs(users[u], OUTPUT_DIRECTORY + "/listening_events/")

        # parse retrieved JSON content
        # NB: all content is returned in unicode format
        try:
            # for all retrieved JSON pages of current user
            for page in range(0, len(content)):
                listening_events = json.loads(content[page])

                # get number of listening events in current JSON
                no_items = len(listening_events["recenttracks"]["track"])

                # read artist and track names for each
                for item in range(0, no_items):
                    artist = listening_events["recenttracks"]["track"][item]["artist"]["#text"]
                    track = listening_events["recenttracks"]["track"][item]["name"]
                    time = listening_events["recenttracks"]["track"][item]["date"]["uts"]
#                    print users[u], artist, track, time

                    # Add listening event to aggregated list of LEs
                    LEs.append([users[u], artist.encode('utf8'), track.encode('utf8'), str(time)])


        except KeyError:                    # JSON tag not found
            print "JSON tag not found!"
            continue


    # Write retrieved listening events to text file
    with open(LE_FILE, 'w') as outfile:             # "a" to append
        outfile.write('user\tartist\ttrack\ttime\n')
        for le in LEs:          # for all listening events
            outfile.write(le[0] + "\t" + le[1] + "\t" + le[2] + "\t" + le[3] + "\n")
