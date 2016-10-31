# Wikipedia fetcher to download "music context" data for an artist list.

# Load required modules
import os
import urllib
import csv
import re


# Parameters
WIKIPEDIA_URL = "http://en.wikipedia.org/wiki/"
WIKIPEDIA_URL_SS = "http://en.wikipedia.org/wiki/Special:Search/"

#ARTISTS_FILE = "./UAM_100u_artists.txt"          # text file containing Last.fm user names
#OUTPUT_DIRECTORY = "./crawls_wikipedia_100u"     # directory to write output to
ARTISTS_FILE = "../testfiles/UAM_artists.txt"          # text file containing Last.fm user names
OUTPUT_DIRECTORY = "./output/crawls_wikipedia"     # directory to write output to

USE_INDEX_IN_OUTPUT_FILE = True             # use [index].html as output file name (if set to False, the url-encoded artist name is used)
SKIP_EXISTING_FILES = True                  # skip files already retrieved


# Simple function to read content of a text file into a list
def read_file(fn):
    items = []                                      # list to hold artist names
    with open(fn, 'r') as f:
        reader = csv.reader(f, delimiter='\t')      # create reader
        reader.next()                     # in case we have a header
        for row in reader:
            items.append(row[0])
    return items


# Function to fetch a Wikipedia page, using artist name as query input
def fetch_wikipedia_page(query):
    # retrieve content from URL
    query_quoted = urllib.quote(query)

    url = WIKIPEDIA_URL_SS + query_quoted
    try:
        print "Retrieving data from " + url
        content = urllib.urlopen(url).read()
        regexp_1 = re.compile(r'h1.*firstHeading.*Search result.*/h1')  # check if it only retrieves the search page
        regexp_2 = re.compile(r'div.*mw-content-text.*may refer to:')   # check if it only retrieves the "may refer to" page
        if (regexp_1.search(content) is None) and (regexp_2.search(content) is None):
            return content
        else:
            return ""
    except IOError:                # return empty content in case some IO / socket error occurred
        return ""


# Main program
if __name__ == '__main__':

    # Create output directory if non-existent
    if not os.path.exists(OUTPUT_DIRECTORY):
        os.makedirs(OUTPUT_DIRECTORY)

    # Read artist list
    artists = read_file(ARTISTS_FILE)

    # Retrieve Wikipedia pages for all artists
    # Either use index as output file name
    if USE_INDEX_IN_OUTPUT_FILE:
        for i in range(0, len(artists)):
            html_fn = OUTPUT_DIRECTORY + "/" + str(i) + ".html"     # target file name
            # check if file already exists
            if os.path.exists(html_fn) & SKIP_EXISTING_FILES:       # if so and it should be skipped, skip the file
                print "File already fetched: " + html_fn
                continue
            # otherwise, fetch HTML content
            html_content = fetch_wikipedia_page(artists[i])

            # write to output file
            print "Storing content to " + html_fn
            with open(html_fn, 'w') as f:
                f.write(html_content)

            # if html_content != "":
            #     # write to output file
            #     print "Storing content to " + html_fn
            #     with open(html_fn, 'w') as f:
            #         f.write(html_content)
            # else:
            #     print "No data available. File skipped."
    else:
        # Or use url-encoded artist name
        for a in artists:
            html_fn = OUTPUT_DIRECTORY + "/" + urllib.quote(a) + ".html"     # target file name
            # check if file already exists
            if os.path.exists(html_fn) & SKIP_EXISTING_FILES:       # if so and it should be skipped, skip the file
                print "File already fetched: " + html_fn
                continue
            # otherwise, fetch HTML content
            html_content = fetch_wikipedia_page(a)
            # write to output file
            print "Storing content to " + html_fn
            with open(html_fn, 'w') as f:
                f.write(html_content)
