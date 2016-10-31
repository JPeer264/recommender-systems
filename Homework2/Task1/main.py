import helper # helper.py
import csv

TESTFILES        = "../testfiles/"
ARTISTS          = TESTFILES + "LFM1b_artists.txt"
C1KU_ARTISTS_IDX = TESTFILES + "C1ku_idx_artists.txt"
OUTPUT_DIR       = "./output/"
CHOSEN_ARTISTS   = OUTPUT_DIR + "artists.txt"

def save_artists_into_file():
    """
    calculates and sorts the playcounts of each user and its artist

    :return: a sorted array of all artists with given playcounts
    """
    artists = helper.read_csv(C1KU_ARTISTS_IDX)
    sorted_artists = ""

    with open(ARTISTS, 'r') as f:
        reader  = csv.reader(f, delimiter='\t')      # create reader
        headers = reader.next()                     # skip header

        for index, row in enumerate(reader, start = 1):
            artist_id = row[0]
            artist    = row[1]

            if artist_id in artists:
                sorted_artists += artist + "\n"

    helper.ensure_dir(OUTPUT_DIR)

    text_file = open(CHOSEN_ARTISTS, 'w')

    text_file.write(sorted_artists)
    text_file.close()
# /save_artists_into_file

# Main
if __name__ == "__main__":
    # first get all artists from the LFM1b_artists and
    # just save artists which are saved in C1ku_idx_artists.txt
    #save_artists_into_file()

    helper.log_highlight("Done")