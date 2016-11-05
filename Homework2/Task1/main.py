import helper # helper.py
import csv
import Musixmatch_fetcher as mf

TESTFILES        = "../testfiles/"
ARTISTS          = TESTFILES + "LFM1b_artists.txt"
USERS            = TESTFILES + "LFM1b_users.txt"
C1KU_ARTISTS_IDX = TESTFILES + "C1ku_idx_artists.txt"
C1KU_USERS_IDX   = TESTFILES + "C1ku_idx_users.txt"
OUTPUT_DIR       = "./output/"
CHOSEN_ARTISTS   = OUTPUT_DIR + "artists.txt"
CHOSEN_USERS     = OUTPUT_DIR + "users.txt"

def save_lfmb_c1ku_combined_file(c1ku_file, lfmb1_file, output_file, header_string):
    helper.log_highlight('save ' + output_file)

    LFM1b_file = mf.read_txt(lfmb1_file)
    sorted_string = header_string + "\n"

    with open(c1ku_file, 'r') as f:
        reader  = csv.reader(f, delimiter='\t')      # create reader
        headers = reader.next()                     # skip header

        for index, row in enumerate(reader, start = 1):
            the_id = row[0]

            sorted_string += LFM1b_file[the_id] + "\n"

    helper.ensure_dir(OUTPUT_DIR)

    text_file = open(output_file, 'w')

    text_file.write(sorted_string)
    text_file.close()
# /save_lfmb_c1ku_combined_file

# Main
if __name__ == "__main__":
    # first get all artists from the LFM1b_artists and
    # just save artists which are saved in C1ku_idx_artists.txt
    save_lfmb_c1ku_combined_file(C1KU_ARTISTS_IDX, ARTISTS, CHOSEN_ARTISTS, 'artists')
    save_lfmb_c1ku_combined_file(C1KU_USERS_IDX, USERS, CHOSEN_USERS, 'users')

    helper.log_highlight("Done")