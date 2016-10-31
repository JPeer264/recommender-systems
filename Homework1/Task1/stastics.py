import csv
import helper  # helper.py

USER_FILE = "./output/user_info/limited_user_list.csv"
LE_FILE = "./output/user_info/listening_history.txt"

def le_total():
    le = 0

    with open(LE_FILE, 'r') as f:
        reader  = csv.reader(f, delimiter='\t')      # create reader
        headers = reader.next()                     # skip header

        for row in reader:
            le += 1

    return le
# /le_total

def unique_tracks_total():
    unique_tracks = 0
    users_artist_count = {}

    with open(LE_FILE, 'r') as f:
        reader  = csv.reader(f, delimiter='\t')      # create reader
        headers = reader.next()                     # skip header

        for row in reader:
            user  = row[0]
            track = row[4]

            try:
                users_artist_count[user].append(track)
            except:
                users_artist_count[user] = []
                users_artist_count[user].append(track)


    for user in users_artist_count:
        unique_tracks += len(helper.get_unique_items(users_artist_count[user]))

    return unique_tracks
# /unique_tracks_total

def unique_artists_total():
    all_artists = []

    with open(LE_FILE, 'r') as f:
        reader  = csv.reader(f, delimiter='\t')      # create reader
        headers = reader.next()                     # skip header

        for row in reader:
            artist = row[2]

            all_artists.append(artist)

    return len(helper.get_unique_items(all_artists))
# /unique_artists_total

# Main
if __name__ == "__main__":
    helper.log_highlight('Users In Total')
    print len(helper.read_csv(USER_FILE))
    print ''
    helper.log_highlight('Unique Tracks In Total')
    print unique_tracks_total()
    print ''
    helper.log_highlight('Unique Artists In Total')
    print unique_artists_total()
    print ''
    helper.log_highlight('Listening Events In Total')
    print le_total()
    print ''

