import csv
import helper  # helper.py
import os


USER_FILE = './Testfiles/C1ku_users_extended.csv'
OUTPUT_DIR = './user_info/'

def read_user_file(uf):
    """
    Read csv-File
    TODO delete this - discuss in team
    """
    users = []

    # Open file - read only
    with open(uf, 'r') as f:
        # Read csv - Files and interprete tabs as new item
        reader = csv.reader(f, delimiter='\t')

        # Loop through reader
        for r in reader:
            # Append user-names only
            users.append(r[0])

        return users

def limit_user(all_users, maxAmountOfUsers):
    """
    Get a minimum of 5 users and 50 unique artists and return users

    :param all_users: a list of all users
    :param maxAmountOfUsers: how many users should get saved

    :return: returns limited_users
    """
    limited_users = []
    all_artist_names = []

    # Loop through list of all users
    for user in all_users:
        # Get artist-history from users via LastFM-API call
        top_artists = helper.api_user_call("gettopartists", user)
        artists = top_artists['topartists']['artist']

        # Create list of artist-names (all_artists) by iterating
        # through list with all artist information (artists)
        for artist in artists:
            artist_name = artist['name']

            all_artist_names.append(artist_name)

        # Fill list with users
        limited_users.append(user)

        # Delete duplicates from all_artist_names and save to new list all_artist_names
        all_artist_names = helper.get_unique_items(all_artist_names)

        # Limit amount of unique artists to a minimum of 50 and limit amount of users to minimum of 5
        # If true - stop for loop and return users (limited_users)
        if len(all_artist_names) >= 50 and len(limited_users) >= maxAmountOfUsers:
            return limited_users

# /limit_user

def lfm_prepare_history_string(track, user_name):
    """
    prepares a string for the txt file

    :param track: an object with tracks
    :param user: an user object with id and name

    :return: creates a string with metadata combined with tabs
    """

    for t in track:
        artist_id = t['artist']['mbid']
        artist_name = t['artist']['#text']
        track_id = t['mbid']
        track_name = t['name']
        timestamp = t['date']['uts']

    user_history = user_name + "\t" + artist_id + "\t" + artist_name + "\t" + track_id + "\t" + track_name + "\t" + timestamp
    return user_history.encode('utf-8')

# /lfm_prepare_history_string

def lfm_prepare_user_characteristics_string(user):
    """
    prepares a string for the txt file

    :param track: an object with tracks
    :param user: an user object with id and name

    :return: creates a string with metadata combined with tabs
    """

    user_name = user['name']
    user_country = user['country']
    user_age = user['age']
    user_gender = user['gender']
    user_playcount = user['playcount']
    user_playlists = user['playlists']
    user_registered = user['registered']['unixtime']

    user_characteristics = user_name + "\t" + user_country + "\t" + user_age + "\t" + user_gender + "\t" + user_playcount + "\t" + user_playlists + "\t" + user_registered
    return user_characteristics.encode('utf-8')

# /lfm_prepare_history_string

def lfm_save_history_of_users(users):
    """
    saves the history of users

    :param users: an array of users
    """

    content = ""

    print "Saving listening history . . ."

    # mkdir in py
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    for user in users:
        recent_tracks = helper.api_user_call("getrecenttracks", user)
        recent_track = recent_tracks['recenttracks']['track']
        listening_history = lfm_prepare_history_string(recent_track, user)
        content += listening_history + "\n"

    output_file = OUTPUT_DIR + '/listening_history.txt'

    text_file = open(output_file, 'w')
    text_file.write(content)
    text_file.close()

    print "Successfully created listening_history.txt"

    return

# /lfm_save_history_of_users

def lfm_save_user_characteristics(users):

    content = ""

    print "Saving user characteristics . . ."

    # mkdir in py
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    for user in users:
        user_info = helper.api_user_call("getinfo", user)
        user = user_info['user']
        user_string = lfm_prepare_user_characteristics_string(user)
        content += user_string + "\n"

    output_file = OUTPUT_DIR + '/users_characteristics.txt'

    text_file = open(output_file, 'w')
    text_file.write(content)
    text_file.close()

    print "Successfully created users_characteristics.txt"

    return

# Main
if __name__ == "__main__":
    users = read_user_file(USER_FILE)

    limited_users = limit_user(users, 10)

    lfm_save_history_of_users(limited_users)
    lfm_save_user_characteristics(limited_users)

# CTRL + Shift + E for Debug
# CTRL + Shift + R for Run
