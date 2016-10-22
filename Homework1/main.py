import csv
import helper # helper.py

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
    TODO do not print before return
    Get a minimum of 5 users and 50 unique artists and return users

    :param all_users: a list of all users
    :param maxAmountOfUsers: how many users should get saved

    :return: returns limite
    """
    limited_users    = []
    all_artist_names = []

    # Loop through list of all users
    for user in all_users:
        # Get artist-history from users via LastFM-API call
        top_artists = helper.api_user_call("gettopartists", user)
        artists     = top_artists['topartists']['artist']

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
            print len(all_artist_names)
            print len(limited_users)

            return limited_users
# /limit_user

def lfm_prepare_history_string(track, user):
    """
    prepares a string for the txt file

    :param track: an object with tracks
    :param user: an user object with id and name

    :return: creates a string with metadata combined with tabs
    """
    user_id   = "USER_ID"
    user_name = "USER_NAME"

    for t in track:
        artist_id   = t['artist']['mbid']
        artist_name = t['artist']['#text']
        track_id    = t['mbid']
        track_name  = t['name']
        timestamp   = t['date']['uts']

    return user_id + "\t" + user_name + "\t" + artist_id + "\t" + artist_name + "\t" + track_id + "\t" + track_name + "\t" + timestamp
# /lfm_prepare_history_string

def lfm_save_history_of_users(users):
    """
    TODO no printing
    TODO save into file
    saves the history of users

    :param users: an array of users
    """
    for user in users:
        recent_tracks = helper.api_user_call("getrecenttracks", user)
        one_track     = recent_tracks['recenttracks']['track']
        print lfm_prepare_history_string(one_track, user)

    return
# /lfm_save_history_of_users

# Main
if __name__ == "__main__":
    users = read_user_file(USER_FILE)

    limited_users = limit_user(users, 5)
    lfm_save_history_of_users(limited_users)

# CTRL + Shift + E for Debug
# CTRL + Shift + R for Run
