# the cleaner the code - the happier the programmers
__author__ = 'jpeer'

###########
# IMPORTS #
###########
import helper # helper.py
import os
import json
import multiprocessing
from multiprocessing import Pool
from functools import partial
from thread import start_new_thread, allocate_lock

####################
# GLOBAL VARIABLES #
####################
VERBOSE        = True
OUTPUT_DIR     = './output/results/'
NUM_THREADS    = 0
THREAD_STARTED = False
LOCK           = allocate_lock()

def run_recommender(run_function, run_method, neighbors=[1, 2, 5, 10, 20, 50], recommender_artists=[10, 20, 30, 50, 100, 200, 300]):
    """
    runs automatically the run function, this funciton must be declared in the parameters
    it also saves automatically a json string with the parameters - the file name is as follows:
    K(K_number)_R(Recommended_artists).json

    :param run_function: the run fuction, from the single recommender
    :param run_method: the string which describes the current recommender
    :param neighbors: a list of different neighbors
    :param recommender_artists: a list of different artists to recommend
    """
    # for threading
    global NUM_THREADS, THREAD_STARTED, LOCK

    LOCK.acquire()
    NUM_THREADS += 1
    THREAD_STARTED = True
    LOCK.release()
    # for threading

    k_sorted       = {}
    r_sorted       = {}
    data_to_append = {}
    all_files      = {}
    output_filedir = OUTPUT_DIR + run_method + '/'
    all_files_path = output_filedir + 'all.json'

    helper.ensure_dir(output_filedir + 'recommended/')

    for neighbor in neighbors:
        k_sorted['K' + str(neighbor)] = []

        for recommender_artist in recommender_artists:
            k_sorted['R' + str(recommender_artist)] = []
            file_path       = output_filedir + 'K' + str(neighbor) + '_R' + str(recommender_artist) + '.json'
            file_path_reco  = output_filedir + 'recommended/' + 'K' + str(neighbor) + '_R' + str(recommender_artist) + '.json'
            data_to_append  = {'neighbors': neighbor, 'recommended_artists': recommender_artist}
            data            = run_function(neighbor, recommender_artist)
            recommended     = data['recommended']
            formated_recommended = {}

            # delete this
            # 1. not valid json
            # 2. not necessary for the specific files
            del data['recommended']

            data_to_append.update(data)

            for key, value in recommended.iteritems():
                # convert everything to strings
                # due to otherwise it is not a valid json
                formated_recommended[key] = {}

                if len(value) == 0:
                    continue

                for kf, fold_recommended in value.iteritems():
                    formated_recommended[key][kf] = {}
                    formated_recommended[key][kf]['recommended'] = {}
                    formated_recommended[key][kf]['order'] = []

                    for artist, ranking in fold_recommended.iteritems():
                        formated_recommended[key][kf]['recommended'][str(artist)] = str(ranking)
                        formated_recommended[key][kf]['order'].append(artist)

            print formated_recommended
            # write json file for hybrids
            content = json.dumps(formated_recommended, indent=4, sort_keys=True)
            f = open(file_path_reco, 'w')
            f.write(content)
            f.close()

            # write json file for csv
            content = json.dumps(data_to_append, indent=4, sort_keys=True)
            f = open(file_path, 'w')
            f.write(content)
            f.close()

    # for threading
    LOCK.acquire()
    NUM_THREADS -= 1
    LOCK.release()
    # for threading
# /run_recommender

def run_multithreading(run_function, run_method, neighbors=[1, 2, 5, 10, 20, 50], recommender_artists=[10, 20, 30, 50, 100, 200, 300]):
    """
    Starts a new thread and runs internally run_recommender
    / Speed boost / 80%-20% - depends on the processor and other opened programms
    Emojis aren't supported in python code and comments :(

    :param run_function: the run fuction, from the single recommender
    :param run_method: the string which describes the current recommender
    :param neighbors: a list of different neighbors
    :param recommender_artists: a list of different artists to recommend
    """
    global THREAD_STARTED, NUM_THREADS

    for index, value in enumerate(recommender_artists, start = 1):
        start_new_thread(run_recommender, (run_function, run_method, neighbors, [ value ],))

    while not THREAD_STARTED:
        pass
    while NUM_THREADS > 0:
        pass
# /run_async
