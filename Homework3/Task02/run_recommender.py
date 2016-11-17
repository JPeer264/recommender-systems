__author__ = 'jpeer'

import helper
import os
import json
import multiprocessing
from multiprocessing import Pool
from functools import partial

VERBOSE = True
OUTPUT_DIR = './output/results/'

def run_recommender(run_function, run_method, neighbors=[1, 2, 5, 10, 20, 50], recommender_artists=[10, 20, 30, 50, 100, 200, 300]):
    """
    TODO DESCRIPTION
    """
    k_sorted = {}
    r_sorted = {}
    data_to_append = {}
    all_files = {}
    output_filedir = OUTPUT_DIR + run_method + '/'
    all_files_path = output_filedir + 'all.json'

    helper.ensure_dir(output_filedir)

    for neighbor in neighbors:
        k_sorted['K' + str(neighbor)] = []

        for recommender_artist in recommender_artists:
            k_sorted['R' + str(recommender_artist)] = []
            # Define path to file that should be created
            file_path = output_filedir + 'K' + str(neighbor) + '_recommended' + str(recommender_artist) + '.json'
            # prepare for appending
            data_to_append = {'neighbors': neighbor, 'recommended_artists': recommender_artist}
            data = run_function(neighbor, recommender_artist)

            data_to_append.update(data)

            # write json file
            content = json.dumps(data_to_append, indent=4, sort_keys=True)
            f = open(file_path, 'w')
            f.write(content)
            f.close()
# /run_recommender

def run_multiprocessing(run_function, run_method, neighbors=[1, 2, 5, 10, 20, 50], recommender_artists=[10, 20, 30, 50, 100, 200, 300]):
    """
    TODO DESCRIPTION
    """
    processors = multiprocessing.cpu_count()
    print processors
    pool = Pool(processes=processors)


    func = partial(run_recommender, run_function, run_method, neighbors)
    print func
    pool.map(func, [recommender_artists])
    pool.close()
    pool.join

# /run_async
