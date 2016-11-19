__author__ = 'jpeer'

###########
# IMPORTS #
###########
import os
import csv
import json
import glob

####################
# GLOBAL VARIABLES #
####################
RESULT_OUTPUT  = './output/results/'
METHOD_FOLDERS = [
    'RB_user',
    'RB_Artists',
    'PB',
    'HR_SEB_Lyrics',
    'HR_SEB_Wiki',
    'HR_RB_Wiki',
    'HR_RB_Lyrics',
    'HR_SCB_Wiki',
    'HR_SCB_Lyrics',
    'CB_Wiki',
    'CB_Musixmatch',
    'CF',
]

OUTPUT_DIR = ''

def generate_csv(method):
    """
    This will generate and save a csv file
    The folder in the METHOD_FOLDERS must be available in the RESULT_OUTPUT
    Also in this directory there must be json files with "neighbors", "avg_prec", "avg_rec" and "f1_score" in it
    The output dir for the csv is the same as the input directory

    :param method: the folder/method name
    """
    k_sorted       = {}
    r_sorted       = {}
    runned_methods = {}
    runned_methods[method] = []

    # csv headers
    csv_k_sorted_header = [
        ['Sorted by K values'],
        ['']
    ]

    csv_recommended_sorted_header = [
        ['Sorted by recommended artist values'],
        ['']
    ]

    sub_header = ['', 'MAP', 'MAR', 'F1 score']
    all_jsons  = sorted(glob.glob(OUTPUT_DIR + '/*.json'), key=os.path.getmtime)

    for one_json in all_jsons:
        with open(one_json) as data_file:
            data = json.load(data_file)

        runned_methods[method].append(data)

    for result_obj in runned_methods[method]:
        data_neighbors = [
            result_obj['neighbors'],
            result_obj['avg_prec'],
            result_obj['avg_rec'],
            result_obj['f1_score']
        ]

        data_recommended_artists = [
            result_obj['recommended_artists'],
            result_obj['avg_prec'],
            result_obj['avg_rec'],
            result_obj['f1_score']
        ]

        try:
            k_sorted[result_obj['neighbors']].append(data_recommended_artists)
        except:
            k_sorted[result_obj['neighbors']] = []
            k_sorted[result_obj['neighbors']].append(sub_header)
            k_sorted[result_obj['neighbors']].append(data_recommended_artists)

        try:
            r_sorted[result_obj['recommended_artists']].append(data_neighbors)
        except:
            r_sorted[result_obj['recommended_artists']] = []
            r_sorted[result_obj['recommended_artists']].append(sub_header)
            r_sorted[result_obj['recommended_artists']].append(data_neighbors)

    # sort items by K or R to compare values and get the best
    for key, value in r_sorted.items():
        # fill with meta info
        csv_recommended_sorted_header.append([''])
        csv_recommended_sorted_header.append([str(key) + ' recommended artists. '])

        for data in value:
            csv_recommended_sorted_header.append(data)

    for key, value in k_sorted.items():
        # fill with meta info
        csv_k_sorted_header.append([''])
        csv_k_sorted_header.append([str(key) + ' neighbors. '])

        for data in value:
            csv_k_sorted_header.append(data)

    b = open(OUTPUT_DIR + '/sorted_neighbors.csv', 'w')
    a = csv.writer(b)

    a.writerows(csv_k_sorted_header)
    b.close()

    b = open(OUTPUT_DIR + '/sorted_recommender.csv', 'w')
    a = csv.writer(b)

    a.writerows(csv_recommended_sorted_header)
    b.close()
# /generate_csv

# self executing
if __name__ == '__main__':
    for method in METHOD_FOLDERS:
        OUTPUT_DIR = RESULT_OUTPUT + method

        generate_csv(method)
