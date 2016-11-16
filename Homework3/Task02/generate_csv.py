# Implementation of a simple evaluation framework for recommender systems algorithms.
# This script further implements different baseline recommenders: collaborative filtering,
# content-based recommender, random recommendation, and simple hybrid methods.
# It also implements a score-based fusion technique for hybrid recommendation.
__author__ = 'mms'

# Load required modules
import json
import csv
import glob

# Parameters
RESULT_OUTPUT = './output/results/'
METHOD_FOLDER = 'CF'

OUTPUT_DIR = RESULT_OUTPUT + METHOD_FOLDER


# Main program, for experimentation.
if __name__ == '__main__':
    runned_methods = {}
    runned_methods[METHOD_FOLDER] = []

    k_sorted = {}
    r_sorted = {}

    # data
    neighbors = [ 1, 2, 3, 5, 10, 20, 50 ]
    recommender_artists = [ 10, 20, 30, 50, 100, 200, 300 ]

    for neighbor in neighbors:
        k_sorted['K' + str(neighbor)] = []

        for recommender_artist in recommender_artists:
            r_sorted['R' + str(recommender_artist)] = []

    # write csv
    csv_k_sorted_header = [
        ['Sorted by K values'],
        ['']
    ]

    csv_recommended_sorted_header = [
        ['Sorted by recommended artist values'],
        ['']
    ]

    all_jsons = glob.glob(OUTPUT_DIR + '/*.json')

    for one_json in all_jsons:
        with open(one_json) as data_file:
            data = json.load(data_file)

        runned_methods[METHOD_FOLDER].append(data)

    for result_obj in runned_methods[METHOD_FOLDER]:
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

        k_sorted['K' + str(result_obj['neighbors'])].append(data_recommended_artists)
        r_sorted['R' + str(result_obj['recommended_artists'])].append(data_neighbors)


    for key, value in r_sorted.items():
        if key[0] == 'R':
            # fill with meta info
            csv_recommended_sorted_header.append([''])
            csv_recommended_sorted_header.append([str(key) + ' recommended artists. '])

            for data in value:
                csv_recommended_sorted_header.append(data)

    for key, value in k_sorted.items():
        if key[0] == 'K':
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
