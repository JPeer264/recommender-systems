__authors__ = [
    'Aichbauer Lukas',
    'Leitner Bianca',
    'Stoecklmair Jan Peer',
    'Taferner Mario'
]

import os
import csv
import glob
import json

def read_csv(file):
    data = []
    with open(file, 'r') as f:
        reader  = csv.reader(f, delimiter='\t')
        headers = reader.next()

        for row in reader:
            item = row[0]
            data.append(item)

    return data
# /read_csv

def get_unique_items(iterable):
    """
    Deletes duplicates in array
    https://stackoverflow.com/questions/32664180/why-does-removing-duplicates-from-a-list-produce-none-none-output

    :param iterable: an array to remove duplicates

    :return: an array with no duplicates
    """
    seen = set()
    result = []

    for item in iterable:
        if item not in seen:
            seen.add(item)
            result.append(item)

    return result
# /get_unique_items

def log(text):
    if VERBOSE:
        print text
# /log

def log_highlight(text):
    """
    #################
    ## Highlightes ## any given text and print it
    #################

    :param text:
    """
    hashes = ""

    for i in text:
        hashes += "#"

    print ""
    print "###" + hashes + "###"
    print "## " + text + " ##"
    print "###" + hashes + "###"
    print ""

    return
# /log_highlight

def number_to_text(number):
    """
    change 1 to first
    change 2 to second
    ...

    :param number: an integer
    :return: a string
    """
    return {
        1: 'first   ',
        2: 'second  ',
        3: 'third   ',
        4: 'fourth  ',
        5: 'fifth   ',
        6: 'sixth   ',
        7: 'seventh ',
        8: 'eigth   ',
        9: 'nineth  ',
        10: 'tenth   ',
    }.get(number, '')
# /number_to_text

def ensure_dir(directory):
    """
    Ensures that the directory exists. If the directory structure does not exist, it is created.

    :param directory: any path as string
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
# /ensure_dir

def read_for_hybrid(method, neighbors, recommended_artists, user, fold):
    input_dir = './output/results/' + method + '/recommended/'

    with open(input_dir + 'K' + str(neighbors) + '_R' + str(recommended_artists) + '.json') as data_file:
        data = json.load(data_file)

    if len(data[str(user)]) == 0:
        return {}

    return_data = {}
    picked_data = data[str(user)][str(fold)]

    for item in picked_data['order']:
        return_data[item] = float(picked_data['recommended'][str(item)])

    return return_data
# /read_for_hybrid
