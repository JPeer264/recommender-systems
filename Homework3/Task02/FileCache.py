import json
import helper # helper.py

class FileCache:
    """
    Caches a file for hybrid
    """
    def __init__(self, method, neighbors, recommended_artists):
        input_dir = './output/results/' + method + '/recommended/'

        with open(input_dir + 'K' + str(neighbors) + '_R' + str(recommended_artists) + '.json') as data_file:
            data = json.load(data_file)

        self.data = data

    def read_for_hybrid(self, user, fold):
        data = self.data

        print data

        if len(data[str(user)]) == 0:
            return {}

        return_data = {}
        picked_data = data[str(user)][str(fold)]

        for item in picked_data['order']:
            return_data[item] = float(picked_data['recommended'][str(item)])

        return return_data
    # /read_for_hybrid
# /FileCache
