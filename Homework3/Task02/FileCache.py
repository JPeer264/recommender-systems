__author__ = 'jpeer'

import json
import helper # helper.py
import operator

class FileCache:
    """
    Caches a file for hybrid
    """
    def __init__(self, method, neighbors, recommended_artists):
        input_dir = './output/results/' + method + '/recommended/'

        with open(input_dir + 'K' + str(neighbors) + '_R' + str(recommended_artists) + '.json') as data_file:
            data = json.load(data_file)


        self.file = input_dir + 'K' + str(neighbors) + '_R' + str(recommended_artists) + '.json'
        self.data = data

    def read_for_hybrid(self, user, fold):
        data = self.data

        if len(data[str(user)]) == 0:
            return {}

        return_data = {}
        picked_data = data[str(user)][str(fold)]

        for item in picked_data['order']:
            return_data[item] = float(picked_data['recommended'][str(item)])

        return return_data
    # /read_for_hybrid

    def normalize(self):
        """
        If we forgot to normalize the data - but were too late to fix that in the recommender
        """
        normalized_data = {}

        for user, user_data in self.data.items():
            normalized_data[user] = {}

            for fold, fold_data in user_data.items():
                normalized_data[user][fold] = {}
                normalized_data[user][fold]['order'] = fold_data['order']

                sorted_fold_data = sorted(fold_data['recommended'].items(), key=operator.itemgetter(1), reverse=True)
                max_value = sorted_fold_data[0][1]

                new_dict_recommended_artists_idx = {}

                for i in sorted_fold_data:
                    new_dict_recommended_artists_idx[str(i[0])] = str(float(i[1]) / float(max_value))

                normalized_data[user][fold]['recommended'] = new_dict_recommended_artists_idx

        # write json file for hybrids
        content = json.dumps(normalized_data, indent=4, sort_keys=True)
        f = open(self.file, 'w')
        f.write(content)
        f.close()
    # /normalize
# /FileCache
