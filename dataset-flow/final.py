import json
import os

DATASET_TYPE = 'train'
test_dataset = []
for DATASET_TYPE in ['train', 'val_seen', 'val_unseen']:
    for TYPE in ['no', 'with']:
        with open(f'outputs/after_gpt/REVERIE_{DATASET_TYPE}_{TYPE}_landmark.json') as fp:
            data = json.load(fp)

        new_data = []
        for i in data:
            new_i = i.copy()
            new_i['instructions'].append(new_i['new_instruction'])
            new_i['path'] = [ new_i['path'], new_i['explore_path'] ]
            del new_i['new_instruction']
            del new_i['back_path']
            del new_i['explore_path']
            new_i['target_objects'] = [
                new_i['relation']['target_object'],
                new_i['new_target_object']
            ]
            del new_i['relation']
            new_i['found'].append(False)

            new_data.append(new_i)

        if not os.path.isdir('outputs/final'):
            os.mkdir('outputs/final')

        with open(f'outputs/final/REVERIE_{DATASET_TYPE}_{TYPE}_landmark.json', 'w') as fp:
            json.dump(new_data, fp)
