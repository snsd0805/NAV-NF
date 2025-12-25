import json
import os

for DATASET_TYPE in ['val_seen', 'val_unseen', 'train']:
    print(DATASET_TYPE)

    with open(f'outputs/after_landmark/REVERIE_{DATASET_TYPE}.json') as fp:
        data = json.load(fp)

    has_landmark_data, no_landmark_data = [], []
    for i in data:
        landmark = i['relation']['landmark']
        if 'None' in landmark or 'none' in landmark:
            no_landmark_data.append(i)
        else:
            has_landmark_data.append(i)

    print('has landmark: ', len(has_landmark_data))
    print('no landmark: ', len(no_landmark_data))

    # save
    if not os.path.isdir('./outputs/split/'):
        os.mkdir('./outputs/split')
    with open(f'outputs/split/REVERIE_{DATASET_TYPE}_has_landmark.json', 'w') as fp:
        json.dump(has_landmark_data, fp)
    with open(f'outputs/split/REVERIE_{DATASET_TYPE}_no_landmark.json', 'w') as fp:
        json.dump(no_landmark_data, fp)
