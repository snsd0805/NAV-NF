import os
import json
import random

if not os.path.isdir('outputs/final_merge'):
    os.mkdir('outputs/final_merge')

TEST_ENV = ['8WUmhLawc2A', 'XcA2TqTSSAj', 'r1Q1Z4BcV1o', 'VFuaQ6m2Qom', '82sE5b5pLXE', 'kEZ7cmS4wCh', 'qoiz87JEwZ2', 'Vvot9Ly1tCj', 'ULsKaCPVFJR', 'JmbYfDe2QKZ']
TEST_ENV += ['TbHJrupSAjP', 'zsNo4HB9uLZ']



test_with, test_no = [], []
for DATASET_TYPE in ['train', 'val_seen', 'val_unseen']:

    data = []
    test_data = []

    l = 0

    no_tmp_data = []
    with open(f'outputs/final/REVERIE_{DATASET_TYPE}_no_landmark.json') as fp:
        tmp_data = json.load(fp)
        for i in tmp_data:
            if i['scan'] not in TEST_ENV:
                no_tmp_data.append(i)
            else:
                test_no.append(i)

        random.shuffle(no_tmp_data)
        data += no_tmp_data
        l = len(no_tmp_data)

    with_tmp_data = []
    with open(f'outputs/final/REVERIE_{DATASET_TYPE}_with_landmark.json') as fp:
        tmp_data = json.load(fp)
        for i in tmp_data:
            if i['scan'] not in TEST_ENV:
                with_tmp_data.append(i)
            else:
                test_with.append(i)

        random.shuffle(with_tmp_data)
        data += with_tmp_data

    
    random.shuffle(data)
    print(DATASET_TYPE , len(data))

    with open(f'outputs/final_merge/REVERIE_{DATASET_TYPE}.json', 'w') as fp:
        json.dump(data, fp)
    
l = len(test_no)
random.shuffle(test_no)
random.shuffle(test_with)
test_data = test_no + test_with

with open(f'outputs/final_merge/REVERIE_test.json', 'w') as fp:
    json.dump(test_data, fp)
print(len(test_data))
