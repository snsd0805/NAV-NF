import json
import os
import random

if not os.path.isdir('./outputs/for_navgpt'):
    os.mkdir('./outputs/for_navgpt')

with open('./outputs/final_merge/REVERIE_val_unseen.json') as fp:
    annotations = json.load(fp)
with open('/data/Matterport3DSimulator-duet-front-path/VLN-DUET/datasets/REVERIE/exprs_map/finetune/dagger-vitbase-new-reverie-only-front-path-seed.0/preds/submit_val_unseen_dynamic.json') as fp:
# with open('/data/Matterport3DSimulator-duet/VLN-DUET/datasets/REVERIE/exprs_map/finetune/dagger-vitbase-new-reverie-only-front-path-seed.0/preds/submit_val_unseen_dynamic.json') as fp:
# with open('/data/Matterport3DSimulator-duet/VLN-DUET/datasets/REVERIE/exprs_map/finetune/dagger-vitbase-new-reverie-only-front-path-seed.0/preds/detail_val_unseen_dynamic.json') as fp:
    preds = json.load(fp)

starts = {}
fronts = {}
for pred in preds:
    instr_id = pred['instr_id']
    path = [ item for sublist in pred['trajectory'] for item in sublist  ]
    assert instr_id not in starts, "key error"
    starts[instr_id] = path[-1]
    fronts[instr_id] = path


new_data = {0: [], 1: []}
loss = 0
with open('./data/node_region.json') as fp:
    node_region = json.load(fp)
for anno in annotations:
    for i in range(2):
        new_i = {}
        new_i['distance'] = anno['distance']
        new_i['ix'] = anno['ix']
        new_i['scan'] = anno['scan']
        new_i['id'] = anno['id']
        new_i['instructions_l'] = anno['instructions_l']
        new_i['path_id'] = anno['path_id']
        new_i['objId'] = anno['objId']
        new_i['path'] = anno['path'][i]
        new_i['heading'] = anno['heading']
        new_i['instruction'] = anno['instructions'][i]
        new_i['new_reverie_id'] = f"{anno['id']}_{i}"
        new_i['gt_found'] = anno['found'][i]
        new_i['fronts'] = fronts[new_i['new_reverie_id']]
        tmp_found = True if i == 0 else False
        assert new_i['gt_found'] == tmp_found, 'found error'
        new_i['target'] = anno['target_objects'][i]
        new_i['start'] = starts[new_i['new_reverie_id']]
        new_i['clip_target'] = new_i['target'].replace('the ', '').replace('a ', '').replace('an ', '')

        new_data[i].append(new_i)
        
print(f"loss {loss} instructions")
print(len(new_data[0]))
print(len(new_data[1]))

random.shuffle(new_data[0])
random.shuffle(new_data[1])
# final_data = new_data[0][:50] + new_data[1][:50]
SIZE = 500
final_data = new_data[0][:int(SIZE/2)] + new_data[1][:int(SIZE/2)]
final_data = new_data[0] + new_data[1]


counter = 0
for i in final_data:
    scan = i['scan']
    gt_region = node_region[scan][i['path'][-1]]
    pred_region = node_region[scan][i['fronts'][-1]]
    if gt_region == pred_region:
        counter += 1
print(f'counter: {counter} / {len(final_data)}')

random.shuffle(final_data)
print(len(final_data))
with open('./outputs/for_navgpt/REVERIE_val_unseen_instr.json', 'w') as fp:
    json.dump(final_data, fp)
