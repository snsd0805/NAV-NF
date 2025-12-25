import json
import os
from reverie_utils import \
    get_obj2vps, \
    get_node_region, \
    get_connectivity, \
    get_same_room_vps

if not os.path.isdir('/data/New_REVERIE/outputs/only_front_path') :
    os.mkdir('/data/New_REVERIE/outputs/only_front_path')
with open('data/node_region.json') as fp:
    node_region = json.load(fp)


obj2vps = get_obj2vps()
connectivity, positions = get_connectivity()

for DATASET_TYPE in ['test']:
# for DATASET_TYPE in ['train', 'val_seen', 'val_unseen']:
    with open(f'/data/New_REVERIE/outputs/final_merge/REVERIE_{DATASET_TYPE}.json') as fp:
        data = json.load(fp)

    rooms = {}

    if DATASET_TYPE == 'val_unseen' or DATASET_TYPE == 'test':
        NEW_DATASET_TYPE = 'val_unseen'
    else:
        NEW_DATASET_TYPE = DATASET_TYPE
    with open(f'/data/New_REVERIE/outputs/after_gpt/REVERIE_{NEW_DATASET_TYPE}_with_landmark.json') as fp:
        tmp = json.load(fp)
        for i in tmp:
            assert i['id'] not in rooms, "Multi ids"
            rooms[i['id']] = i['relation']['target_area']
    with open(f'/data/New_REVERIE/outputs/after_gpt/REVERIE_{NEW_DATASET_TYPE}_no_landmark.json') as fp:
        tmp = json.load(fp)
        for i in tmp:
            assert i['id'] not in rooms, "Multi ids"
            rooms[i['id']] = i['relation']['target_area']

    new_data = []
    for i in data:
        new_i = i.copy()

        scan = i['scan']
        obj_id = i['objId']


        new_paths = []
        for path in i['path']:
            # original stop point
            original_stop_vp = path[-1]
            original_stop_vp_region = node_region[scan][original_stop_vp]


                    # get vps should explore if in not found case
            should_visit_rooms = set()
            should_visit_vps = obj2vps[f'{scan}_{obj_id}']
            for vp in should_visit_vps:
                print("Add Region", node_region[scan][vp], vp)
                should_visit_rooms.add(node_region[scan][vp])

            # get the vps in the same room
            same_room_vps = get_same_room_vps(connectivity, node_region, scan, should_visit_vps)

            # get the start vp in the target room
            room_start_vp = None
            front_path = []
            for vp in path:
                if vp not in same_room_vps:
                    front_path.append(vp)
                else:
                    room_start_vp = vp
                    break


            front_path.append(room_start_vp)
            new_paths.append(front_path)
            print(node_region[scan][front_path[-1]])
            assert node_region[scan][front_path[-1]] == node_region[scan][original_stop_vp], 'incorrect stop point in the target room'
            print(room_start_vp)
            print(path)
            print(front_path)
            print()
        new_i['path'] = new_paths
        # new_i['instructions'] = [ "Go to {}".format(rooms[i['id']]) ] * 2
        print("=== ")

        new_data.append(new_i)
    with open(f'/data/New_REVERIE/outputs/only_front_path/REVERIE_{DATASET_TYPE}.json', 'w') as fp:
        json.dump(new_data, fp)
    
