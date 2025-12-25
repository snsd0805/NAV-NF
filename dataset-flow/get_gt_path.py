import json
import os
from reverie_utils import \
    get_obj2vps, \
    get_node_region, \
    get_connectivity, \
    get_same_room_vps
from itertools import permutations


# key = scan_objid
# ans[key] = [vps]
obj2vps = get_obj2vps()

# node_region[scan][vp] = id(int)
node_region = get_node_region()

# ans[from] = [available_access_vps]
# positions[vp] = [x, y, z]
connectivity, positions = get_connectivity()

for DATASET_TYPE in ['train', 'val_seen', 'val_unseen']:
    print(DATASET_TYPE)
    with open(f'outputs/split/REVERIE_{DATASET_TYPE}_has_landmark.json') as fp:
        data = json.load(fp)

    def get_distance(pose1, pose2):
        return ((pose1[0]-pose2[0])**2 + (pose1[1]-pose2[1])**2 + (pose1[2]-pose2[2])**2) ** 0.5

    def find_path(medium, start, stop):
        print(start, stop)
        print(medium[start])
        m_vp = medium[start][stop]
        if m_vp == -1: 
            return []

        front_path = find_path(medium, start, m_vp)
        back_path = find_path(medium, m_vp, stop)
        return front_path + [m_vp] + back_path

    counter = 0
    new_data = []
    LOSS_COUNTER = 0
    for i in data:
        counter += 1
        new_i = i.copy()

        scan = i['scan']
        obj_id = i['objId']
        should_visit_vps = obj2vps[f'{scan}_{obj_id}']

        # get the start vp in the target room
        same_room_vps = get_same_room_vps(connectivity, node_region, scan, should_visit_vps)
        print(same_room_vps)

        '''
        if i['id'] == '6753_768':
            print("HERE===")
            print(same_room_vps)

            room_start_vp = None
            front_path = []
            for vp in i['path']:
                print(node_region[i['scan']][vp])
                if vp not in same_room_vps:
                    front_path.append(vp)
                else:
                    room_start_vp = vp
                    break
        '''

        else:
            room_start_vp = None
            front_path = []
            for vp in i['path']:
                if vp not in same_room_vps:
                    front_path.append(vp)
                else:
                    room_start_vp = vp
                    break

        # print(len(same_room_vps), len(should_visit_vps))
        # print(room_start_vp, same_room_vps)
        # get vps should explore if in not found case


        # assume that all vps can see target objects are in the same room.


        # get shortest distance matrix
        #   reset the distance matrix
        distance_matrix = {}
        medium_vp = {}
        for i_vp in same_room_vps:
            if i_vp not in distance_matrix:
                distance_matrix[i_vp] = {}
                medium_vp[i_vp] = {}
            for j_vp in same_room_vps:
                if j_vp not in distance_matrix:
                    distance_matrix[j_vp] = {}
                    medium_vp[j_vp] = {}
                medium_vp[i_vp][j_vp] = -1
                if i_vp == j_vp:
                    distance_matrix[i_vp][j_vp] = 0
                else:
                    distance_matrix[i_vp][j_vp] = 99999
                    distance_matrix[j_vp][i_vp] = 99999

        #   set up the neighbor's distance
        for i_vp in same_room_vps:
            for j_vp in same_room_vps:
                assert (j_vp in connectivity[i_vp]) == (i_vp in connectivity[j_vp]), "Something wrong in the connectivity data."
                if j_vp in connectivity[i_vp]:      # if they connected.
                    distance = get_distance(positions[i_vp], positions[j_vp])
                    distance_matrix[i_vp][j_vp] = distance
                    distance_matrix[j_vp][i_vp] = distance

    #  print(i_vp, j_vp, distance)

        # floyed warshall
        print("Same room vps: ", same_room_vps)
        for k_vp in same_room_vps:
            for i_vp in same_room_vps:
                for j_vp in same_room_vps:
                    if distance_matrix[i_vp][j_vp] > (distance_matrix[i_vp][k_vp] + distance_matrix[k_vp][j_vp]):
                        distance_matrix[i_vp][j_vp] = (distance_matrix[i_vp][k_vp] + distance_matrix[k_vp][j_vp])
                        medium_vp[i_vp][j_vp] = k_vp

        print(f"should_visit_vps: {should_visit_vps}")
        perms = permutations(should_visit_vps)
        min_distance, min_perm = 99999, []
        print(i)

        for perm in perms:
            print(perm)
            prev_vp = room_start_vp
            all_distance = 0
            if perm[0] != room_start_vp:
                all_path = [room_start_vp]
            else:
                all_path = []
            for vp in perm:
                print(f"    {prev_vp} -> {vp}")
                print(node_region[scan][prev_vp], node_region[scan][vp])
                print("LINK:", connectivity[prev_vp])
                print("ROOM: ", [ node_region[scan][a] for a in connectivity[prev_vp] ])
                mid_path = (find_path(medium_vp, prev_vp, vp) + [vp])
                print(f"        {mid_path}")
                all_path += mid_path
                all_distance += distance_matrix[prev_vp][vp]
                prev_vp = vp
            print(all_path)
            print(all_distance)
            print()

            if all_distance < min_distance:
                min_distance = all_distance
                min_perm = all_path

        if len(front_path) > 0 and len(min_perm) > 1 and node_region[i['scan']][i['path'][0]] != node_region[i['scan']][i['path'][-1]] :
            final_path = front_path + min_perm
            print(f"BEST PATH: {min_perm}")
            print(f"all: {final_path}")
            print(f"front: {front_path}")
            new_i['explore_path'] = final_path
            new_i['back_path'] = min_perm
            new_data.append(new_i)
        else:
            LOSS_COUNTER += 1
        print("="*20)

    print(f"LOSS:{LOSS_COUNTER}")

    print(len(new_data))

    if not os.path.isdir('outputs/gt_path/'):
        os.mkdir('outputs/gt_path')
    with open(f'outputs/gt_path/REVERIE_{DATASET_TYPE}_with_landmark.json', 'w') as fp:
        json.dump(new_data, fp)


