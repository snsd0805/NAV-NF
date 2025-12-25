import json
import os
from reverie_utils import \
    get_obj2vps, \
    get_node_region, \
    get_connectivity, \
    get_objs_same_room, \
    get_same_room_vps, \
    get_vp2objs
from itertools import permutations


# key = scan_objid
# ans[key] = [vps]
obj2vps = get_obj2vps()

# node_region[scan][vp] = id(int)
node_region = get_node_region()

# ans[from] = [available_access_vps]
# positions[vp] = [x, y, z]
connectivity, positions = get_connectivity()

# ans[vp] = [objid1, objid2.......]
vp2objs = get_vp2objs()

DATASET_TYPE = 'val_unseen'
for DATASET_TYPE in ['train', 'val_seen', 'val_unseen']:
    with open(f'outputs/split/REVERIE_{DATASET_TYPE}_no_landmark.json') as fp:
        data = json.load(fp)

    def get_distance(pose1, pose2):
        return ((pose1[0]-pose2[0])**2 + (pose1[1]-pose2[1])**2 + (pose1[2]-pose2[2])**2) ** 0.5

    def find_path(medium, start, stop):
        m_vp = medium[start][stop]
        if m_vp == -1: 
            return []

        front_path = find_path(medium, start, m_vp)
        back_path = find_path(medium, m_vp, stop)
        return front_path + [m_vp] + back_path

    new_data = []
    max_vps_in_same_room = 0
    for i in data:
        new_i = i.copy()

        scan = i['scan']
        obj_id = i['objId']


        should_visit_vps = obj2vps[f'{scan}_{obj_id}']

        # get the start vp in the target room
        same_room_vps = get_same_room_vps(connectivity, node_region, scan, should_visit_vps)
        print(same_room_vps)


        original_path = i['path']
        room_start_vp = None
        front_path = []
        for vp in i['path']:
            if vp not in same_room_vps:
                front_path.append(vp)
            else:
                room_start_vp = vp
                break

        # get the vps in the same room
        # print(len(same_room_vps), len(should_visit_vps))
        # print(room_start_vp, same_room_vps)


            

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

        # floyed warshall
        for k_vp in same_room_vps:
            for i_vp in same_room_vps:
                for j_vp in same_room_vps:
                    if distance_matrix[i_vp][j_vp] > (distance_matrix[i_vp][k_vp] + distance_matrix[k_vp][j_vp]):
                        distance_matrix[i_vp][j_vp] = (distance_matrix[i_vp][k_vp] + distance_matrix[k_vp][j_vp])
                        medium_vp[i_vp][j_vp] = k_vp

        all_objs = get_objs_same_room(vp2objs, same_room_vps)
        # print('    ', len(all_objs))

        
        def get_neighbors(vps, i):
            ans = []
            for j in vps:
                if i != j and j in connectivity[i]:
                    ans.append(j)
            return ans


        back_path = []
        push = False
        for vp in original_path:
            if vp in same_room_vps:
                push = True
            if push:
                back_path.append(vp)
        assert original_path == (front_path+back_path), "Path error"


        visited_obj = set()
        for vp in back_path:
            visited_obj.update(vp2objs[vp])
        room_start_vp = back_path[-1]

        neighbors = get_neighbors(same_room_vps, room_start_vp)
        explore_percent = float( len(visited_obj) / len(all_objs) )
        print(f"    step to {room_start_vp} -> {explore_percent} %")
        # while float(visited_obj_len/len(all_objs)) < 85:

        explore_steps = 0
        while neighbors != []:
            max_obj_len, max_vp = 0, None
            for neighbor in neighbors:
                unvisit_objs = set()
                for obj_id in vp2objs[neighbor]:
                    if obj_id not in visited_obj:
                        unvisit_objs.add(obj_id)
                if len(unvisit_objs) > max_obj_len:
                    max_obj_len = len(unvisit_objs)
                    max_vp = neighbor

            if max_vp != None:

                back_path.append(max_vp)
                explore_steps += 1
                # print(max_vp)
                neighbors = get_neighbors(same_room_vps, max_vp)
                visited_obj.update(vp2objs[max_vp])

                explore_percent = float( len(visited_obj) / len(all_objs) )
                print(f"    step to {max_vp} -> {explore_percent} %")
            else:
                neighbors = []


        explore_percent = float( len(visited_obj) / len(all_objs) )

        while explore_percent < 0.8:
            max_obj_len, max_vp = 0, None
            for vp in same_room_vps:
                unvisit_objs = set()
                for obj_id in vp2objs[vp]:
                    if obj_id not in visited_obj:
                        unvisit_objs.add(obj_id)
                if len(unvisit_objs) > max_obj_len and distance_matrix[back_path[-1]][vp]:
                    max_obj_len = len(unvisit_objs)
                    max_vp = vp

            if max_vp != None:
                print(f"Move to {max_vp}")

                mid_path = find_path(medium_vp, back_path[-1], max_vp) + [max_vp]

                explore_steps += 1
                for vp in mid_path:
                    back_path.append(vp)
                    visited_obj.update(vp2objs[vp])

                explore_percent = float( len(visited_obj) / len(all_objs) )
                print(f"    step to {max_vp} -> {explore_percent} %")
            else:
                raise ValueError("Some error")

        print(room_start_vp)
        print("Front:", front_path)
        print("Back:", back_path)
        print("New:", front_path + back_path)
        print("Original:", original_path)

        print(explore_percent, '%')
        print(f"explore {len(back_path)} nodes")
        print("="*20)

        
        if explore_steps > 1 and len(front_path) > 0:
            final_path = front_path + back_path
            new_i['explore_path'] = final_path
            new_i['explore_percent'] = explore_percent * 100
            new_i['back_path'] = back_path
            new_data.append(new_i)


    if not os.path.isdir('outputs/gt_path/'):
        os.mkdir('outputs/gt_path')

    with open(f'outputs/gt_path/REVERIE_{DATASET_TYPE}_no_landmark.json', 'w') as fp:
        json.dump(new_data, fp)
        
