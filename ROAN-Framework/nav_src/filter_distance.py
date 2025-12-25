from glob import glob
import math
from collections import defaultdict
import json

def load_json(filename):
    with open(filename) as fp:
        data = json.load(fp)

    return data

def load_floorplan():
    region_label_lookup = load_region_label_lookup()

    house_files = glob('/data/code/research/VLN/base_dir/v1/scans/*/house_segmentations/*.house')
    #room_lookups = {}
    #floor_lookups = {}
    #room_bbox_lookups = {}
    #node_coor_lookups = {}

    node_region_lookups = {}
    region_room_lookups = {}
    region_object_lookups = {}
    node_locations_lookups = {}

    for house_file in house_files:
        scan_id = house_file.split("/")[-3]
        regions, floors, node_id_regions, node_id_floors = {}, {}, {}, {}
        room_bboxes = {}
        node_coors = {}
        node_locations = {}
        region_objects = defaultdict(list)
        object_name_lookup = {}
        #print(scan_id, datetime.now())
        #house_lines = []
        for line in open(house_file):
            house_line = line.strip()
            #house_lines.append(line.strip())

            #for house_line in house_lines[1:]:
            house_line_cols = house_line.split()
            house_line_type = house_line_cols[0]
            house_line_cols = house_line_cols[1:]

            if house_line_type=='R':
                region_index, level_index, _, _, label, px, py, pz, xlo, ylo, zlo, xhi, yhi, zhi, height,_,_,_,_ = house_line_cols
                regions[region_index] = region_label_lookup[label]
                floors[region_index] = level_index
                room_bboxes[region_index] = {
                    'name': region_label_lookup[label],
                    'floor': level_index
                }
                #for var_name in ['px', 'py', 'pz', 'xlo', 'ylo', 'zlo', 'xhi', 'yhi', 'zhi', 'height']:
                    # room_bboxes[region_index][var_name] = float(eval(var_name))

            if house_line_type=='P':
                node_id, panorama_index, region_index, _, px, py, pz, _,_,_,_,_ = house_line_cols
                node_id_regions[node_id] = region_index#regions[region_index]
                node_locations[node_id] = (px, py, pz)
                #node_id_floors[node_id] = int(floors[region_index]) + 1
                #node_coors[node_id] = (float(px), float(py), float(pz))
                #raise
            #if house_line_type=='I':
                #break
            if house_line_type=='C':
                category_index, category_mapping_index, category_mapping_name, mpcat40_index, mpcat40_name, _,_,_,_,_ = house_line_cols
                object_name_lookup[category_index] = category_mapping_name

            if house_line_type=='O':
                object_index, region_index, category_index, px, py, pz, a0x, a0y, a0z, a1x, a1y, a1z, r0, r1, r2, _, _, _, _, _, _, _, _ = house_line_cols
                if category_index=='-1' or region_index=='-1':
                    #print("error")
                    continue
                region_objects[region_index].append(object_name_lookup[category_index])
        #room_lookups[scan_id] = node_id_regions
        #floor_lookups[scan_id] = node_id_floors
        region_room_lookups[scan_id] = room_bboxes
        node_region_lookups[scan_id] = node_id_regions
        node_locations_lookups[scan_id] = node_locations
        region_object_lookups[scan_id] = {k:sorted(v) for k,v in region_objects.items()}
        #node_coor_lookups[scan_id] = node_coors
    return node_region_lookups, region_room_lookups, region_object_lookups, node_locations_lookups

def load_region_label_lookup():
    region_label_lookup = {
    'a': 'bathroom',
    'b': 'bedroom',
    'c': 'closet',
    'd': 'dining room',
    'e': 'entryway',#/foyer/lobby (should be the front door, not any door)
    'f': 'familyroom',# (should be a room that a family hangs out in, not any area with couches)
    'g': 'garage',#
    'h': 'hallway',#
    'i': 'library',# (should be room like a library at a university, not an individual study)
    'j': 'laundryroom',#/mudroom (place where people do laundry, etc.)
    'k': 'kitchen',#
    'l': 'living room',# (should be the main "showcase" living room in a house, not any area with couches)
    'm': 'meeting room',#/conferenceroom
    'n': 'lounge',# (any area where people relax in comfy chairs/couches that is not the family room or living room
    'o': 'office',# (usually for an individual, or a small set of people)
    'p': 'porch',#/terrace/deck/driveway (must be outdoors on ground level)
    'r': 'recreation',#/game (should have recreational objects, like pool table, etc.)
    's': 'stairs',#
    't': 'toilet',# (should be a small room with ONLY a toilet)
    'u': 'utility room',#/toolroom
    'v': 'tv',# (must have theater-style seating)
    'w': 'gym',#workout/gym/exercise
    'x': 'outdoor',# areas containing grass, plants, bushes, trees, etc.
    'y': 'balcony',# (must be outside and must not be on ground floor)
    'z': 'other room',# (it is clearly a room, but the function is not clear)
    'B': 'bar',#
    'C': 'classroom',#
    'D': 'dining booth',#
    'S': 'spa',#/sauna
    'Z': 'junk',# (reflections of mirrors, random points floating in space, etc.)
    '-': 'no label',#
    }
    return region_label_lookup

def get_distance(nodeA, nodeB):
    a_x = float(nodeA[0])
    a_y = float(nodeA[1])
    b_x = float(nodeB[0])
    b_y = float(nodeB[1])
    return math.sqrt( (a_x-b_x) ** 2 + (a_y-b_y) ** 2)

node_region, region_room, region_obj, node_locations = load_floorplan()


data = load_json('../datasets/REVERIE/annotations/REVERIE_val_unseen_instr.json')
    
counter = 0
new_data = []
for i in data:
    scan = i['scan']

    stop = i['path'][-1]

    room_id = node_region[scan][stop]
    room_name = region_room[scan][node_region[scan][stop]]
    
    stop_location = node_locations[scan][stop]



    max_distance, max_node = 0.0, None
    for k, v in node_region[scan].items():
        if v == room_id:
            k_location = node_locations[scan][k]
            distance = get_distance(stop_location, k_location)

            if distance >= 3.0 and distance > max_distance:
                max_distance = distance
                max_node = k

    if max_node:
        print(scan, stop, room_name, room_id)
        counter += 1
        print(max_node, max_distance)
        print()

        new_i = i.copy()
        new_i['start'] = max_node
        new_i['stop'] = stop
        new_data.append(new_i)
print(counter)
with open('../datasets/REVERIE/annotations/new_REVERIE_val_unseen_instr.json', 'w') as fp:
    json.dump(new_data, fp)





