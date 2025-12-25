import json
import shutil
import random
import os
from reverie_utils import \
    get_obj2vps, \
    get_node_region, \
    get_connectivity, \
    get_same_room_vps

if not os.path.isdir('validate'):
    os.mkdir('validate')

with open('./outputs/final_merge/REVERIE_train.json') as fp:
    data = json.load(fp)

random.shuffle(data)

# key = scan_objid
# ans[key] = [vps]
obj2vps = get_obj2vps()

# node_region[scan][vp] = id(int)
node_region = get_node_region()

# ans[from] = [available_access_vps]
# positions[vp] = [x, y, z]
connectivity, positions = get_connectivity()


for i in data[:50]:
    uid = i['id']
    if not os.path.isdir('validate/{}'.format(uid)):
        os.mkdir('validate/{}'.format(uid))
        os.mkdir('validate/{}/pano'.format(uid))
        os.mkdir('validate/{}/scans'.format(uid))
    print(i)

    scan = i['scan']
    obj_id = i['objId']
    should_visit_vps = obj2vps[f'{scan}_{obj_id}']

    same_room_vps = get_same_room_vps(connectivity, node_region, scan, should_visit_vps)

    with open('validate/{}/target_{}.txt'.format(uid, i['new_target_object'].replace(' ', '_')), 'w') as fp:
        fp.write('{}\n'.format(scan))
        fp.write('{}\n'.format(i['instructions'][0]))
        fp.write('{}\n'.format(i['instructions'][1]))

    for vp in same_room_vps:
        shutil.copyfile('/data/base_dir/v1/only_pano/{}/{}_skybox_small.jpg'.format(scan, vp), 'validate/{}/pano/{}.jpg'.format(uid, vp))
        for i in [0, 1, 2, 3, 4, 5]:
            shutil.copyfile('/data/base_dir/v1/scans/{}/matterport_skybox_images/{}_skybox{}_sami.jpg'.format(scan, vp, i), 'validate/{}/scans/{}_{}.jpg'.format(uid, vp, i))

     
