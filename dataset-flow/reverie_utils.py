import json
import os

def get_obj2vps():
    '''
    check which viewpoints can see the object

    key = scan_objid
    ans[key] = [vps]
    '''
    ans = {}
    for file in os.listdir('./data/BBox'):
        scan, vp = file.replace('.json', '').split('_')
        with open(f'./data/BBox/{file}') as fp:
            data = json.load(fp)
        for obj_id, obj_info in data[vp].items():
            if obj_info['visible_pos']:
                key = f'{scan}_{obj_id}'
                if key not in ans:
                    ans[key] = [vp]
                else:
                    ans[key].append(vp)
    return ans

def get_connectivity():
    '''
    ans[from] = [available_access_vps]
    positions[vp] = [x, y, z]
    '''
    ans = {}
    positions = {}
    for file in os.listdir('connectivity'):
        if 'json' in file:
            scan = file.replace('_connectivity.json', '')
            with open(f'connectivity/{file}') as fp:
                data = json.load(fp)
            for i, item in enumerate(data):
                if item['included']:
                    for j, conn in enumerate(item['unobstructed']):
                        if conn and data[j]['included']:
                            from_vp = data[i]['image_id']
                            to_vp = data[j]['image_id']
                            positions[from_vp] = [
                                item['pose'][3], item['pose'][7], item['pose'][11]
                            ]

                            if from_vp not in ans:
                                ans[from_vp] = [to_vp]
                            else:
                                ans[from_vp].append(to_vp)
    return ans, positions

def get_node_region():
    '''
    node_region[scan][vp] = id(int)
    '''
    with open('./data/node_region.json') as fp:
        node_region = json.load(fp)

    return node_region

# def get_same_room_vps(connectivity, node_region, scan, vp):
#     ans = set([vp])
#     for _vp, region in node_region[scan].items():
#         if _vp in connectivity:                 # means it's a accessable vp
#             if region == node_region[scan][vp]: # means it's in the same room
#                 ans.add(_vp)
#     return list(ans)

def get_same_room_vps(connectivity, node_region, scan, should_visit_vps):
    vps = set()
    visited = []

    def dfs(start):
        # print(start)
        for vp in connectivity[start]:
            # dfs 找 同一個種類 region 且相鄰的點
            if vp not in visited and node_region[scan][vp] == node_region[scan][start_vp]:
                vps.add(vp)
                visited.append(vp)
                dfs(vp)


    for start_vp in should_visit_vps:
        vps.add(start_vp)
        visited = []
        dfs(start_vp)
    # print(vps)
    return list(vps)

def get_vp2objs():
    '''
    ans[vp] = [objid1, objid2......]
    '''
    ans = {}
    for file in os.listdir('./data/BBox'):
        scan, vp = file.replace('.json', '').split('_')
        with open(f'./data/BBox/{file}') as fp:
            data = json.load(fp)
        objs = []
        for obj_id, obj_info in data[vp].items():
            if obj_info['visible_pos']:
                objs.append(obj_id)

        assert vp not in ans, "error"
        ans[vp] = objs

    return ans

def get_objs_same_room(vp2objs, vps):
    ans = set()
    for vp in vps:
        for obj in vp2objs[vp]:
            ans.add(obj)
    return list(ans)
