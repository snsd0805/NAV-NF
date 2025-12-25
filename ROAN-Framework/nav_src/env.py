''' Batched REVERIE navigation environment '''

import json
import os
import numpy as np
import random
import networkx as nx
from collections import defaultdict
from glob import glob

from utils.data import load_nav_graphs
from eval_utils import cal_dtw, cal_cls
from utils.graph_utils import NavGraph

ERROR_MARGIN = 3.0

obj2vps = {}
bbox_data = json.load(open('/data/Matterport3DSimulator-duet/VLN-DUET/datasets/REVERIE/annotations/BBoxes.json'))
for scanvp, value in bbox_data.items():
    scan, vp = scanvp.split('_')
    # for all visible objects at that viewpoint
    for objid, objinfo in value.items():
        if objinfo['visible_pos']:
            # if such object not already in the dict
            obj2vps.setdefault(scan+'_'+objid, [])
            obj2vps[scan+'_'+objid].append(vp)

def load_floorplan():
    region_label_lookup = load_region_label_lookup()

    house_files = glob('/data/code/research/VLN/base_dir/v1/scans/*/house_segmentations/*.house')

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

with open('./node_region.json') as fp:
    node_region = json.load(fp)

class Simulator(object):
    ''' A simple simulator in Matterport3D environment '''

    def __init__(
            self,
            navigable_dir: str,):
        self.heading = 0
        self.elevation = 0
        self.scan_ID = ''
        self.viewpoint_ID = ''
        self.navigable_dir = navigable_dir
        self.navigable_dict = {}
        self.candidate = {}
        self.gmap = NavGraph()

        self.node_region, self.region_room, self.region_obj, self.node_locations = load_floorplan()


    def newEpisode(
            self, 
            scan_ID: str, 
            viewpoint_ID: str,
            heading: int, 
            elevation: int,
            start: str,
            target: str,
            clip_target: str,
        ):
        self.heading = heading
        self.elevation = elevation
        self.scan_ID = scan_ID
        self.viewpoint_ID = viewpoint_ID
        self.start = start
        self.target = target
        self.clip_target = clip_target
        # Load navigable dict
        navigable_path = os.path.join(self.navigable_dir, self.scan_ID + '_navigable.json')
        with open(navigable_path, 'r') as f:
            self.navigable_dict = json.load(f)

        '''
        self.navigable_dict = {}
        for start, v in navigable_dict.items():
            self.navigable_dict[start] = {}
            # print("BEFORE: ", len(navigable_dict[start]))
            for to, _v in navigable_dict[start].items():
                start_region = self.node_region[scan_ID][start]
                to_region = self.node_region[scan_ID][to]
                if start_region == to_region:
                    self.navigable_dict[start][to] = _v 
                # print(start_region, to_region)
            # print("AFTER: ", len(self.navigable_dict[start]))
        '''

        # Get candidate
        self.getCandidate()
    
    def updateGraph(self):
        # build graph
        for candidate in self.candidate.keys():
            self.gmap.update_connection(self.viewpoint_ID, candidate)

    def getState(self) -> dict:
        self.state = {
            'scanID': self.scan_ID,
            'viewpointID': self.viewpoint_ID,
            'heading': self.heading,
            'elevation': self.elevation,
            'candidate': self.candidate,
            'start': self.start,
            'target': self.target,
            'clip_target': self.clip_target,
        }
        return self.state
    
    def getCandidate(self):
        """
        Get the agent's candidate list from pre-stored navigable dict.
        """
        self.candidate = self.navigable_dict[self.viewpoint_ID]
        self.updateGraph()
    
    def makeAction(self, next_viewpoint_ID):
        """
        Make action and update the agent's state.
        """
        if next_viewpoint_ID == self.viewpoint_ID:
            return
        elif next_viewpoint_ID in self.candidate.keys():
            self.heading = self.candidate[next_viewpoint_ID]['heading']
            self.elevation = self.candidate[next_viewpoint_ID]['elevation']
        self.viewpoint_ID = next_viewpoint_ID
        self.getCandidate()


class EnvBatch(object):
    ''' A simple wrapper for a batch of MatterSim environments,
        using discretized viewpoints and pretrained features '''

    def __init__(self, navigable_dir, feat_db=None, batch_size=100):
        """
        1. Load pretrained image feature
        2. Init the Simulator.
        :param feat_db: The name of file stored the feature.
        :param batch_size:  Used to create the simulator list.
        """
        self.feat_db = feat_db
        
        self.sims = []
        for i in range(batch_size):
            sim = Simulator(navigable_dir)
            self.sims.append(sim)

    def _make_id(self, scanId, viewpointId):
        return scanId + '_' + viewpointId

    def newEpisodes(self, scanIds, viewpointIds, headings, starts, targets, clip_targets):
        for i, (scanId, viewpointId, heading, start, target, clip_target) in enumerate(zip(scanIds, viewpointIds, headings, starts, targets, clip_targets)):
            self.sims[i].newEpisode(scanId, viewpointId, heading, 0, start, target, clip_target)

    def getStates(self):
        """
        Get list of states augmented with precomputed image features. rgb field will be empty.
        Agent's current view [0-35] (set only when viewing angles are discretized)
            [0-11] looking down, [12-23] looking at horizon, [24-35] looking up
        :return: [ ((36, 2048), sim_state) ] * batch_size
        """
        feature_states = []
        for i, sim in enumerate(self.sims):
            state = sim.getState()

            feature = self.feat_db.get_image_observation(state["scanID"], state["viewpointID"])
            feature_states.append((feature, state))
        return feature_states

    def makeActions(self, next_viewpoint_IDs):
        ''' Take an action using the full state dependent action interface (with batched input)'''
        for i, next_viewpoint_ID in enumerate(next_viewpoint_IDs):
            self.sims[i].makeAction(next_viewpoint_ID)


class REVERIENavBatch(object):
    ''' Implements the REVERIE navigation task, using discretized viewpoints and pretrained features '''

    def __init__(
        self, view_db, instr_data, connectivity_dir, navigable_dir,
        batch_size=1, seed=0, name=None
    ):
        self.env = EnvBatch(navigable_dir, feat_db=view_db, batch_size=batch_size)
        self.data = instr_data
        self.scans = set([x['scan'] for x in self.data])
        self.connectivity_dir = connectivity_dir
        self.batch_size = batch_size
        self.name = name

        self.gt_trajs = self._get_gt_trajs(self.data) # for evaluation

        # use different seeds in different processes to shuffle data
        '''
        self.seed = seed
        random.seed(self.seed)
        random.shuffle(self.data)
        '''


        self.ix = 0
        self._load_nav_graphs()
        
        self.buffered_state_dict = {}
        print('%s loaded with %d instructions, using splits: %s' % (
            self.__class__.__name__, len(self.data), self.name))

    def _get_gt_trajs(self, data):
        gt_trajs = {
            x['new_reverie_id']: (x['scan'], x['path']) \
                for x in data if len(x['path']) > 1
        }
        return gt_trajs

    def size(self):
        return len(self.data)

    def _load_nav_graphs(self):
        """
        load graph from self.scan,
        Store the graph {scan_id: graph} in self.graphs
        Store the shortest path {scan_id: {view_id_x: {view_id_y: [path]} } } in self.paths
        Store the distances in self.distances. (Structure see above)
        Load connectivity graph for each scan, useful for reasoning about shortest paths
        :return: None
        """
        print('Loading navigation graphs for %d scans' % len(self.scans))
        self.graphs = load_nav_graphs(self.connectivity_dir, self.scans)
        self.shortest_paths = {}
        for scan, G in self.graphs.items():  # compute all shortest paths
            self.shortest_paths[scan] = dict(nx.all_pairs_dijkstra_path(G))
        self.shortest_distances = {}
        for scan, G in self.graphs.items():  # compute all shortest paths
            self.shortest_distances[scan] = dict(nx.all_pairs_dijkstra_path_length(G))

    def _next_minibatch(self, batch_size=None, **kwargs):
        """
        Store the minibach in 'self.batch'
        """
        if batch_size is None:
            batch_size = self.batch_size
        
        batch = self.data[self.ix: self.ix+batch_size]
        if len(batch) < batch_size:
            random.shuffle(self.data)
            self.ix = batch_size - len(batch)
            batch += self.data[:self.ix]
        else:
            self.ix += batch_size
        self.batch = batch

    def reset_epoch(self, shuffle=False):
        ''' Reset the data index to beginning of epoch. Primarily for testing.
            You must still call reset() for a new episode. '''
        if shuffle:
            random.shuffle(self.data)
        self.ix = 0

    def _get_obs(self):
        obs = []
        for i, (feature, state) in enumerate(self.env.getStates()):
            item = self.batch[i]

            ob = {
                'obs' : feature["detail"],
                'obs_summary' : feature["summary"],
                'objects' : feature["objects"],
                # 'instr_id' : item['instr_id'],
                # 'action_plan' : item['action_plan'],
                'scan' : state['scanID'],
                'viewpoint' : state['viewpointID'],
                'heading' : state['heading'],
                'elevation' : state['elevation'],
                'candidate': state['candidate'],
                'instruction' : item['instruction'],
                'gt_path' : item['path'],
                'path_id' : item['path_id'],
                'start': item['start'],
                'new_reverie_id': item['new_reverie_id'],
                'target': item['target'],
                'clip_target': item['clip_target']
            }
            # RL reward. The negative distance between the state and the final state
            # There are multiple gt end viewpoints on REVERIE. 

            '''
            if ob['instr_id'] in self.gt_trajs:
                ob['distance'] = self.shortest_distances[ob['scan']][ob['viewpoint']][item['path'][-1]]
            else:
                ob['distance'] = 0
            '''

            obs.append(ob)
        return obs

    def reset(self, **kwargs):
        ''' Load a new minibatch / episodes. '''
        self._next_minibatch(**kwargs)
        
        scanIds = [item['scan'] for item in self.batch]
        viewpointIds = [item['path'][0] for item in self.batch]
        headings = [item['heading'] for item in self.batch]
        starts = [item['start'] for item in self.batch]
        targets = [item['target'] for item in self.batch]
        clip_targets = [item['clip_target'] for item in self.batch]
        self.env.newEpisodes(scanIds, starts, headings, starts, targets, clip_targets)
        return self._get_obs()

    def step(self, next_viewpoint_IDs):
        ''' Take action (same interface as makeActions) '''
        self.env.makeActions(next_viewpoint_IDs)
        return self._get_obs()

    ############### Nav Evaluation ###############
    def _get_nearest(self, shortest_distances, goal_id, path):
        near_id = path[0]
        near_d = shortest_distances[near_id][goal_id]
        for item in path:
            d = shortest_distances[item][goal_id]
            if d < near_d:
                near_id = item
                near_d = d
        return near_id

    def _eval_item(self, scan, pred_path, gt_path, gt_found, found, gt_objid):
        scores = {}

        shortest_distances = self.shortest_distances[scan]

        path = sum(pred_path, [])
        # assert gt_path[0] == path[0], 'Result trajectories should include the start position'

        nearest_position = self._get_nearest(shortest_distances, gt_path[-1], path)

        scores['nav_error'] = shortest_distances[path[-1]][gt_path[-1]]
        scores['oracle_error'] = shortest_distances[nearest_position][gt_path[-1]]

        scores['action_steps'] = len(pred_path) - 1
        scores['trajectory_steps'] = len(path) - 1
        scores['trajectory_lengths'] = np.sum([shortest_distances[a][b] for a, b in zip(path[:-1], path[1:])])

        gt_lengths = np.sum([shortest_distances[a][b] for a, b in zip(gt_path[:-1], gt_path[1:])])

        scores['found_success'] = float(gt_found == found)

        goal_viewpoints = set(obj2vps['%s_%s'%(scan, str(gt_objid))])
        
        pred_stop_region = node_region[scan][path[-1]]
        gt_stop_region = node_region[scan][gt_path[-1]]

        # scores['success'] = float(scores['nav_error'] < ERROR_MARGIN)
        scores['success'] = float(path[-1] in goal_viewpoints)
        scores['room_success'] = float(gt_stop_region == pred_stop_region)
        # scores['oracle_success'] = float(scores['oracle_error'] < ERROR_MARGIN)
        scores['oracle_success'] = float(any(x in goal_viewpoints for x in path))
        
        scores['spl'] = scores['success'] * gt_lengths / max(scores['trajectory_lengths'], gt_lengths, 0.01)
        scores['sspl_1'] = scores['success'] * gt_lengths / max(scores['trajectory_lengths'], gt_lengths, 0.01) * scores['found_success']
        scores['sspl_2'] = scores['room_success'] * gt_lengths / max(scores['trajectory_lengths'], gt_lengths, 0.01) * scores['found_success']
        scores['sspl_3'] = scores['oracle_success'] * gt_lengths / max(scores['trajectory_lengths'], gt_lengths, 0.01) * scores['found_success']

        scores['ss_1'] = scores['success'] * scores['found_success']
        scores['ss_2'] = scores['room_success'] * scores['found_success']
        scores['ss_3'] = scores['oracle_success'] * scores['found_success']

        scores.update(
            cal_dtw(shortest_distances, path, gt_path, scores['success'], ERROR_MARGIN)
        )
        scores['CLS'] = cal_cls(shortest_distances, path, gt_path, ERROR_MARGIN)

        return scores

    def eval_metrics(self, preds):
        ''' Evaluate each agent trajectory based on how close it got to the goal location 
        the path contains [view_id, angle, vofv]'''
        print('eval %d predictions' % (len(preds)))

        metrics = defaultdict(list)
        for item in preds:
            instr_id = item['instr_id']
            traj = item['trajectory']
            obj_id = instr_id.split('_')[1]
            scan, gt_traj = self.gt_trajs[instr_id]
            traj_scores = self._eval_item(scan, traj, gt_traj, item['gt_found'], item['found'], obj_id)
            for k, v in traj_scores.items():
                metrics[k].append(v)
            metrics['instr_id'].append(instr_id)
        
        avg_metrics = {
            'action_steps': np.mean(metrics['action_steps']),
            'steps': np.mean(metrics['trajectory_steps']),
            'lengths': np.mean(metrics['trajectory_lengths']),
            'nav_error': np.mean(metrics['nav_error']),
            'oracle_error': np.mean(metrics['oracle_error']),
            'sr': np.mean(metrics['success']) * 100,
            'room_success': np.mean(metrics['room_success']) * 100,
            'found_success': np.mean(metrics['found_success']) * 100,
            'oracle_sr': np.mean(metrics['oracle_success']) * 100,
            'spl': np.mean(metrics['spl']) * 100,
            'sspl_1': np.mean(metrics['sspl_1']) * 100,
            'sspl_2': np.mean(metrics['sspl_2']) * 100,
            'sspl_3': np.mean(metrics['sspl_3']) * 100,
            'ss_1': np.mean(metrics['ss_1']) * 100,
            'ss_2': np.mean(metrics['ss_2']) * 100,
            'ss_3': np.mean(metrics['ss_3']) * 100,
            'nDTW': np.mean(metrics['nDTW']) * 100,
            'SDTW': np.mean(metrics['SDTW']) * 100,
            'CLS': np.mean(metrics['CLS']) * 100,
        }
        return avg_metrics, metrics
        
