import matplotlib.pyplot as plt
from datetime import datetime
import os
import requests
from io import BytesIO
import json
from PIL import Image
import numpy as np
import openai
from openai import OpenAI
import google.generativeai as genai


from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.engine.predictor_glip import GLIPDemo

THRESHOLD = 0.7
OPENAI_API_KEY = os.environ['OPENAI_API_KEY']

CONFIG_FILE = "configs/pretrain/glip_Swin_L.yaml"
WEIGHT_FILE = "MODEL/glip_large_model.pth"
BBOXES_JSON_FILE = '/data/Matterport3DSimulator-duet/VLN-DUET/datasets/REVERIE/annotations/BBoxes.json'
NAVIGABLE_PATH = '/data/NavGPT_data/navigable'
DATASET = 'train'

client = OpenAI(api_key=OPENAI_API_KEY)

def load(path):
    '''
    Given an url of an image, downloads the image and returns a PIL image
    '''
    # response =  requests.get(url)
    pil_image = Image.open(path).convert("RGB")
    image = np.array(pil_image)[:, :, [2, 1, 0]]
    return image

def update_config(cfg):
    cfg.local_rank = 0
    cfg.num_gpus = 1
    cfg.merge_from_file(CONFIG_FILE)
    cfg.merge_from_list(['MODEL.WEIGHT', WEIGHT_FILE])
    cfg.merge_from_list(['MODEL.DEVICE', 'cuda'])
    return cfg

def load_json(fn):
    with open(fn) as f:
        ret = json.load(f)
    return ret

def dump_json(data, fn, force=False):
    if not force:
        assert not os.path.exists(fn)
    with open(fn, 'w') as f:
        json.dump(data, f)

def get_pano_path(scan, vps):
    # path = "/data/cube2panorama/panorama/{}_{}.jpg".format(scan, vp) 
    ans = []
    for vp in vps:
        for i in range(1, 5):
            ans.append(f'/data/base_dir/v1/scans/{scan}/matterport_skybox_images/{vp}_skybox{i}_sami.jpg')

    return ans

REPLACE_OBJECT_TEMPLATE = '''
You should find a new target object to replace the old target_object and return me a new instruction.
Notice: the new target object must doesn't look like any objects(should be different type) in avoid_objects list.
Sometimes, you should review your answer and change the verb to which is suitable for the new target objects.
Important: you can only replace the target object and the verb about it.

Example:
inputs:
{}
outputs:
{}
explanation:
    First, Choose 'the mirror' as the new target object because it isn't in the 'avoid_objects' list, and 'take' is a good verb for 'the mirror'.
    So the new instructions is 'Go to bedroom at the back left side of the house and take the mirror nearest the bedroom door.'

Now it is your turn:
inputs: 
___inputs___
outputs:
'''

def get_replace_object_template() -> str:
    inputs = {
        'instruction': 'Go to bedroom at the back left side of the house and turn on the lamp nearest the bedroom door',
        'target_object': 'lamp',
        'avoid_objects': ['window', 'lamp', 'picture', 'bed'],
    }
    outputs = {
        'new_target_object': 'the mirror',
        'new_instruction': 'Go to bedroom at the back left side of the house and take the mirror nearest the bedroom door',
    }
    template = REPLACE_OBJECT_TEMPLATE.format(json.dumps(inputs, indent=4), json.dumps(outputs, indent=4))
    return template


def get_navigable_viewpoints(scan: str, viewpoint: str) -> list:
    '''
        Get all neighbor vps around it.
    '''
    data = load_json(f'{NAVIGABLE_PATH}/{scan}_navigable.json')    
    navigable_viewpoints = []
    for k, v in data[viewpoint].items():
        navigable_viewpoints.append(k)
    
    return navigable_viewpoints

def get_objects_in_the_rooms(bboxes: dict, scan: str, viewpoint: str) -> list:
    '''
        Get all touchable objects around this viewpoint.
        
        Touchable: define by REVERIE datasets, means the objects is close to this point (maybe 1m).
    '''
    objs = set()
    for k, v in bboxes[f'{scan}_{viewpoint}'].items():
        objs.add(v['name'].replace('#', ' '))
    return list(objs)

def get_avoid_objs(bboxes: dict, scan: str, vps: list) -> list:
    '''
        Get objects around this viewpoint
        
        First, it call get_navigable_viewpoints() to get the neighbor viewpoints.
        Then, it get all the objects around its neighbor, we assume these objects is all visible bbox in this room
        We need this list to avoid generating the objects that exist in this room

    '''
    objs = []
    for i in vps:
        tmp_objs = get_objects_in_the_rooms(bboxes, scan, i)
        objs += tmp_objs
    
    return list(set(objs))

def query(openai: OpenAI, prompt: str):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo-0125",
        response_format={ "type": "json_object" },
        messages=[
            {"role": "system", "content": "Please output JSON."},
            {"role": "user", "content": prompt}
        ]
    )
    
    return (
        json.loads(response.choices[0].message.content),
        response.usage.total_tokens
    )

def gemini_query(prompt: str):
    response = requests.post('http://127.0.0.1:8000', data={'prompt': prompt})
    return response.json()['response']


if __name__ == '__main__':


    cfg = update_config(cfg)
    demo = GLIPDemo(
                cfg,
                min_image_size=800,
                confidence_threshold=THRESHOLD,
                show_mask_heatmaps=False
            )
    demo.color = 255

    if not os.path.isdir(f'THRESHOLD_{THRESHOLD}'):
        os.mkdir(f'THRESHOLD_{THRESHOLD}')

    for DATASET in ['train', 'val_seen', 'val_unseen']:
        for T in ['no', 'with']:
            with open(f'/data/New_REVERIE/outputs/gt_path/REVERIE_{DATASET}_{T}_landmark.json') as fp:
                data = json.load(fp)

            old_results = {}
            '''
            with open(f'/tmp/outputs/after_gpt/REVERIE_{DATASET}_{T}_landmark.json') as fp:
                tmp_data = json.load(fp)

                for i in tmp_data:
                    old_results[i['id']] = i
            '''

            with open(f'/data/New_REVERIE/outputs/after_gpt/REVERIE_{DATASET}_{T}_landmark.json') as fp:
                tmp_data = json.load(fp)
                
                for i in tmp_data:
                    old_results[i['id']] = i

            print(old_results.keys())


            bboxes = load_json(BBOXES_JSON_FILE)

            template = get_replace_object_template()

            new_data = []

            tokens = 0
            LOSS_NUM = 0
            for idx, r in enumerate(data):

                if idx%50==0:
                    dump_json(new_data, f'/data/New_REVERIE/outputs/after_gpt/REVERIE_{DATASET}_{T}_landmark.json', force=True)
                    print('\n', end='')

                path_id = r['id']
                print(path_id)
                if path_id in old_results:
                    print("++++++++++++++++++++++++++++++++++")
                    print("++++++++++++++++++++++++++++++++++")
                    print("++++++++++++++++++++++++++++++++++")
                    print("++++++++++++++++++++++++++++++++++")
                    print("++++++++++++++++++++++++++++++++++")
                    new_r = r.copy()
                    new_r['new_instruction'] = old_results[path_id]['new_instruction']
                    new_r['new_target_object'] = old_results[path_id]['new_target_object']
                    new_data.append(new_r)
                else:

                    OK = False

                    print(datetime.now(), idx, '/', len(data), DATASET)
                    print("LOSS_NUM:", LOSS_NUM)
                    print(f"    use {tokens} tokens")

                    scan = r['scan']
                    explore_vps = r['back_path']
                    # print(r['path'])
                    # print(r['explore_path'])
                    # print(explore_vps)
                    avoid_objs = get_avoid_objs(bboxes, scan, explore_vps)

                    new_r = r.copy()

                    error = False
                    try_time = 0
                    while not OK:
                        # try:
                        try_time += 1
                        inputs = {
                            'instruction': r['instructions'][0],
                            'target_object': r['relation']['target_object'],
                            'avoid_objects': avoid_objs,
                        }        
                        prompt = template.replace('___inputs___', json.dumps(inputs, indent=4))
                        # print(prompt)
                        response, total_tokens = query(client, prompt)
                        # response = gemini_query(prompt)
                        print(response)
                        total_tokens = 0

                        if 'status' in response:
                            error = True
                            break

                        if try_time >= 30:
                            error = True
                            break

                        # print(response, type(response))


                        new_instruction = response['new_instruction']
                        new_target_object = response['new_target_object']
                        print()
                        print(inputs['instruction'])
                        print(inputs['target_object'])
                        print(new_instruction)
                        print(new_target_object)
                        print()
                        tokens += total_tokens

                        # new_r['instruction'] = inputs['instruction']
                        new_r['new_instruction'] = new_instruction
                        new_r['new_target_object'] = new_target_object

                        image_paths = get_pano_path(scan, explore_vps)

                        found_count = 0
                        for path in image_paths:
                            image = load(path)
                            result, top_predictions = demo.run_on_web_image(image, new_target_object, demo.confidence_threshold)
                            scores = top_predictions.get_field('scores') 
                            found_count += scores.shape[0]

                        if found_count == 0:
                            OK = True
                            new_data.append(new_r)
                            print("= "*50)
                        else:
                            avoid_objs += [ new_target_object ]
                            print(f"New avoid_objs={avoid_objs}")
                    if error:
                        LOSS_NUM += 1
                        continue

            dump_json(new_data, f'..//outputs/after_gpt/REVERIE_{DATASET}_{T}_landmark.json', force=True)
