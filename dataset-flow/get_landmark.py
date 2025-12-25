import os
import json
from typing import Tuple, List
from llm import *

#  api setup
genai.configure(api_key=os.environ["GEMINI_API_KEY"])

PROMPT_TEMPLATE = '''
You are a housework planner.
You will receive a instruction, you should return the target area, target object and sometimes you can find a landmark about the target object.

A landmark is an object that can help locate the target object. Please identify the landmark described in the instruction that indicates the position of the target object.

You should consider what room the agent should enter. Then what object should it find and what landmark is near the target.
Important: The object that help find the room is not the landmark.
Important: you should not return the article(a, an, the...).

Example 1:
    inputs: {
        "instruction": "Go to the hallway and bring me the picture acrros from the table."
    }

    outputs: {
        "target_area": "hallway",
        "target_object": "picture",
        "landmark": "table"
    }
    the landmark is "table" because the agent should find the table first, then it can find the picture near it.

Example 2:
    inputs: {
        "instruction": "Go to the hallway and bring me the clock"
    }

    outputs: {
        "target_area": "hallway",
        "target_object": "clock",
        "landmark": "None"
    }
    there is no landmark in the instruction so that you should return "None".

Example 3:
    inputs: {
        "instruction": "Go to the massage room with the bamboo plant and cover the diamond shaped window"
    }

    outputs: {
        "target_area": "message room",
        "target_object": "diamond shaped window",
        "landmark": "None"
    }
    bamboo plant is not the landmark beacuse it's to descrbe the room not indicates the position of the target object

It's your turn now:
    inputs: {
        "instruction": "___inputs___"
    }

    outputs: 
'''

# load original instructions
for DATASET_TYPE in ['train', 'val_seen', 'val_unseen']:
    DATASET_TYPE = 'val_seen'
    with open(f'data/REVERIE_{DATASET_TYPE}.json') as fp:
        data = json.load(fp)


# only get the first instruction as the original instruciton
# 
    new_data = []
    error_index = []
    for index, i in enumerate(data):
        print(index, len(data))
# for index in [9, 11, 10]:
        # i = data[index]
        new_i = i.copy()
        instrs = i['instructions']
        scan = i['scan']
        uid = i['id']

        instr = instrs[0]
        
        prompt = PROMPT_TEMPLATE.replace('___inputs___', instr)
        print(instr)
        try:
            response = gemini_query(prompt)
            print(response)
            print("=" * 20)
            new_i['relation'] = response
            new_i['instructions'] = [instr]
            new_i['found'] = [True]
            new_data.append(new_i)
        except:
            error_index.append({
                'index': index,
                'prompt': prompt,
            })



# generate outputs
    if not os.path.isdir('./outputs'):
        os.mkdir('./outputs')
    with open(f'outputs/after_landmark/REVERIE_{DATASET_TYPE}.json', 'w') as fp:
        json.dump(new_data, fp)

    for i in error_index:
        print(i)


