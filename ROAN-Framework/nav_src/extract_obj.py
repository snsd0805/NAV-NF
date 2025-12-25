import openai
from openai import OpenAI
import os
import json

OPENAI_API_KEY = os.environ['OPENAI_API_KEY']

PROMPT_TEMPLATE = """
You are a good housework assistant, please help me to find the target object in a housework instruction.
You will receive a housework instruction, and you need to return the target object and its location.

For example:
Input:
    {
        "instruction": "Enter the kitchen and pick up the cup on the table"
    }
Output:
    {
        "target": "the cup on the table"
    }

Now, it's your turn:
Input:
    {
        "instruction": ___input___
    }
Output:
"""

def query(openai: OpenAI, prompt: str):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo-1106",
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

def load_json(filename):
    with open(filename) as fp:
        data = json.load(fp)
    return data

def dump_json(data, filename):
    with open(filename, 'w') as fp:
        json.dump(data, fp)

if __name__ == '__main__':

    client = OpenAI(api_key=OPENAI_API_KEY)
    data = load_json('../datasets/REVERIE/annotations/REVERIE_val_unseen_instr.json')

    for index, i in enumerate(data):
        instr = i['instruction']
        prompt = PROMPT_TEMPLATE.replace('___input___', instr)

        OK = False
        while not OK:
            response, token = query(client, prompt)
            if 'target' in response:
                target = response['target']
                OK = True
        i['target'] = target
        print(instr)
        print(target)
        print()

    dump_json(data, 'new_REVERIE_val_unseen_instr.json')
    

