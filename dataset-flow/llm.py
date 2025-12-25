import json
from openai import OpenAI
from typing import Tuple, List

import google.generativeai as genai
import os


def openai_query(client: OpenAI, prompt: str) -> Tuple[str, int]:
    '''

    Query from OPENAI API

    args:
        OpenAI object: for api request
        prompt: (str)
    return:
        response: (str)
        total_tokens: (int)

    '''
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        response_format={ "type": "json_object" },
        messages=[
            {"role": "system", "content": "Please output JSON."},
            {"role": "user", "content": prompt}
        ]
    )

    return (
        json.loads(response.choices[0].message.content),
        response.usage.total_tokens,
    )


def gemini_query(prompt: str):
    model = genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            generation_config={"response_mime_type": "application/json"})
    response = model.generate_content(prompt)
    return json.loads(response.text)

