import io
import json
import torch
import base64
import requests
from PIL import Image
from openai import OpenAI
from requests import Response
from utils.img import img_to_bytes
from utils.client import BaseClient
from utils import API_KEYS, PROMPTS, BASE_DIR

OPENAI_API_KEY = API_KEYS['OPENAI_API_KEY']
    
openai_client = OpenAI(api_key=OPENAI_API_KEY)

class OpenAIClient(BaseClient):
    def __init__(self):
        self.log_path = f"{BASE_DIR}/logs/openai.log"
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {OPENAI_API_KEY}"
        }

    def respond(self, prompt: str, **kwargs) -> str:
        url = "https://api.openai.com/v1/chat/completions"
        
        body = {
            'model': "gpt-4o-mini",
            "messages": [
                {
                "role": "user",
                "content": [
                    {
                    "type": "text",
                    "text": prompt
                    }
                ]
                }
            ],
            "max_tokens": 300
        }

        return self.decode(self.post(url, headers=self.headers, json=body))
    
    def caption(self, img: Image, prompt = PROMPTS['caption'], **kwargs) -> str:
        url = "https://api.openai.com/v1/chat/completions"
        
        ext = "png"
        img_base64 = base64.b64encode(img_to_bytes(img)).decode('utf-8')

        body = {
            'model': "gpt-4o-mini",
            "messages": [
                {
                "role": "user",
                "content": [
                    {
                    "type": "text",
                    "text": prompt
                    },
                    {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/{ext};base64,{img_base64}"
                    }
                    }
                ]
                }
            ],
            "max_tokens": 300
        }

        return self.decode(self.post(url, headers=self.headers, json=body))

    def decode(self, response: Response) -> str:
        return response.json()['choices'][0]['message']['content']
