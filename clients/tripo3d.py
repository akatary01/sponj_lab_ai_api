import json
import base64
import requests
import websockets
from PIL import Image
from utils.img import img_to_bytes
from utils.client import BaseClient
from utils import API_KEYS, BASE_DIR

TRIPO_3D_API_KEY = API_KEYS['TRIPO_3D_API_KEY']

class Tripo3dClient(BaseClient):
    def __init__(self):
        self.face_limit = 15000
        self.headers = {
            "Authorization": f"Bearer {TRIPO_3D_API_KEY}"
        }

        self.log_path = f"{BASE_DIR}/logs/tripo3d.log"
        
        self.base_url = "https://api.tripo3d.ai/v2/openapi"
        self.base_socket_url = "wss://api.tripo3d.ai/v2/openapi"

        self.endpoints = {
            "task": f"{self.base_url}/task",
            "upload": f"{self.base_url}/upload",
            "watch": f"{self.base_socket_url}/task/watch"
        }

    def text_to_mesh(self, prompt: str, **kwargs):
        body = {
            "prompt": prompt,
            "type": "text_to_model",
            "face_limit": self.face_limit,
            "model_version": "v2.0-20240919",

            **kwargs
        }

        return self(self.endpoints["task"], headers=self.headers, json=body)

    def img_to_mesh(self, img: Image, **kwargs):
        upload_response = self.upload(img).json()

        body = {
            "type": "image_to_model",
            "model_version": "v2.0-20240919",
            "file": {
                "type": "png",
                "file_token": upload_response['data']["image_token"],
                "url": base64.b64encode(img_to_bytes(img)).decode('utf-8')
            },
            "face_limit": self.face_limit,

            **kwargs
        }

        return self(self.endpoints["task"], headers=self.headers, json=body)

    def convert_mesh(self, task_id: str, format = "obj", **kwargs):
        body = {
            "pbr": False,
            "quad": False,
            "texture": False,
            "format": format,
            "type": "convert_model",
            "face_limit": self.face_limit,
            "original_model_task_id": task_id,

            **kwargs
        }

        return self(self.endpoints["task"], headers=self.headers, json=body)
    
    def get_task(self, task_id: str, **kwargs):
        print(f"Fetching task {task_id}...")
        url = f"{self.endpoints['task']}/{task_id}"
        return requests.get(url, headers=self.headers).json()

    async def watch(self, task_id = "all", on_success = None, **kwargs):
        all = task_id == "all"
        url = f"{self.endpoints['watch']}/{task_id}"
        self.log(f"[Tripo3dClient][watch] >> watching {url}...")

        try:
            async with websockets.connect(url, additional_headers=self.headers, ping_interval=None) as websocket:
                while True:
                    message = await websocket.recv()
                    try:
                        data = json.loads(message)
                        self.log(f"[Tripo3dClient][watch] (data) >> {data}")

                        data = data['data']
                        result = data['result']
                        status = data['status']
                        task_id = data['task_id']
                        task_input = data['input']
                        
                        if status == 'success':
                            ext = "glb"
                            if 'format' in task_input:
                                ext = task_input['format'].lower()
                            
                            mesh_url = None
                            if ext == "glb":
                                mesh_url = result['pbr_model']['url']
                            elif ext == "obj": 
                                mesh_url = result['model']['url']

                            if on_success is not None and mesh_url is not None:
                                on_success(task_id, mesh_url, ext, data['type'])

                        if status not in ['running', 'queued']:
                            if not all: break
                            
                    except json.JSONDecodeError:
                        self.log(f"[Tripo3dClient][watch] non-json message >> {message}")
                        if not all: break
        except Exception as error:
            print(error)
            # await self.watch(task_id, on_success, **kwargs)
        return data
    
    def upload(self, img: Image, **kwargs):
        files = {
            "file": img_to_bytes(img)
        }

        return self.post(self.endpoints["upload"], files=files, headers=self.headers)
    
    def __call__(self, url, **kwargs):
        task_response = self.post(url, **kwargs).json()

        task_id = task_response['data']['task_id']
        return task_id