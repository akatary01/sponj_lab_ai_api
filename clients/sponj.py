import base64
from PIL import Image
from typing import List

from utils import BASE_DIR
from utils.mesh import SponjMesh
from utils.img import img_to_bytes
from utils.client import BaseClient

class SponjClient(BaseClient):
    def __init__(self):
        self.base_url = "http://34.127.57.13/api" # for development
        
        self.log_path = f"{BASE_DIR}/logs/sponj.log"
        self.endpoint = {
            "notify": f"{self.base_url}/task/notify",
            "csrf": f"{self.base_url}/user/csrf"
        }

        self.csrf_token, self.cookies = self.get_csrf()
    
    def format_params(self, **kwargs):
        return {
            key: value 
            for key, value in kwargs.items()
        }
    
    def check_size(self, img: Image) -> Image:
        if img.width > 512 or img.height > 512:
            h_to_w = img.height / img.width
            w_to_h = img.width / img.height

            if img.width > img.height:
                img = img.resize((512, int(512*h_to_w)))
            else:
                img = img.resize((int(512*w_to_h), 512))
        return img
        
    def send_img(self, uid: str, task_id: str, img: Image):
        img = self.check_size(img)

        params = self.format_params(uid=uid, task_id=task_id)
        body = {
            "type": "generate",
            "out": {
                "type": "img",
                "img": base64.b64encode(img_to_bytes(img)).decode('utf-8')
            }
        }

        # TODO: implement better dev setup

        self.fetch(self.endpoint['notify'], params, 'POST', json=body)
        
    def send_mesh(self, uid: str, task_id: str, mesh: SponjMesh):
        params = self.format_params(uid=uid, task_id=task_id)
        mesh_json = mesh.json()

        body = {
            "type": "generate",
            "out": {
                "type": "mesh",
                "mesh": {
                    "gif": mesh_json['gif'],
                    "glb": base64.b64encode(mesh_json['glb'].read()).decode('utf-8'),
                }
            }
        }

        # TODO: implement better dev setup

        self.fetch(self.endpoint['notify'], params, 'POST', json=body)

    def send_labels(self, uid: str, task_id: str, labels: List[int]):
        params = self.format_params(uid=uid, task_id=task_id)
        body = {
            "type": "edit",
            "out": {
                "type": "mesh",
                "labels": labels
            }
        }

        # TODO: implement better dev setup

        self.fetch(self.endpoint['notify'], params, 'POST', json=body)

    def get_csrf(self):
        response = self.get(self.endpoint['csrf'])
        self.log(f"[get_csrf] (json) >> {response.json()}")
        self.log(f"[get_csrf] (cookies) >> {response.cookies}")

        return response.cookies['csrftoken'], response.cookies
    
    def fetch(self, url: str, params: dict, method: {'POST', 'GET'}, param_key="params", **kwargs):
        method_func = None
        if method == 'POST':
            method_func = self.post
        elif method == 'GET':
            method_func = self.get
        else:
            raise ValueError(f"[SponjClient] >> Expected method to be 'POST' or 'GET' but got {method}")
        
        return method_func(
            url, 
            params=params, 
            param_key=param_key, 
            cookies=self.cookies,
            headers={
                "X-CSRFToken": self.csrf_token 
            },
            **kwargs, 
        )