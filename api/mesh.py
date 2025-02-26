import shutil
import zipfile
import sys

from utils.mesh import SponjMesh
from utils.thread import run_in_bg
from utils import download, rename
from utils.img import base64_to_img
from api.schema import Geometry, Style
from api.img.generate import img_generate
from api.process import parse_geo, parse_style
from api.vars import (
    sd_client, 
    sponj_task_to_uid,
    sponj_client,
    openai_client, 
    tripo3d_client, 
    glb_task_to_path, 
    obj_task_to_path, 
)
sys.path.append("/home/farazfaruqi/stable-point-aware-3d")
from gen_setup import *
from inference import generate_from_image

# Generates the mesh and sends the mesh to the user. 
def mesh_generate(sponj_task_id: str, geo: Geometry, style: Style, is_sketch: bool = False) -> str | None:
    geo_prompt, geo_img = parse_geo(geo)
    style_prompt, style_img = parse_style(style)

    if geo_prompt:
        pass
        # TODO: add support for generating mesh from text prompt
    
    elif geo_img:
        
        img = base64_to_img(geo.img)
        if img:
            run_in_bg(generate_from_image_wrapper, img, on_mesh, sponj_task_id, is_async=True)

    return sponj_task_id

def generate_from_image_wrapper(img, on_success, sponj_task_id):
    sponj_client.log(f"[mesh_generator] >> generating mesh...")
    glb_mesh_path, obj_mesh_path, _ = generate_from_image([img])
    sponj_client.log(f"[mesh_generator] >> generated mesh at {obj_mesh_path}")

    on_success(sponj_task_id, obj_mesh_path, glb_mesh_path)

def on_mesh(sponj_task_id: str, obj_mesh_path: str, mesh_glb_path: str):
    mesh = SponjMesh(obj_mesh_path, glb_path=mesh_glb_path)

    # mesh.get_largest_cc() 
    # mesh.decimate(targetfacenum=15000)

    gif_path = mesh_glb_path.replace(".glb", ".gif")
    mesh.generate_gif(gif_path, frames=12, save_kwargs={"facecolor": "#887e7e"})
    
    uid = sponj_task_to_uid[sponj_task_id]
    sponj_client.send_mesh(uid, sponj_task_id, mesh) # sending the mesh back to the backend. 
