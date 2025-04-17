import shutil
import zipfile
import sys
from typing import List
import pickle

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

# Uncomment to Bring Spar
# sys.path.append("/home/farazfaruqi/stable-point-aware-3d")
# from gen_setup import *
# from inference import generate_from_image, generate_from_text

# # Trellis
sys.path.append("/home/farazfaruqi/TRELLIS/trellis_api_utils/")
from trellis_setup import *
from trellis_inference import generate_from_image, generate_from_text, edit_and_generate
#TODO: Faraz to add faraz_edit_mesh import 

# /home/farazfaruqi/TRELLIS/trellis_inference.py
# Generates the mesh and sends the mesh to the user. 
def mesh_generate(sponj_task_id: str, geo: Geometry, style: Style, is_sketch: bool = False) -> str:
    geo_prompt, geo_img = parse_geo(geo)
    style_prompt, style_img = parse_style(style)

    if geo_prompt:
        run_in_bg(generate_from_prompt_wrapper, geo_prompt, on_mesh, sponj_task_id, is_async=True)
    
    elif geo_img:
        
        img = base64_to_img(geo.img)
        if img:
            run_in_bg(generate_from_image_wrapper, img, on_mesh, sponj_task_id, is_async=True)

    return sponj_task_id

def mesh_edit(sponj_task_id: str, vertices: List[List[float]], faces: List[List[int]], prompt: str, selected_vertices: List[List[float]]) -> str:
    # Faraz's call here
    run_in_bg(faraz_edit_mesh_wrapper, vertices, faces, prompt, selected_vertices, on_mesh, sponj_task_id, is_async=True)
    return sponj_task_id

def faraz_edit_mesh_wrapper(vertices: List[List[float]], faces: List[List[int]], prompt: str, selected_vertices: List[List[float]], on_success, sponj_task_id):
    sponj_client.log(f"[faraz_edit_mesh_wrapper] >> editing mesh...")
    sponj_client.log(f"Vertices {len(vertices)}")
    sponj_client.log(f"Selected Vertices {len(selected_vertices)}")
    sponj_client.log(f"Editing Prompt {prompt}")
    with open("/home/farazfaruqi/TRELLIS/api_out/intermediate_editing/vertices.pkl", 'wb') as file:
        pickle.dump(vertices, file)
    with open("/home/farazfaruqi/TRELLIS/api_out/intermediate_editing/faces.pkl", 'wb') as file:
        pickle.dump(faces, file)
    with open("/home/farazfaruqi/TRELLIS/api_out/intermediate_editing/selected_vertices.pkl", 'wb') as file:
        pickle.dump(selected_vertices, file)

    sponj_client.log(f"[mesh_generator] >> editing mesh")
    glb_mesh_path, obj_mesh_path, _ = edit_and_generate(sponj_client, prompt)
    sponj_client.log(f"[mesh_generator] >> generated mesh at {obj_mesh_path}")
    on_success(sponj_task_id, obj_mesh_path, glb_mesh_path)


    #TODO: Faraz implement this method which takes in the verts, faces, selected_vertices and returns the obj and glb pathes
    #Note: the selected_vertices is an array of the actual verts not their indicies because indicies change all the time in frontend
    # glb_mesh_path, obj_mesh_path, _ = faraz_edit_mesh(vertices, faces, selected_vertices)
    #TODO: once you add the faraz_edit_mesh function then uncomment the below
    # sponj_client.log(f"[mesh_generator] >> generated mesh at {obj_mesh_path}")

    # on_success(sponj_task_id, obj_mesh_path, glb_mesh_path)

def generate_from_prompt_wrapper(prompt, on_success, sponj_task_id):
    sponj_client.log(f"[mesh_generator] >> generating mesh...")
    glb_mesh_path, obj_mesh_path, _ = generate_from_text(prompt)
    sponj_client.log(f"[mesh_generator] >> generated mesh at {obj_mesh_path}")

    on_success(sponj_task_id, obj_mesh_path, glb_mesh_path)

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
    try:
        mesh.generate_gif(gif_path, frames=12, save_kwargs={"facecolor": "#887e7e"})
    except Exception as error:
        sponj_client.log(f"[mesh saving] >> Gif generation failed with: {error}") 
    
    uid = sponj_task_to_uid[sponj_task_id]
    sponj_client.send_mesh(uid, sponj_task_id, mesh) # sending the mesh back to the backend. 
