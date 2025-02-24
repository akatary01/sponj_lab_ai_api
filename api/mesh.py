import shutil
import zipfile

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

def mesh_generate(sponj_task_id: str, geo: Geometry, style: Style, is_sketch: bool = False) -> str | None:
    mesh_id = None
    geo_prompt, geo_img = parse_geo(geo)
    style_prompt, style_img = parse_style(style)

    if geo_prompt:
        if style_img:
            style_prompt = openai_client.caption(style_img)
        
        mesh_id = tripo3d_client.text_to_mesh(prompt=f"{geo_prompt}. With the following style: {style_prompt}")
    
    elif geo_img:
        if style_img or style_prompt or is_sketch:
            img = img_generate(None, geo, style, is_sketch)
        else:
            img = base64_to_img(geo.img)
        if img:
            mesh_id = tripo3d_client.img_to_mesh(img)
            glb_task_to_path[mesh_id] = sponj_task_id

    run_in_bg(tripo3d_client.watch, on_success=on_mesh, is_async=True, task_id=mesh_id)
    return mesh_id  

def on_mesh(task_id: str, mesh_url: str, ext: {'glb', 'obj'}, type: {'text_to_model', 'image_to_model', 'convert_model'}):
    if ext == "glb":
        if task_id not in glb_task_to_path:
            print(f"glb task {task_id} not in {glb_task_to_path}")
            return 

        sponj_task_id = glb_task_to_path[task_id]
        _, glb_path = download(mesh_url, sponj_task_id, ext)

        obj_task_id = tripo3d_client.convert_mesh(task_id)
        
        del glb_task_to_path[task_id]
        glb_task_to_path[obj_task_id] = glb_path
        obj_task_to_path[obj_task_id] = sponj_task_id
        
        run_in_bg(tripo3d_client.watch, on_success=on_mesh, is_async=True, task_id=obj_task_id)
    if ext == "obj":
        if task_id not in obj_task_to_path:
            print(f"obj task {task_id} not in {obj_task_to_path}")
            return 
        
        sponj_task_id = obj_task_to_path[task_id]
        obj_dir, zip_path = download(mesh_url, sponj_task_id, "zip")

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(obj_dir)

        obj_path = rename(obj_dir, "obj", sponj_task_id)[0]
        obj_task_to_path[task_id] = obj_path
        
        
        glb_path = glb_task_to_path[task_id]
        mesh = SponjMesh(obj_path, glb_path=glb_path)

        # mesh.get_largest_cc() 
        # mesh.decimate(targetfacenum=15000)

        gif_path = obj_path.replace(".obj", ".gif")
        mesh.generate_gif(gif_path, frames=12, save_kwargs={"facecolor": "#887e7e"})
        
        uid = sponj_task_to_uid[sponj_task_id]
        sponj_client.send_mesh(uid, sponj_task_id, mesh)
        
        # cleanup
        shutil.rmtree(obj_dir) 
        del glb_task_to_path[task_id]
        del obj_task_to_path[task_id]
