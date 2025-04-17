import json
import uuid
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from utils.thread import run_in_bg
from api.segment import segment_mesh
from api.img.control import img_structure, img_style
from api.mesh import mesh_generate, on_mesh, mesh_edit
from api.img.generate import img_generate, img_caption, style_generate
from api.schema import AIRequest, SegmentRequest, CaptionRequest, EditRequest
from api.img.edit import img_recolor, img_remove_bg, img_replace, img_replace_bg
from api.vars import tripo3d_client, sponj_task_to_uid, segmentation_queue, sponj_client

BASE_DIR = Path(__file__).resolve().parent
with open(f"{BASE_DIR}/config.json", "r") as config:
    CONFIG = json.loads(config.read())

# run_in_bg(tripo3d_client.watch, on_success=on_mesh, is_async=True)

api = FastAPI(
    docs_url="/ai/docs",
    title="Sponj AI API",
)

api.add_middleware(
    CORSMiddleware,
    **CONFIG
)

# mesh endpoints
@api.post("/ai/mesh/generate")
def ai_mesh_generate(mesh_info: AIRequest):
    uid = mesh_info.uid
    sponj_task_id = str(uuid.uuid4())

    sponj_task_to_uid[sponj_task_id] = uid
    style = style_generate(mesh_info.styles)
    run_in_bg(mesh_generate(sponj_task_id, mesh_info.geos[0], style, mesh_info.is_sketch))
    sponj_client.log(f"[ai_mesh_generate] >> generated mesh with task_id: {sponj_task_id}")

    return {"task_id": sponj_task_id}

@api.post("/ai/mesh/segment")
def ai_mesh_segment(mesh: SegmentRequest):
    uid = mesh.uid
    sponj_task_id = str(uuid.uuid4())

    faces = mesh.faces
    vertices = mesh.vertices

    sponj_task_to_uid[sponj_task_id] = uid
    segmentation_queue.append(sponj_task_id)
    run_in_bg(segment_mesh, sponj_task_id, vertices, faces)

    return {"task_id": sponj_task_id}


@api.post("/ai/mesh/edit")
def ai_mesh_segment(mesh: EditRequest):
    uid = mesh.uid
    sponj_task_id = str(uuid.uuid4())

    faces = mesh.faces
    prompt = mesh.prompt
    vertices = mesh.vertices
    selected_vertices = mesh.selected_vertices

    sponj_task_to_uid[sponj_task_id] = uid
    sponj_client.log(f"[ai_mesh_edit] >> edited mesh with task_id: {sponj_task_id}")
    run_in_bg(mesh_edit, sponj_task_id, vertices, faces, prompt, selected_vertices)

    return {"task_id": sponj_task_id}

# img endpoints
@api.post("/ai/img/caption")
def ai_img_caption(caption_info: CaptionRequest):
    uid = caption_info.uid
    
    caption = img_caption(caption_info.img)
    return {"caption": caption}

@api.post("/ai/img/generate")
def ai_img_generate(img_info: AIRequest):
    uid = img_info.uid
    sponj_task_id = str(uuid.uuid4())

    sponj_task_to_uid[sponj_task_id] = uid
    style = style_generate(img_info.styles)

    run_in_bg(img_generate, sponj_task_id, img_info.geos[0], style, img_info.is_sketch)

    return {"task_id": sponj_task_id}

@api.post("/ai/img/replace")
def ai_img_replace(img_info: AIRequest):
    uid = img_info.uid
    sponj_task_id = str(uuid.uuid4())

    sponj_task_to_uid[sponj_task_id] = uid
    style = style_generate(img_info.styles)

    run_in_bg(img_replace, sponj_task_id, img_info.geos[0], style, img_info.is_sketch)

    return {"task_id": sponj_task_id}

@api.post("/ai/img/recolor")
def ai_img_recolor(img_info: AIRequest):
    uid = img_info.uid
    sponj_task_id = str(uuid.uuid4())

    sponj_task_to_uid[sponj_task_id] = uid
    style = style_generate(img_info.styles)
    run_in_bg(img_recolor, sponj_task_id, img_info.geos[0], style, img_info.is_sketch)

    return {"task_id": sponj_task_id}
       
@api.post("/ai/img/removeBackground")
def ai_img_remove_bg(img_info: AIRequest):
    uid = img_info.uid
    sponj_task_id = str(uuid.uuid4())

    sponj_task_to_uid[sponj_task_id] = uid
    run_in_bg(img_remove_bg, sponj_task_id, img_info.geos[0], img_info.is_sketch)

    return {"task_id": sponj_task_id}
    
@api.post("/ai/img/replaceBackground")
def ai_img_replace_bg(img_info: AIRequest):
    uid = img_info.uid
    sponj_task_id = str(uuid.uuid4())

    sponj_task_to_uid[sponj_task_id] = uid
    style = style_generate(img_info.styles)
    run_in_bg(img_replace_bg, sponj_task_id, img_info.geos[0], style, img_info.is_sketch)

    return {"task_id": sponj_task_id}

@api.post("/ai/img/structure")
def ai_img_structure(img_info: AIRequest):
    uid = img_info.uid
    sponj_task_id = str(uuid.uuid4())

    sponj_task_to_uid[sponj_task_id] = uid
    style = style_generate(img_info.styles)
    run_in_bg(img_structure, sponj_task_id, img_info.geos[0], style, img_info.is_sketch)

    return {"task_id": sponj_task_id}

@api.post("/ai/img/style")
def ai_img_style(img_info: AIRequest):
    uid = img_info.uid
    sponj_task_id = str(uuid.uuid4())

    sponj_task_to_uid[sponj_task_id] = uid
    style = style_generate(img_info.styles)
    run_in_bg(img_style, sponj_task_id, img_info.geos[0], style, img_info.is_sketch)

    return {"task_id": sponj_task_id}