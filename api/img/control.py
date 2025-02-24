from PIL import Image

from utils import PROMPTS
from api.img import on_img
from api.schema import Geometry, Style
from api.vars import sd_client, openai_client 
from api.process import parse_geo, parse_style

def img_structure(path_id: str | None, geo: Geometry, style: Style, is_sketch: bool = False) -> Image:
    img = None
    geo_prompt, geo_img = parse_geo(geo)
    style_prompt, style_img = parse_style(style)

    if geo_img and is_sketch:
       geo_img = sd_client.sketch_to_img(geo_img, control_strength=geo.strength)

    if style_img:
        style_prompt = openai_client.caption(style_img)

    if geo_prompt:
        geo_img = sd_client.text_to_img(geo_prompt, cfg_scale=geo.strength*10)

    if geo_img and style_prompt:
        img = sd_client.structure(geo_img, style_prompt, control_strength=geo.strength)
    
    if path_id and img: on_img(path_id, img)
    return img 

def img_style(path_id: str | None, geo: Geometry, style: Style, is_sketch: bool = False) -> Image:
    img = None
    geo_prompt, geo_img = parse_geo(geo)
    style_prompt, style_img = parse_style(style)

    if geo_img:
        geo_prompt = openai_client.caption(geo_img)

    if style_prompt:
        style_img = sd_client.text_to_img(style_prompt, cfg_scale=style.strength*10)

    if style_img and geo_prompt:
        img = sd_client.style(style_img, geo_prompt, fidelity=style.strength)
    
    if path_id and img: on_img(path_id, img)
    return img 
