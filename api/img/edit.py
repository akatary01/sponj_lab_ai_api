from PIL import Image

from utils import PROMPTS
from api.img import on_img
from api.schema import Geometry, Style
from api.vars import sd_client, openai_client 
from api.process import parse_geo, parse_style

def img_replace(path_id: str | None, geo: Geometry, style: Style, is_sketch: bool = False) -> Image:
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
        search_prompt = openai_client.caption(geo_img, PROMPTS['search'])
        img = sd_client.replace(geo_img, style_prompt, search_prompt)
    
    if path_id and img: on_img(path_id, img)
    return img 

def img_recolor(path_id: str | None, geo: Geometry, style: Style, is_sketch: bool = False) -> Image:
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
        select_prompt = openai_client.caption(geo_img, PROMPTS['search'])
        img = sd_client.recolor(geo_img, style_prompt, select_prompt)
    
    if path_id and img: on_img(path_id, img)
    return img 

def img_replace_bg(path_id: str | None, geo: Geometry, style: Style, is_sketch: bool = False) -> Image:
    img = None 
    geo_prompt, geo_img = parse_geo(geo)
    style_prompt, style_img = parse_style(style)

    if geo_img and is_sketch:
       geo_img = sd_client.sketch_to_img(geo_img, control_strength=geo.strength)

    if geo_prompt:
        geo_img = sd_client.text_to_img(geo_prompt, cfg_scale=geo.strength*10)

    if geo_img:
        if style_prompt:
            img = sd_client.replace_bg(geo_img, background_prompt=style_prompt, preserve_original_subject=geo.strength)
        
        elif style_img:
            img = sd_client.replace_bg(geo_img, background_reference=style_img, preserve_original_subject=geo.strength)
    
    if path_id and img: on_img(path_id, img)
    return img 

def img_remove_bg(path_id: str | None, geo: Geometry, is_sketch: bool = False) -> Image:
    img = None
    geo_prompt, geo_img = parse_geo(geo)

    if geo_img and is_sketch:
       geo_img = sd_client.sketch_to_img(geo_img, control_strength=geo.strength)

    if geo_prompt:
        geo_img = sd_client.text_to_img(geo_prompt, cfg_scale=geo.strength*10)

    if geo_img:
        img = sd_client.remove_bg(geo_img)
    
    if path_id and img: on_img(path_id, img)
    return img 