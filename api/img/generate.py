from PIL import Image

import base64
from typing import List
from utils import PROMPTS
from api.img import on_img
from api.schema import Geometry, Style
from api.process import parse_geo, parse_style
from utils.img import base64_to_img, img_to_bytes
from api.vars import (
    sd_client, 
    sponj_client, 
    openai_client,
    sponj_task_to_uid
)

def style_generate(styles: List[Style]):
    n = len(styles)
    if n == 0: return 
    if n == 1: return styles[0]
    else:
        style = Style(strength=0.5)
        prompt = Style(prompt="", strength=0.5)

        for style_i in styles:
            if style_i.prompt:
                prompt.prompt += f", {style_i.prompt}"
            elif style_i.img:
                style_i.strength = 0.5
                img = img_generate(None, style, style_i)
                style = Style(img=base64.b64encode(img_to_bytes(img)).decode('utf-8'))
        
        if prompt.prompt and style.img:
            if len(prompt.prompt) > 150:
                prompt.prompt = openai_client.respond(f"Rephrase the following in 150 characters: {prompt.prompt}")
            img = img_generate(None, style, prompt)
            return Style(img=base64.b64encode(img_to_bytes(img)).decode('utf-8'))
        elif prompt: 
            return prompt
        return style

def img_generate(path_id: str | None, geo: Geometry, style: Style, is_sketch: bool = False) -> Image:
    sponj_client.log(f"[img_generator] >> generating img (is_sketch: {is_sketch})...")

    img = None
    geo_prompt, geo_img = parse_geo(geo)
    style_prompt, style_img = parse_style(style)

    if geo_img and is_sketch:
       geo_caption = openai_client.caption(geo_img)
       geo_img = sd_client.sketch_to_img(geo_img, control_strength=geo.strength)

       img = sd_client.structure(geo_img, geo_caption, control_strength=geo.strength)

    if geo_prompt:
        if style_img:
            img = sd_client.style(style_img, geo_prompt, fidelity=style.strength)
    
        elif style_prompt:
            img = sd_client.text_to_img(f"{geo_prompt}. With following style: {style_prompt}", cfg_scale=geo.strength*10)        
        
        else:
            img = sd_client.text_to_img(geo_prompt, cfg_scale=geo.strength*10)

    elif geo_img:
        if style_img:
            prompt = openai_client.caption(geo_img)
            img = sd_client.style(style_img, prompt, fidelity=style.strength)

        elif style_prompt:
            img = sd_client.img_to_img(geo_img, style_prompt, strength=style.strength, cfg_scale=geo.strength*10)
       
        elif not is_sketch:
            img = sd_client.img_to_img(geo_img, PROMPTS['imgToImg'], strength=1 - geo.strength, cfg_scale=geo.strength*10)
    
    elif style_prompt:
        img = sd_client.text_to_img(style_prompt, cfg_scale=style.strength*10)
    
    elif style_img:
        img = style_img
    
    if path_id and img: on_img(path_id, img)
    return img 

def img_caption(base64_str: str):
    img = base64_to_img(base64_str)
    return openai_client.caption(img)
