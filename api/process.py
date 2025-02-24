from PIL import Image
from typing import Tuple
from utils.img import base64_to_img
from api.schema import Geometry, Style
from api.vars import (
    openai_client 
)

def parse_geo(geo: Geometry | None):
    return parse_data(geo)

def parse_style(style: Style | None):
    return parse_data(style)

def parse_data(data: Geometry | Style | None):
    prompt, img = None, None

    if data: 
        prompt, img = data.prompt, data.img
    
    if img: img = base64_to_img(data.img)
    
    return prompt, img
