import base64
import numpy as np 
from PIL import Image
from io import BytesIO
from matplotlib.figure import Figure

def fig_to_img(fig: Figure, save_kwargs: dict = {}):
    io_buf = BytesIO()
    fig.savefig(io_buf, format='raw', **save_kwargs)
    io_buf.seek(0)
    img_arr = np.reshape(
        np.frombuffer(io_buf.getvalue(), dtype=np.uint8),
        newshape=(int(fig.bbox.bounds[3]), int(fig.bbox.bounds[2]), -1)
    )
    
    io_buf.close()
    return Image.fromarray(img_arr)

def img_to_bytes(img: Image):
    img_byte_arr = BytesIO()
    img.save(img_byte_arr, format='PNG')
    return img_byte_arr.getvalue()

def bytes_to_img(img_bytes: bytes):
    return Image.open(BytesIO(img_bytes))

def base64_to_img(base64_str: str):
    return Image.open(BytesIO(base64.decodebytes(bytes(base64_str, "utf-8"))))