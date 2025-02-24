from PIL import Image

from api.vars import (
    sponj_task_to_uid,
    sponj_client, 
)

def on_img(path_id: str, img: Image):
    uid = sponj_task_to_uid[path_id]
    sponj_client.send_img(uid, path_id, img)

    del sponj_task_to_uid[path_id]