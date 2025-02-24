from typing import List
from utils.segment import SegmentedSponjMesh
from api.vars import sponj_client, sponj_task_to_uid, segmentation_queue

def segment_mesh(path_id: str, vertices: List[List[float]], faces: List[List[int]], k=3) -> str | None:
    while True:
        if segmentation_queue[0] == path_id:
            break

    mesh = SegmentedSponjMesh(faces=faces, vertices=vertices)

    labels, n, duration = mesh.segment(k=k) # k is the collapse factor, allows segmentation to run in O((n/k)^2log(n/k) + n) instead of O(n^2log(n))
    segmentation_queue.pop(0)
    
    on_segmented_mesh(path_id, labels)

def on_segmented_mesh(path_id: str, labels: List[int]):
    uid = sponj_task_to_uid[path_id]
    sponj_client.send_labels(uid, path_id, labels)