from clients.sd import SDClient
from clients.sponj import SponjClient
from clients.openai import OpenAIClient
from clients.tripo3d import Tripo3dClient

sd_client = SDClient()
sponj_client = SponjClient()
openai_client = OpenAIClient()
tripo3d_client = Tripo3dClient()

glb_task_to_path = {}
obj_task_to_path = {}
sponj_task_to_uid = {}

segmentation_queue = []