import os 
import math
import json
import requests
import numpy as np
from io import BytesIO
from pathlib import Path
from threading import Thread
import matplotlib.colors as mcolors
from matplotlib import pyplot as plt

BASE_DIR = Path(__file__).resolve().parent.parent

COLORS = mcolors.TABLEAU_COLORS
COLORS_KEYS = list(COLORS.keys())

TMP_DIR = f"{BASE_DIR}/tmp"

with open(f"{BASE_DIR}/keys.json", "r") as config:
  API_KEYS = json.loads(config.read())

with open(f"{BASE_DIR}/assets/prompts.json", "r") as prompts:
  PROMPTS = json.loads(prompts.read())

def display(imgs, r, c=-1, w=3, h=3, scatter=False, subplot_kw={}, show_axis=False):
  """ Display images in a grid """
  n = len(imgs)
  if c == -1: c = math.ceil(n / r)

  fig, axes = plt.subplots(r, c, figsize=(w*c, h), subplot_kw=subplot_kw)

  for i in range(r):
    for j in range(c):
      if r > 1: 
        if c > 1: axe = axes[i][j]; z = i*r + j
        else: axe = axes[i]; z = i
      elif c > 1: axe = axes[j]; z = j
      else: axe = axes; z = 0

      if z < n:
        if scatter:
          xx, yy, zz = imgs[z]['img']
          axe.scatter(xx, yy, zz, c=COLORS[COLORS_KEYS[i % len(COLORS)]])
        else:
          axe.imshow(np.array(imgs[z]['img']))
        axe.title.set_text((imgs[z]['label']))
        if not show_axis: axe.axis("off")
  plt.show()

def bytes_from_url(url: str):
  response = requests.get(url)

  return BytesIO(response.content)

def download(url: str, id: str, ext: str):
  response = requests.get(url)

  path_dir = f"{TMP_DIR}/{id}"
  if not Path(path_dir).exists():
    Path(path_dir).mkdir(parents=True, exist_ok=True)
  
  file_path = f"{path_dir}/{id}.{ext}"
  if response.status_code == 200:
    with open(file_path, 'wb') as file:
      for chunk in response.iter_content(chunk_size=8192):  # Read in chunks
        file.write(chunk)
    
  return path_dir, file_path

def rename(dir_path: str, ext: str, new_name: str, k = 1):
  renamed_files = []

  i = 0
  for root, _, files in os.walk(dir_path):
    for file in files:
      if file.endswith(ext):
        old_path = os.path.join(root, file)

        if k <= 1:
          new_name = f"{new_name}.{ext}"
        else:
          new_name = f"{new_name}_{i}.{ext}"

        new_path = os.path.join(root, new_name)

        os.rename(old_path, new_path)
        renamed_files.append(new_path)

        i += 1
        if i > k: return renamed_files

  return renamed_files
