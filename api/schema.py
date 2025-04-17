from pydantic import BaseModel 
from typing import List, Optional

class Auth(BaseModel):
    uid: str 
    
class Geometry(BaseModel):
    img: Optional[str] = None
    prompt: Optional[str] = None
    strength: Optional[float] = 0.7
    
class Style(BaseModel):
    img: Optional[str] = None
    prompt: Optional[str] = None
    strength: Optional[float] = 0.3

class AIRequest(Auth):
    styles: List[Style]
    geos: List[Geometry]
    is_sketch: Optional[bool]

class CaptionRequest(Auth):
    img: str

class SegmentRequest(Auth):
    faces: List[List[int]]
    vertices: List[List[float]]

class EditRequest(Auth):
    prompt: str
    faces: List[List[int]]
    vertices: List[List[float]]
    selected_vertices: List[List[float]]