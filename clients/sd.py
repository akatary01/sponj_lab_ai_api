from PIL import Image
from requests import Response
from utils.client import BaseClient
from utils import API_KEYS, PROMPTS, BASE_DIR
from utils.img import img_to_bytes, bytes_to_img

SD_API_KEY = API_KEYS['SD_API_KEY']
default_kwargs = {
    'output_format': "png",
    'control_strength': 0.7, 
    'negative_prompt': PROMPTS['negative'], 
}

class SDClient(BaseClient):
    def __init__(self):
        super().__init__()
        self.api_key = SD_API_KEY
        self.headers = {
            'Accept': "image/*",
            'Authorization': f"Bearer {self.api_key}"
        }

        self.base_url = "https://api.stability.ai/v2beta/stable-image"

        self.endpoints = {
            "edit": {
                "inpaint": f"{self.base_url}/edit/inpaint",
                "replace": f"{self.base_url}/edit/search-and-replace",
                "recolor": f"{self.base_url}/edit/search-and-recolor",
                "remove_bg": f"{self.base_url}/edit/remove-background",
                "replace_bg": f"{self.base_url}/edit/replace-background-and-relight",
            },
            "control": {
                "style": f"{self.base_url}/control/style",
                "sketch": f"{self.base_url}/control/sketch",
                "structure": f"{self.base_url}/control/structure",
            },
            "generate": {
                "text": f"{self.base_url}/generate/sd3",
                "image": f"{self.base_url}/generate/sd3"
            }
        }

        self.log_path = f"{BASE_DIR}/logs/sd.log"
    
    # generate
    def text_to_img(self, prompt: str, **kwargs) -> Image:
        params = {
            "model": "sd3.5-large",
            "prompt" : self.process_prompt(prompt),
        }
        kwargs['files'] = {'none': None}
        url = self.endpoints["generate"]["text"]
        return self(url, params, default_kwargs={}, **kwargs)

    def img_to_img(self, img: Image, prompt: str, **kwargs) -> Image:
        params = {
            "model": "sd3.5-large",
            "mode": "image-to-image",
            "prompt" : self.process_prompt(prompt),
        }
        kwargs['files'] = {"image" : img_to_bytes(img)}
        url = self.endpoints["generate"]["text"]
        return self(url, params, default_kwargs={}, **kwargs)
    
    # edit 
    def inpaint(self, img: Image, prompt: str, mask: Image = None, **kwargs) -> Image:
        img = self.check_size(img)
        
        params = {"prompt" : self.process_prompt(prompt)}
        kwargs['files'] = {"image" : img_to_bytes(img)}
        if mask:
            kwargs['files']['mask'] = img_to_bytes(mask)

        url = self.endpoints["edit"]["inpaint"]
        return self(url, params, **kwargs)
    
    def replace(self, img: Image, prompt: str, search_prompt: str, **kwargs) -> Image:
        img = self.check_size(img)

        params = {"prompt" : self.process_prompt(prompt), "search_prompt" : search_prompt}
        kwargs['files'] = {"image" : img_to_bytes(img)}

        url = self.endpoints["edit"]["replace"]
        return self(url, params, **kwargs)

    def recolor(self, img: Image, prompt: str, select_prompt: str, **kwargs) -> Image:
        img = self.check_size(img)

        params = {"prompt" : self.process_prompt(prompt), "select_prompt" : select_prompt}
        kwargs['files'] = {"image" : img_to_bytes(img)}

        url = self.endpoints["edit"]["recolor"]
        return self(url, params, **kwargs)

    def replace_bg(self, subject_img: Image, background_reference = None, background_prompt = None, **kwargs) -> Image:
        subject_img = self.check_size(subject_img)

        params = {}
        kwargs['files'] = {"subject_image" : img_to_bytes(subject_img)}

        if background_reference:
            kwargs['files']['background_reference'] = img_to_bytes(background_reference)
        if background_prompt:
            params['background_prompt'] = background_prompt

        url = self.endpoints["edit"]["replace_bg"]
        return self(url, params, **kwargs)

    def remove_bg(self, img: Image, **kwargs) -> Image:
        img = self.check_size(img)

        params = {}
        kwargs['files'] = {"image" : img_to_bytes(img)}

        url = self.endpoints["edit"]["remove_bg"]
        return self(url, params, **kwargs)
    
    # control
    def structure(self, img: Image, prompt: str, **kwargs) -> Image:
        img = self.check_size(img)

        params = {"prompt" : self.process_prompt(prompt)}
        kwargs['files'] = {"image" : img_to_bytes(img)}

        url = self.endpoints["control"]["structure"]
        return self(url, params, **kwargs)
    
    def sketch_to_img(self, img: Image, prompt = PROMPTS['sketchToImg'], **kwargs) -> Image:
        img = self.check_size(img)

        params = {"prompt" : self.process_prompt(prompt)}
        kwargs['files'] = {"image" : img_to_bytes(img)}

        url = self.endpoints["control"]["sketch"]
        return self(url, params, **kwargs)
    
    def style(self, img: Image, prompt: str, **kwargs) -> Image:
        img = self.check_size(img)

        params = {"prompt" : self.process_prompt(prompt)}
        kwargs['files'] = {"image" : img_to_bytes(img)}

        url = self.endpoints["control"]["style"]
        return self(url, params, **kwargs)
    
    def check_size(self, img: Image) -> Image:
        if img.width > 1024 or img.height > 1024:
            h_to_w = img.height / img.width
            w_to_h = img.width / img.height

            if img.width > img.height:
                img = img.resize((1024, int(1024*h_to_w)))
            else:
                img = img.resize((int(1024*w_to_h), 1024))
        return img

    def process_prompt(self, prompt: str) -> str:
        # return f"{prompt}. {PROMPTS['sd']['systemPrompt']}"
        return prompt
    
    def __call__(self, url: str, params, default_kwargs = default_kwargs, **kwargs) -> Image:
        post_kwargs = {}
        if 'files' in kwargs:
            post_kwargs['files'] = kwargs.pop('files')
        
        params = {
            **params,
            **default_kwargs,
            **kwargs
        }

        response = self.post(url, params=params, headers=self.headers, **post_kwargs)
        return self.decode(response)

    def decode(self, response: Response) -> Image:
        return bytes_to_img(response.content)
