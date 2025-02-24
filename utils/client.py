import json 
import requests 
from copy import deepcopy
from utils.logger import Logger

class BaseClient(Logger):
    def parse_params(self, params):
        return params
    
    def post(self, url, params=None, param_key="data", **kwargs):
        self.log(f"[post] >> sending request to {url}...")
        
        if params is not None:
            data = self.parse_params(params)
            kwargs[param_key] = data

        args_to_log = deepcopy(kwargs)
        if 'files' in kwargs: args_to_log['files'] = {}
        if 'json' in args_to_log:
            if 'out' in args_to_log['json']:
                args_to_log['json']['out'] = {}
                
            for arg_name, arg_value in [('img', ''), ('file', {}), ('mesh', {})]:
                if arg_name in args_to_log['json']:
                    args_to_log['json'][arg_name] = arg_value
            
            for i in range(len(args_to_log['json'].get('messages', []))):
                message = args_to_log['json']['messages'][i]
                for j in range(len(message.get('content', []))):
                    if 'image_url' in message['content'][j]:
                        args_to_log['json']['messages'][i]['content'][j]['image_url']['url'] = ""

        self.log(f"[post] (kwargs) >> {args_to_log}")
        response = requests.post(url, **kwargs)

        return self.validate(response)
    
    def get(self, url, params=None,**kwargs):
        self.log(f"[get] >> sending request to {url}...")
        
        if params is not None:
            for key, val in params.items():
                url += f"&{key}={val}"

        self.log(f"[get] (kwargs) >> {kwargs}")
        response = requests.get(url, **kwargs)
        return self.validate(response)

    def validate(self, response):
        if not response.ok:
            error_message = f"[Error] >> HTTP {response.status_code}: {response.text}"
            self.log(error_message)
            raise Exception(error_message)
        
        return response
            
    async def apost(self, url, result_url, params, **kwargs):
        pass
        # TODO: implement send request and watch for status != 202 and return generated img response


