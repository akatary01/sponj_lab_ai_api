import asyncio
from threading import Thread

def run_in_bg(fn, *args, is_async=False, **kwargs):
    target = lambda: fn(*args, **kwargs)
    if is_async:
        target = lambda: asyncio.run(fn(*args, **kwargs))
        
    thread = Thread(target=target)
    thread.start()

    return thread