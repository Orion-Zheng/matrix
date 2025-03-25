import time
from contextlib import contextmanager
import asyncio

import torch
import torch.distributed as dist
import ray
from ray.util.queue import Queue

@contextmanager
def timer(label="Block"):
    start_time = time.perf_counter()
    yield
    if torch.cuda.is_available():
        torch.cuda.synchronize()  # Ensures all CUDA operations are completed before measuring time
    end_time = time.perf_counter()
    
    if dist.is_initialized() and dist.get_rank() == 0:
        print(f"{label} took {end_time - start_time:.6f} seconds")
    else:
        print(f"{label} took {end_time - start_time:.6f} seconds")
        
@ray.remote(num_cpus=1, max_concurrency=3)
class QueueManager:
    def __init__(self, maxsize=0):
        self.queue = Queue(maxsize=maxsize)

    def put(self, item):
        self.queue.put(item)

    def get(self):
        print(f"[QueueManager] get() called")
        return self.queue.get()

    def get_batch(self):
        """一次性把队列里当前已有的所有元素全拿出来，返回列表。"""
        items = []
        # ray.util.queue.Queue 没有原生的 get_nowait() 方法，但如果 queue.empty() == False，
        # 那么 self.queue.get() 会立刻返回，而不会阻塞。
        while not self.queue.empty():
            items.append(self.queue.get())
        return items

    def print_queue(self):
        print(f"[QueueManager] Current queue: {self.queue}")


@ray.remote
class SharedVar:
    def __init__(self, value=None):
        self.value = value
        self._condition = asyncio.Condition()

    async def set(self, new_value):
        async with self._condition:
            self.value = new_value
            self._condition.notify_all()  # 通知等待中的 .get()

    async def get(self):
        async with self._condition:
            while self.value is None:
                await self._condition.wait()
            return self.value

# @ray.remote
# class SharedVar:
#     def __init__(self, value=None):
#         self.value = value

#     def set(self, new_value):
#         self.value = new_value

#     def get(self):
#         return self.value
           
# Usage:
# 1. Connect to an existing ray cluster
# ray.init(address='auto')  
# 2. Create a queue
# queue = QueueManager.options(namespace='the_name_space', name="the_queue_name").remote()
# 3. (For consumer) 
# queue = ray.get_actor("the_queue_name", namespace="the_name_space")  # Ensure the actor is started
# content = ray.get(queue.get.remote())  # this will block until a content is available
# 4. (For producer)
# queue = ray.get_actor("the_queue_name", namespace="the_name_space")
# ray.get(queue.put.remote(content))  # this is non-blocking