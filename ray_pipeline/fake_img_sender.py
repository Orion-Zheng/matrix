import time
import random
from PIL import Image

import ray

ray.init(address="auto")
post2web_queue = ray.get_actor("post2web_queue", namespace="matrix")

batch_size = 5
send_white = True  # 控制交替黑白

while True:
    image_batch = []

    for _ in range(batch_size):
        if send_white:
            print("Creating white image")
            img = Image.new("RGB", (400, 300), (255, 255, 255))
        else:
            print("Creating black image")
            img = Image.new("RGB", (400, 300), (0, 0, 0))

        image_batch.append(img)
        send_white = not send_white  # 每张图交替黑白

    print(f"Sending batch of {len(image_batch)} images")
    for image in image_batch:
        post2web_queue.put.remote(image)
    # queue.put.remote(image_batch)

    # Sleep 随机时间在 0.1s 到 0.3s 之间
    sleep_time = random.uniform(0.1, 0.3)
    print(f"Sleeping for {sleep_time:.3f} seconds\n")
    time.sleep(sleep_time)