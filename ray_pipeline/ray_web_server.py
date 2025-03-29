import argparse
import io
import asyncio

import base64
from PIL import Image
import uvicorn
from fastapi import FastAPI, WebSocket
import ray
from ray_pipeline_utils import QueueManager, SharedVar, timer
# TODO: Add max frames in the queue
app = FastAPI()

if not ray.is_initialized():
    ray.init(address="auto")
# Use try except to avoid re-creating the same actor (uvicorn may run the script multiple times)
post2web_queue = ray.get_actor("post2web_queue", namespace="matrix")    
current_action = ray.get_actor("current_action", namespace="matrix")
    
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("📡 Client connected to image stream")
    ray.get(current_action.set.remote(None))
    TICK_DURATION = 1  # 每次批量播放的总时长

    try:
        while True:
            # 1) 记录当前循环开始时间
            start_time = asyncio.get_event_loop().time()
            # print("⏰ Start time:", start_time)
            # 2) 一次性把队列里的所有图都拿出来
            img_list = await asyncio.to_thread(
                lambda: ray.get(post2web_queue.get_batch.remote())  # get batch 本身就是返回一个list，如果本身元素也是list，那么就是一个双层list
            )
            if img_list:
                print("📦 Got images from the queue")
                # print("📦 len(img_list): ", img_list)
            if img_list and isinstance(img_list[0], list):
                img_list = [item for sublist in img_list for item in sublist]
            if len(img_list) > 0:
                print(f"📦 Got {len(img_list)} images from the queue")
                # print(img_list)
            # 3) 过滤只要 PIL.Image
            valid_imgs = [img for img in img_list if isinstance(img, Image.Image)]
            if valid_imgs:
                # 计算“单张图显示时长”
                per_image_delay = TICK_DURATION / len(valid_imgs)
                print(
                    f"📦 This batch: {len(valid_imgs)} images, "
                    f"each gets {per_image_delay:.4f}s"
                )

                # 依次播放
                for idx, pil_img in enumerate(valid_imgs):
                    buffer = io.BytesIO()
                    pil_img.save(buffer, format="JPEG")
                    img_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

                    await websocket.send_text(img_b64)
                    # print(f"✅ Sent frame {idx+1}/{len(valid_imgs)} in this batch")

                    # 等待 per_image_delay
                    await asyncio.sleep(per_image_delay)
            else:
                # 如果没图，就啥也不做
                pass

            # 4) 确保一个循环正好是 0.25s
            elapsed = asyncio.get_event_loop().time() - start_time
            if elapsed < TICK_DURATION:
                await asyncio.sleep(TICK_DURATION - elapsed)

    except Exception as e:
        print(f"❌ WebSocket error: {e}")
    finally:
        print("🧹 WebSocket disconnected, cleaning up")
        
        
@app.websocket("/ws/keyboard")
async def keyboard_input(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            key = await websocket.receive_text()
            if key in ['w', 'a', 's', 'd', '']:
                ray.get(current_action.set.remote(key))
                print("Received key from front-end:", key)
    except Exception as e:
        print("Keyboard channel closed:", e)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start FastAPI WebSocket server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind the server to")
    parser.add_argument("--port", type=int, default=8001, help="Port to run the server on")

    args = parser.parse_args()
    uvicorn.run("ray_web_server:app", host=args.host, port=args.port)
    