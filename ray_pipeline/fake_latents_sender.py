import time
import torch
import ray

if __name__ == "__main__":
    ray.init(address='auto')  # connect to ray cluster
    dit2vae_queue = ray.get_actor("dit2vae_queue", namespace='matrix')  # get the named queue manager
    current_action = ray.get_actor("current_action", namespace='matrix') 
    latents = torch.load("/workspace/matrix/latents_100.pt", map_location=torch.device('cpu'))
    # latents = latents[:, :10]  # decode 100 latents at a time will cause OOM on 4090
    # ray.get(queue.put.remote(latents))
    
    # a for loop to send the latents to the queue one by one(simulate the DiT process)
    for i in range(latents.shape[1]):
        latent = latents[:, i:i+1]
        print(f"Sending latent {i+1}/{latents.shape[1]}")
        print("shape of latent: ", latent.shape)
        ray.get(dit2vae_queue.put.remote(latent))
        
        time.sleep(0.25)
        print("Current Action: ", ray.get(current_action.get.remote()))