import time
import ray
from ray_pipeline_utils import QueueManager, SharedVar, timer

if __name__ == "__main__":
    ray.init(address='auto')  # connect to ray cluster
    
    current_action = SharedVar.options(namespace='matrix', name="current_action").remote(None)  # Create a queue for accepting latest action commands
    
    dit2vae_queue = QueueManager.options(namespace='matrix', name="dit2vae_queue").remote()  # DiT --> VAE
    vae2post_queue = QueueManager.options(namespace='matrix', name="vae2post_queue").remote()  # VAE --> Postprocessing
    post2web_queue = QueueManager.options(namespace='matrix', name="post2web_queue").remote()  # Postprocessing --> Web
    actors = ray.util.list_named_actors(all_namespaces=True)
    print("Actors in all namespaces: ", [actor_name for actor_name in actors])
    
    # === IMPORTANT NOTE ===
    # These queues and shared variables are Ray actors. If the main process exits right after creating them,
    # and no other processes are using them or holding references, Ray may automatically clean them up.
    # To keep these actors alive and available for other components (e.g., workers, web server),
    # you must keep the main process running or create them inside a persistent service.
    
    try:
        print("Queues are running. Press Ctrl+C to exit.")
        while True:
            time.sleep(10)  # Keep the process alive to prevent actors from being destroyed
    except KeyboardInterrupt:
        print("Shutting down.")