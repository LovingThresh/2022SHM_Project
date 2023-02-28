import os
from mmflow.apis import init_model, inference_model
from mmflow.utils import register_all_modules

register_all_modules()

flow_model = init_model(config=r"V:\2022SHM_Project\displacement_measurement\configs\raft\raft_8xb2_100k_flyingchairs-368x496.py")

images_path = 'V:/2022SHM-dataset/project2/case1_images'
images = os.listdir(images_path)

images_1 = [os.path.join(images_path, i) for i in images[:100][:-1]]
images_2 = [os.path.join(images_path, i) for i in images[:100][1:]]

assert len(images_1) == len(images_2)

results = inference_model(flow_model, img1s=images_1, img2s=images_2)



