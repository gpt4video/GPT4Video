import sys
import torch
from PIL import Image
from decord import VideoReader
import numpy as np
from pprint import pprint
from server.server_model_mplug_text import t2vzero

from models.text2video_zero.text2video_zero import Text2Video_Zero
t2vzero = Text2Video_Zero(device="cuda:3")

class SafetyChecker:
    def __init__(self, device):
        self.device = device
        self.replace_video = "./results_videos/17eade9d-70e6-4a8d-a1dc-ed474775730d.mp4"

    def run_safety_checker(self, model, video_path, thresh=16):
        video = self.load_video(video_path)
        video = [np.array(i) for i in video]
        video =  np.array(video)

        video, has_nsfw_concept = model.pipe.run_safety_checker(video, self.device, torch.float16, thresh=thresh)
        print(has_nsfw_concept)
        if sum(has_nsfw_concept)>4:
            return video_path, True
        else:
            return video_path, False
        
    def load_video(self, path, num_frames=10):
        vr = VideoReader(path, height=224, width=224)
        total_frames = len(vr)
        frame_indices = self.get_index(total_frames, num_frames)
        images_group = list()
        for frame_index in frame_indices:
            img = Image.fromarray(vr[frame_index].asnumpy()).convert('RGB')
            images_group.append(img)
        return images_group
    
    def get_index(self, num_frames, num_segments):
        seg_size = float(num_frames - 1) / num_segments
        start = int(seg_size / 2)
        offsets = np.array([
            start + int(np.round(seg_size * idx)) for idx in range(num_segments)
        ])
        return offsets
    
if __name__ == "__main__":
    ck = SafetyChecker('cuda:1')
    video_path = "test.mp4"
    path, nsfw = ck.run_safety_checker(video_path, thresh=6)
    print(path)
    print(nsfw)