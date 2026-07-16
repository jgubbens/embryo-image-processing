import numpy as np
import tifffile
import torch
from torch.utils.data import Dataset

class embryo_video(Dataset):
    
    def __init__(self, yaml_data, vid, states, window_size=5, img_size=(224, 224)):
        self.yaml_data = yaml_data
        self.vid_path = vid

        # Preprocess video
        vid_np = tifffile.imread(vid)
        vid_tensor = torch.from_numpy(vid_np).float()
        # Normalize per-frame
        vmin = vid_tensor.amin(dim=(-2,-1), keepdim=True)
        vmax = vid_tensor.amax(dim=(-2,-1), keepdim=True)
        vid_tensor = (vid_tensor - vmin) / (vmax - vmin + 1e-6)
        # Resize once
        vid_tensor = torch.nn.functional.interpolate(
            vid_tensor.unsqueeze(1), size=img_size, mode='bilinear', align_corners=False
        ).squeeze(1).half()
        self.vid = vid_tensor
        self.vid.share_memory_()
        
        self.STATES = states
        self.window_size = window_size
        self.img_size = img_size
        self.time_between_frames = yaml_data.get('time_between_frames')
        self.earliest = next(k for k in yaml_data if k in self.STATES)
        self.frame_labels = self._derive_labels()

    def _derive_labels(self):
        transitions = {state: frame for state, frame in self.yaml_data.items() if state in self.STATES}
        sorted_t = sorted(transitions.items(), key=lambda x: x[1])
        labels = {}
        for frame in range(1, self.get_frame_count() + 1):
            state = sorted_t[0][0]
            for stage, start_frame in sorted_t:
                if frame >= start_frame:
                    state = stage
            labels[frame] = self.STATES.index(state)
        return labels

    def get_labels(self):
        return [self.frame_labels[i + self.window_size] for i in range(len(self))]

    def __len__(self):
        return max(0, self.get_frame_count() - self.window_size + 1)

    def __getitem__(self, idx):
        frame = idx + self.window_size # last (1-indexed) frame of the window
        window = self.get_frame_window(frame)
        window = window.float()
        return window, self.frame_labels[frame]

    def get_frame_count(self):
        return self.vid.shape[0]

    def get_frame(self, frame):
        if self.vid.shape[0] < frame or frame < 1:
            raise IndexError(f'Attempted to retrieve frame {frame} in {self.vid}')
        return self.vid[frame - 1]
    
    def get_frame_window(self, frame):
        if frame < 1 or frame > self.vid.shape[0]:
            raise IndexError(f'Frame {frame} out of range (1-{self.vid.shape[0]})')
        frames = []
        for i in range(frame - self.window_size + 1, frame + 1):
            if i < 1:
                frames.append(torch.zeros_like(self.vid[0]))
            else:
                frames.append(self.vid[i - 1])
        return torch.stack(frames, axis=0)
    