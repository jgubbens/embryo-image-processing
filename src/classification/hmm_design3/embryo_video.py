import numpy as np
import tifffile
import torch
from torch.utils.data import Dataset

class embryo_video(Dataset):
    
    def __init__(self, yaml_data, vid, states, window_size=5, img_size=(224, 224)):
        self.yaml_data = yaml_data
        self.vid_path = vid
        self.vid = tifffile.imread(vid)
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

    def __len__(self):
        return self.get_frame_count()

    def __getitem__(self, idx):
        frame = idx + 1 # convert to 1 indexing to match labels
        window = self.get_frame_window(frame)
        window = self._preprocess(window)
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
                frames.append(np.zeros_like(self.vid[0]))
            else:
                frames.append(self.vid[i - 1])
        return np.stack(frames, axis=0)
    
    def _preprocess(self, window):
        window = window.astype(np.float32)
        for i in range(window.shape[0]):
            vmin, vmax = window[i].min(), window[i].max()
            if vmax > vmin:
                window[i] = (window[i] - vmin) / (vmax - vmin)
        tensor = torch.from_numpy(window).unsqueeze(0)
        tensor = torch.nn.functional.interpolate(
            tensor, size=self.img_size, mode='bilinear', align_corners=False
        ).squeeze(0)
        return tensor