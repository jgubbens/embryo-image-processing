import json
import numpy as np
import torch

from classification.cnn_classifier import cnn_classifier
from classification.lstm_classifier import lstm_classifier


class HMM_Predictor:

    STATES = ['undetectable', 'NC9', 'NC9M', 'NC10', 'NC10M', 'NC11', 'NC11M', 'NC12', 'NC12M', 'NC13', 'NC13M', 'NC14+']

    def __init__(self, device, model_info_path, time_between_frames):
        self.device = device
        self.n_states = len(self.STATES)
        self.load_pretrained_models(model_info_path)
        self.initialize_live_prediction(time_between_frames)

    def load_pretrained_models(self, model_info_path):
        with open(model_info_path) as f:
            info = json.load(f)
        self.window_size = info['window_size']
        self.lstm_module = info['lstm_module']
        self.live_img_size = tuple(info.get('img_size', [800, 800]))
        self.cnn = cnn_classifier(self.device, self.window_size, self.STATES)
        self.cnn.model.eval()
        self.cnn.load_from_path(info['cnn_model_path'])

        if self.lstm_module:
            self.cnn.remove_head()
            self.lstm = lstm_classifier(self.cnn.get_hidden_size(), self.device, self.STATES, self.cnn)
            self.lstm.load_from_path(info['lstm_model_path'])

        self._load_extra_model_info(info)

    def _load_extra_model_info(self, info):
        # Override in subclasses to load extra state
        pass

    def initialize_live_prediction(self, time_between_frames):
        self.time_between_frames = time_between_frames
        self.frame_idx = 0
        self.predictions = []
        self.frame_buffer = []
        self.lstm_hidden_state = None
        self._initialize_extra_live_state()

    def _initialize_extra_live_state(self):
        # Override in subclasses to initialize extra state
        pass

    def _preprocess_window(self):
        window = np.stack(self.frame_buffer, axis=0).astype(np.float32)
        for i in range(window.shape[0]):
            vmin, vmax = window[i].min(), window[i].max()
            if vmax > vmin:
                window[i] = (window[i] - vmin) / (vmax - vmin)
        tensor = torch.from_numpy(window).unsqueeze(0)
        tensor = torch.nn.functional.interpolate(
            tensor, size=self.live_img_size, mode='bilinear', align_corners=False
        )
        return tensor

    def _select_state(self, model_probs):
        # Override in subclasses to turn model_probs into a state prediction
        pass

    def get_current_state(self):
        if not self.predictions:
            return None
        return self.STATES[self.predictions[-1]]

    def predict_frame(self, frame):
        frame_np = frame.numpy() if isinstance(frame, torch.Tensor) else frame

        self.frame_buffer.append(frame_np)
        if len(self.frame_buffer) > self.window_size:
            self.frame_buffer.pop(0)

        if len(self.frame_buffer) < self.window_size:
            print(f'Frame {self.frame_idx}:\tBuffering ({len(self.frame_buffer)}/{self.window_size})')
            self.frame_idx += 1
            return

        with torch.no_grad():
            x = self._preprocess_window().to(self.device)
            if self.lstm_module:
                feature = self.cnn.model(x)
                model_probs, self.lstm_hidden_state = self.lstm.predict_step(feature, self.lstm_hidden_state)
            else:
                logits = self.cnn.model(x)
                model_probs = torch.softmax(logits, dim=-1).cpu().numpy().squeeze()
        model_pred = np.argmax(model_probs)

        prediction = self._select_state(model_probs)

        model_label = 'LSTM' if self.lstm_module else 'CNN'
        print(f'Frame {self.frame_idx}:\tHMM Prediction: {self.STATES[prediction]}\t{model_label} Prediction: {self.STATES[model_pred]}')

        self.predictions.append(prediction)
        self.frame_idx += 1
