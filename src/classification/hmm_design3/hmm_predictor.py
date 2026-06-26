import json
import numpy as np
from scipy.stats import norm
import tifffile
import torch

from cnn_classifier import cnn_classifier
from lstm_classifier import lstm_classifier

class HMM_Predictor:

    STATES = ['undetectable', 'NC9', 'NC9M', 'NC10', 'NC10M', 'NC11', 'NC11M', 'NC12', 'NC12M', 'NC13', 'NC13M', 'NC14+']

    def __init__(self, device, model_info_path, time_between_frames, img_size=None):
        self.device = device
        self.n_states = len(self.STATES)
        self.load_pretrained_models(model_info_path, img_size)
        self.initialize_live_prediction(time_between_frames)

    def load_pretrained_models(self, model_info_path, img_size):
        with open(model_info_path) as f:
            info = json.load(f)
        self.window_size = info['window_size']
        self.lstm_module = info['lstm_module']
        self.duration_model = {int(k): v for k, v in info['duration_model'].items()}
        if info['preprocess_images']:
            self.live_img_size = tuple(info['img_size'])
        else:
            if img_size is None:
                raise ValueError('img_size must be provided when the model was trained with preprocess_images=False')
            self.live_img_size = img_size
        self.cnn = cnn_classifier(self.device, self.window_size, self.STATES)
        self.cnn.model.eval()
        self.cnn.load_from_path(info['cnn_model_path'])

        if self.lstm_module:
            self.cnn.remove_head()
            self.lstm = lstm_classifier(self.cnn.get_hidden_size(), self.device, self.STATES, self.cnn)
            self.lstm.load_from_path(info['lstm_model_path'])

    def initialize_live_prediction(self, time_between_frames):
        self.current_state = None
        self.time_between_frames = time_between_frames
        self.frames_in_state = 0
        self.frame_idx = 0
        self.predictions = []
        self.frame_buffer = []
        self.lstm_hidden_state = None

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
    
    def _get_duration_probs(self, current_state, seconds_in_state):
        probs = np.zeros(self.n_states)

        if current_state is None:
            return np.ones(self.n_states) / self.n_states

        if current_state in self.duration_model:
            d = self.duration_model[current_state]
            p_stay = 1 - norm.cdf(seconds_in_state, d['mean'], d['std'])
            # Only allow undetectable to jump to NC9
            probs[current_state] = p_stay
            if current_state + 1 < self.n_states:
                probs[current_state + 1] = 1 - p_stay
            else:
                probs[current_state] = 1.0
            # Allow undetectable to jump to any state
            # if current_state == 0:
            #     probs[0] = p_stay
            #     remaining = (1 - p_stay) / (self.n_states - 1)
            #     probs[1:] = remaining
            # else:
            #     probs[current_state] = p_stay
            #     if current_state + 1 < self.n_states:
            #         probs[current_state + 1] = 1 - p_stay
            #     else:
            #         probs[current_state] = 1.0
        else:
            probs[current_state] = 1.0

        probs /= probs.sum() + 1e-9
        return probs

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

        seconds_in_state = self.frames_in_state * self.time_between_frames
        duration_probs = self._get_duration_probs(self.current_state, seconds_in_state)

        combined = model_probs * duration_probs
        combined /= combined.sum()
        prediction = np.argmax(combined)
        prediction = max(prediction, self.current_state or 0)

        model_label = 'LSTM' if self.lstm_module else 'CNN'
        print(f'Frame {self.frame_idx}:\tHMM Prediction: {self.STATES[prediction]}\t{model_label} Prediction: {self.STATES[model_pred]}')

        # Update current state
        self.frames_in_state = self.frames_in_state + 1 if prediction == self.current_state else 1
        self.current_state = prediction
        self.predictions.append(prediction)
        self.frame_idx += 1

if __name__ == "__main__":
    print('Running hidden markov model classification')
    DEVICE = (
        'cuda' if torch.cuda.is_available()
        else 'mps' if torch.backends.mps.is_available()
        else 'cpu'
    )
    print(f'Using device: {DEVICE}')

    # predictor = HMM_Predictor(DEVICE, 'models/model_info.json', time_between_frames=150)
    predictor = HMM_Predictor(DEVICE, 'models/model_info.json', time_between_frames=60)
    
    # Test live classifier
    print('Testing live classifier')
    test_vid = tifffile.imread("data/training_data/processed_tifs/embryo3.tif")
    # test_vid = tifffile.imread("data/hmm_tifs/processed_tifs/NCEmbryo34.tif")
    for frame in test_vid:
        predictor.predict_frame(torch.tensor(frame))