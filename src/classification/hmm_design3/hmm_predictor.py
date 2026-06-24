import json
import numpy as np
from scipy.stats import norm
import tifffile
import torch

from cnn_classifier import cnn_classifier

class HMM_Predictor:

    STATES = ['undetectable', 'NC9', 'NC9M', 'NC10', 'NC10M', 'NC11', 'NC11M', 'NC12', 'NC12M', 'NC13', 'NC13M', 'NC14+']

    def __init__(self, device, trained_cnn_path, duration_model_path, time_between_frames, img_size, window_size=1):
        self.device = device
        self.n_states = len(self.STATES)
        self.window_size = window_size
        self.load_pretrained_models(trained_cnn_path, duration_model_path)
        self.initialize_live_prediction(time_between_frames, img_size)
    
    def load_pretrained_models(self, trained_cnn_path, duration_model_path):
        self.cnn = cnn_classifier(self.device, self.window_size, self.STATES)
        self.cnn.model.eval()
        self.cnn.load_from_path(trained_cnn_path)
        with open(duration_model_path) as f:
            raw = json.load(f)
        self.duration_model = {int(k): v for k, v in raw.items()}

    def initialize_live_prediction(self, time_between_frames, img_size):
        self.current_state = None
        self.time_between_frames = time_between_frames
        self.frames_in_state = 0
        self.frame_idx = 0
        self.predictions = []
        self.live_img_size = img_size
        self.frame_buffer = []
    
    def _preprocess_live_frame(self, frame_np):
        self.frame_buffer.append(frame_np)
        if len(self.frame_buffer) > self.window_size:
            self.frame_buffer.pop(0)
        pad = self.window_size - len(self.frame_buffer)
        window = np.stack(
            [np.zeros_like(self.frame_buffer[0])] * pad + list(self.frame_buffer), axis=0
        ).astype(np.float32)
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
            # probs[current_state] = p_stay
            # if current_state + 1 < self.n_states:
            #     probs[current_state + 1] = 1 - p_stay
            # else:
            #     probs[current_state] = 1.0
            # Allow undetectable to jump to any state
            if current_state == 0:
                probs[0] = p_stay
                remaining = (1 - p_stay) / (self.n_states - 1)
                probs[1:] = remaining
            else:
                probs[current_state] = p_stay
                if current_state + 1 < self.n_states:
                    probs[current_state + 1] = 1 - p_stay
                else:
                    probs[current_state] = 1.0
        else:
            probs[current_state] = 1.0

        probs /= probs.sum() + 1e-9
        return probs

    def predict_frame(self, frame):
        frame_np = frame.numpy() if isinstance(frame, torch.Tensor) else frame

        with torch.no_grad():
            x = self._preprocess_live_frame(frame_np).to(self.device)
            logits = self.cnn.model(x)
            cnn_probs = (torch.softmax(logits, dim=-1).cpu().numpy().squeeze()
        )
        cnn_pred = np.argmax(cnn_probs)
        
        seconds_in_state = self.frames_in_state * self.time_between_frames
        duration_probs = self._get_duration_probs(self.current_state, seconds_in_state)

        combined = cnn_probs * duration_probs
        combined /= combined.sum()
        prediction = np.argmax(combined)
        prediction = max(prediction, self.current_state or 0)

        print(f'Frame {self.frame_idx}:\tHMM Prediction: {self.STATES[prediction]}\tCNN Prediction: {self.STATES[cnn_pred]}')

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

    predictor = HMM_Predictor(DEVICE, 'models/best_hmm_cnn.pt', 'models/duration_model.json', 
                              time_between_frames=150, img_size=(800, 800))
    # predictor = HMM_Predictor(DEVICE, 'models/best_hmm_cnn.pt', 'models/duration_model.json', 
    #                           time_between_frames=60, img_size=(800, 800))
    
    # Test live classifier
    print('Testing live classifier')
    # test_vid = tifffile.imread("data/training_data/histone/embryo3.tif")
    test_vid = tifffile.imread("data/hmm_tifs/processed_tifs/NCEmbryo34.tif")
    for frame in test_vid:
        predictor.predict_frame(torch.tensor(frame))