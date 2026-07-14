import json
import numpy as np
import tifffile
import torch

from classification.cnn_classifier import cnn_classifier
from classification.lstm_classifier import lstm_classifier

class Pure_HMM_Predictor:

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
        self.transition_matrix = np.array(info['transition_matrix'])
        self.live_img_size = tuple(info.get('img_size', [800, 800]))
        self.cnn = cnn_classifier(self.device, self.window_size, self.STATES)
        self.cnn.model.eval()
        self.cnn.load_from_path(info['cnn_model_path'])

        if self.lstm_module:
            self.cnn.remove_head()
            self.lstm = lstm_classifier(self.cnn.get_hidden_size(), self.device, self.STATES, self.cnn)
            self.lstm.load_from_path(info['lstm_model_path'])

    def initialize_live_prediction(self, time_between_frames):
        self.time_between_frames = time_between_frames
        self.viterbi_log_dp = None
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
    
    def _viterbi_step(self, model_probs):
        log_emit = np.log(np.array(model_probs) + 1e-10)
        log_trans = np.log(self.transition_matrix + 1e-10)
        if self.viterbi_log_dp is None:
            self.viterbi_log_dp = log_emit - np.log(self.n_states)
        else:
            candidates = self.viterbi_log_dp[:, None] + log_trans
            self.viterbi_log_dp = np.max(candidates, axis=0) + log_emit
        return int(np.argmax(self.viterbi_log_dp))

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

        prediction = self._viterbi_step(model_probs)

        model_label = 'LSTM' if self.lstm_module else 'CNN'
        print(f'Frame {self.frame_idx}:\tHMM Prediction: {self.STATES[prediction]}\t{model_label} Prediction: {self.STATES[model_pred]}')

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

    predictor = Pure_HMM_Predictor(DEVICE, 'models/pure_hmm/pure_hmm_model_info.json', time_between_frames=60)
    
    # Test live classifier
    print('Testing live classifier')
    test_vid = tifffile.imread("data/training_data/brightfield/embryo3.tif")
    # test_vid = tifffile.imread("data/hmm_tifs/processed_tifs/NCEmbryo34.tif")
    for frame in test_vid:
        predictor.predict_frame(torch.tensor(frame))
