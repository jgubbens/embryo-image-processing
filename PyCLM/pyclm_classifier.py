from cv2 import resize
import json
import numpy as np
from pathlib import Path
import os
import threading
import tifffile
import torch

from skimage.measure import label, regionprops

from pyclm import run_pyclm, PFSPositionMover
from pyclm.core.patterns import PatternContext, OuterPatternMethod
from classification.hybrid_hmm.hybrid_hmm_predictor import Hybrid_HMM_Predictor
from classification.pure_hmm.pure_hmm_predictor import Pure_HMM_Predictor
from processing.extract_embryo import EmbryoExtractor, MultipleEmbryoExtractor


# BASE_PATH = r"E:\Justin\calssification_experiment_trials\20260708"
BASE_PATH = r"PyCLM"
PREDICTOR_PATH = r'models/hybrid_hmm_efficientnetb3_preprocessed/hybrid_hmm_model_info.json'

class ClassifyEmbryo(OuterPatternMethod):

    name = "classify_embryos"
    log_path = Path(BASE_PATH, 'classification_logs.json')
    _log_lock = threading.Lock()
    _classification_logs = {}

    def __init__(self, classify_channel="brightfield", **kwargs):
        super().__init__(channel=classify_channel, **kwargs)

        self._requirements_list = [(classify_channel, True, True)]
        self._classify_channel = classify_channel
        self._timepoint = 0

        DEVICE = (
            'cuda' if torch.cuda.is_available()
            else 'mps' if torch.backends.mps.is_available()
            else 'cpu'
        )

        with self._log_lock:
            if not self.log_path.exists():
                info = {}
                with open(self.log_path, 'w') as f:
                    json.dump(info, f, indent=2)

        self.predictor = Hybrid_HMM_Predictor(DEVICE, PREDICTOR_PATH, time_between_frames=60)
        self.states = self.predictor.STATES
        self.extractor = EmbryoExtractor()

    def generate(self, context: PatternContext) -> np.ndarray:
        print(f"---- stimmulation: {self.experiment_name} ----")
        self._timepoint += context._experiment.pattern.every_t

        # Preprocess frame
        raw_frame = context.raw(self._classify_channel)
        frame = self.extractor.extract_frame(raw_frame)

        os.makedirs(Path(BASE_PATH, "processed_frames"), exist_ok=True)
        tifffile.imwrite(Path(BASE_PATH, "processed_frames", f"{context._experiment.experiment_name}_{self._timepoint}.tif"), frame)

        self.predictor.predict_frame(frame)
        current_state = self.predictor.get_current_state()
        state = self.states.index(current_state) if current_state is not None else None
        experiment_name = context._experiment.experiment_name

        state_label = current_state if current_state is not None else "buffering"
        
        with self._log_lock:
            if experiment_name not in self._classification_logs:
                self._classification_logs[experiment_name] = {}
            self._classification_logs[experiment_name][str(self._timepoint)] = state_label
            with open(self.log_path, "w") as f:
                json.dump(self._classification_logs, f, indent=2)
        # if state is not None and state >= 5 and state < 11: # NC11 <= state < NC14+
        if state is not None and state >= 9: # state >= 13
            # Stimulation with outer bar pattern
            print(f"Stimulation at state: {state_label}")
            stim = super().generate(context)

            if np.sum(stim) == 0:
                print(f"{self.experiment_name} tried to stim but failed")

            return stim
        else:
            print(f"No stimulation at state: {state_label}")
            return np.zeros((int(self.pattern_shape[0]), int(self.pattern_shape[1])), dtype=np.float16)


class ClassifyMultipleEmbryos(OuterPatternMethod):

    name = "classify_embryos"
    log_path = Path(BASE_PATH, 'classification_logs.json')
    _log_lock = threading.Lock()
    _classification_logs = {}

    def __init__(self, classify_channel="brightfield", **kwargs):
        super().__init__(channel=classify_channel, **kwargs)

        self._requirements_list = [(classify_channel, True, True)]
        self._classify_channel = classify_channel
        self._timepoint = 0

        self.DEVICE = (
            'cuda' if torch.cuda.is_available()
            else 'mps' if torch.backends.mps.is_available()
            else 'cpu'
        )

        with self._log_lock:
            if not self.log_path.exists():
                info = {}
                with open(self.log_path, 'w') as f:
                    json.dump(info, f, indent=2)

        self.extractor = MultipleEmbryoExtractor()
        self.tracked_predictors = {}

    def _add_stim(self, stim, mask):
        if mask.shape != stim.shape:
            mask = resize(mask.astype(np.float32), (stim.shape[1], stim.shape[0])) > 0.5

        labeled_mask = label(mask)
        props = regionprops(labeled_mask)

        if len(props) == 0:
            print(f"No props segmented: {self.experiment_name}")
            return stim

        prop = props[0]
        centroid = prop.centroid
        long_axis = (np.sin(prop.orientation), np.cos(prop.orientation))
        axis_length = prop.axis_major_length

        y_arange = np.arange(self.pattern_shape[0])
        x_arange = np.arange(self.pattern_shape[1])

        yy, xx = np.meshgrid(y_arange, x_arange)

        mag = (yy - centroid[1]) * long_axis[0] + (xx - centroid[0]) * long_axis[1]
        mag = np.abs(mag) / (axis_length / 2)

        included = self.apply_magnitude(mag)

        return stim + included * mask

    def generate(self, context: PatternContext) -> np.ndarray:
        print(f"---- stimmulation: {self.experiment_name} ----")
        self._timepoint += context._experiment.pattern.every_t

        # Preprocess frame
        raw_frame = context.raw(self._classify_channel)
        frames = self.extractor.extract_frame(raw_frame)

        stim = np.zeros((int(self.pattern_shape[0]), int(self.pattern_shape[1])), dtype=np.float16)

        should_stim = False

        os.makedirs(Path(BASE_PATH, "processed_frames"), exist_ok=True)
        for id, frame in frames.items():
            if id not in self.tracked_predictors.keys():
                self.tracked_predictors[id] = Hybrid_HMM_Predictor(self.DEVICE, PREDICTOR_PATH, time_between_frames=60)
            tifffile.imwrite(Path(BASE_PATH, "processed_frames", f"{context._experiment.experiment_name}_{id}_{self._timepoint}.tif"), frame.out_frame)
            print(f"Embryo ID: {id}")
            predictor = self.tracked_predictors[id]
            predictor.predict_frame(frame.out_frame)
            current_state = predictor.get_current_state()
            state = predictor.STATES.index(current_state) if current_state is not None else None
            experiment_name = context._experiment.experiment_name

            state_label = current_state if current_state is not None else "buffering"
            
            with self._log_lock:
                if experiment_name not in self._classification_logs:
                    self._classification_logs[experiment_name] = {}
                if str(id) not in self._classification_logs[experiment_name]:
                    self._classification_logs[experiment_name][str(id)] = {}
                self._classification_logs[experiment_name][str(id)][str(self._timepoint)] = state_label
                with open(self.log_path, "w") as f:
                    json.dump(self._classification_logs, f, indent=2)
            # if state is not None and state >= 5 and state < 11: # NC11 <= state < NC14+
            if state is not None and state >= 9: # state >= 13
                # Stimulation with outer bar pattern
                print(f"Stimulation at state: {state_label}")
                stim = self._add_stim(stim, frame.mask)
                should_stim = True
            else:
                print(f"No stimulation at state: {state_label}")
        
        if should_stim and np.sum(stim) == 0:
            print(f"Should stim but didn't: {experiment_name}, t = {self._timepoint}")
        return stim
            

if __name__ == "__main__":
    pattern_methods = {"classify_embryo": ClassifyEmbryo, "classify_multiple_embryos": ClassifyMultipleEmbryos}

    run_pyclm(BASE_PATH, pattern_methods=pattern_methods, position_mover=PFSPositionMover(), gui=True, dry=True)
