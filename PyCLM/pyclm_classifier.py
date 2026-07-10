import json
import numpy as np
from pathlib import Path
import threading
import torch

from pyclm import run_pyclm, PFSPositionMover
from pyclm.core.patterns import PatternContext, OuterPatternMethod

from src.classification.hybrid_hmm.hybrid_hmm_predictor import Hybrid_HMM_Predictor

BASE_PATH = r"E:\Justin\calssification_experiment_trials\20260708"

class ClassifyEmbryos(OuterPatternMethod):

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

        self.predictor = Hybrid_HMM_Predictor(DEVICE, r'C:\Users\Nikon\Desktop\Code\embryo-image-processing\models\model_info.json', time_between_frames=60)
        self.states = self.predictor.STATES

    def generate(self, context: PatternContext) -> np.ndarray:
        print(f"---- stimmulation: {self.experiment_name} ----")
        self._timepoint += context._experiment.pattern.every_t
        self.predictor.predict_frame(context.raw(self._classify_channel))
        state = self.predictor.current_state
        experiment_name = context._experiment.experiment_name

        state_label = self.states[state] if state is not None else "buffering"
        
        with self._log_lock:
            if experiment_name not in self._classification_logs:
                self._classification_logs[experiment_name] = {}
            self._classification_logs[experiment_name][str(self._timepoint)] = state_label
            with open(self.log_path, "w") as f:
                json.dump(self._classification_logs, f, indent=2)
        if state is not None and state >= 9: # NC13
            # Stimulation with outer bar pattern
            print(f"Stimulation at state: {state_label}")
            stim = super().generate(context)

            if np.sum(stim) == 0:
                print(f"{self.experiment_name} tried to stim but no stim happened sorry :(")

            return stim
        else:
            print(f"No stimulation at state: {state_label}")
            return np.zeros((int(self.pattern_shape[0]), int(self.pattern_shape[1])), dtype=np.float16)
        
if __name__ == "__main__":
    pattern_methods = {"classify_embryos": ClassifyEmbryos}

    run_pyclm(BASE_PATH, pattern_methods=pattern_methods, position_mover=PFSPositionMover(), gui=True)
