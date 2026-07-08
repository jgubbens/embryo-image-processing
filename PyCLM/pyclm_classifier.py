import numpy as np
import torch

from pyclm import run_pyclm, PFSPositionMover
from pyclm.core.patterns import PatternContext, OuterPatternMethod

from src.classification.hybrid_hmm.hybrid_hmm_predictor import Hybrid_HMM_Predictor

class ClassifyEmbryos(OuterPatternMethod):

    name = "classify_embryos"

    def __init__(self, classify_channel="brightfield", **kwargs):
        super().__init__(channel=classify_channel, **kwargs)

        self._requirements_list = [(classify_channel, True, True)]
        self._classify_channel = classify_channel

        DEVICE = (
            'cuda' if torch.cuda.is_available()
            else 'mps' if torch.backends.mps.is_available()
            else 'cpu'
        )

        self.predictor = Hybrid_HMM_Predictor(DEVICE, 'models/model_info.json', time_between_frames=60, img_size=(800,800))
        self.states = self.predictor.STATES

    def generate(self, context: PatternContext) -> np.ndarray:
        self.predictor.predict_frame(context.raw(self._classify_channel))
        state = self.predictor.current_state
        if state is not None and state >= 9: # NC13
            # Stimulation with outer bar pattern
            print(f"Stimulation at state: {self.states[state]}")
            return super().generate(context)
        else:
            state_label = self.states[state] if state is not None else "buffering"
            print(f"No stimulation at state: {state_label}")
            return np.zeros((int(self.pattern_shape[0]), int(self.pattern_shape[1])), dtype=np.float16)
        
BASE_PATH = r"C:\Users\Nikon\Desktop\Code\Toettchlab-FBC\test_experiment_outputs\test_gol"

if __name__ == "__main__":
    pattern_methods = {"classify_embryos": ClassifyEmbryos}

    run_pyclm(BASE_PATH, pattern_methods=pattern_methods, position_mover=PFSPositionMover(), gui=True)
