import os
import joblib
import numpy as np
from collections import deque, Counter
from ai.feature_extractor import SignFeatureExtractor


class SignTranslator:
    """
    Modul de inferență ultra-rapid pentru semne statice și dinamice.
    """

    def __init__(self, model_path=None, window_size=5, confidence_threshold=0.30):
        if model_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            model_path = os.path.join(base_dir, "data", "sign_model.pkl")

        self.model_path = model_path
        self.model = None
        self.classes = []
        self.confidence_threshold = confidence_threshold

        # Fereastră scurtă (5 cadre = ~150ms) pentru reacție instantanee la semne rapide
        self.frame_window = deque(maxlen=window_size)
        self.prediction_history = deque(maxlen=2)
        self.last_translated_word = None
        self.frame_counter = 0

        self.load_model()

    def load_model(self):
        if os.path.exists(self.model_path):
            try:
                bundle = joblib.load(self.model_path)
                self.model = bundle["model"]
                # Dezactivăm multiprocessing-ul la inferență (elimină lag-ul masiv pe Windows)
                if hasattr(self.model, "steps"):
                    clf = self.model.steps[-1][1]
                    if hasattr(clf, "n_jobs"):
                        clf.n_jobs = 1
                self.classes = list(bundle["classes"])
                return True
            except Exception as e:
                print(f"[WARN] Could not load AI model: {e}")
        return False

    def translate(self, sign_input_template):
        if not sign_input_template or sign_input_template.get("hands_detected", 0) == 0:
            self.frame_window.clear()
            self.prediction_history.clear()
            self.last_translated_word = None
            return None

        if self.model is None:
            if not self.load_model():
                return None

        frame_feat = SignFeatureExtractor.extract_frame_features(sign_input_template)
        if frame_feat is None:
            return None

        self.frame_window.append(frame_feat)
        self.frame_counter += 1

        # Rulăm clasificatorul AI din 2 în 2 cadre și imediat ce avem minim 2 cadre
        if len(self.frame_window) < 2 or (self.frame_counter % 2 != 0):
            return None

        seq_vector = SignFeatureExtractor.extract_sequence_features(list(self.frame_window))
        if seq_vector is None:
            return None

        probs = self.model.predict_proba([seq_vector])[0]
        best_idx = int(np.argmax(probs))
        best_prob = float(probs[best_idx])
        predicted_label = self.model.classes_[best_idx]

        if best_prob >= self.confidence_threshold:
            self.prediction_history.append(predicted_label)

            # Confirmare rapidă din 2 predicții consecutive
            if len(self.prediction_history) == 2 and (
                self.prediction_history[0] == self.prediction_history[1]
            ):
                if predicted_label != self.last_translated_word:
                    self.last_translated_word = predicted_label
                    # Golim jumătate din fereastră ca următorul semn să fie prins imediat
                    while len(self.frame_window) > 2:
                        self.frame_window.popleft()
                    return f"{predicted_label} ({int(best_prob * 100)}%)"

        return None