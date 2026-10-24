import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf.symbol_database")

import cv2
import mediapipe as mp


class HandTracker:
    """
    Reprezintă blocul [AI] -> [sign input (template)] din arhitectură.
    Optimizat cu model_complexity=0 pentru rulare la FPS maxim în timp real.
    """

    def __init__(self, max_hands=1, detection_conf=0.6, tracking_conf=0.6):
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_styles = mp.solutions.drawing_styles

        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_hands,
            model_complexity=0,  # 0 = Fast/Lite (de 3x mai rapid decât 1)
            min_detection_confidence=detection_conf,
            min_tracking_confidence=tracking_conf,
        )

    def process_frame(self, frame_bgr):
        # Procesăm detecția pe o copie micșorată (320x240) pentru viteză maximă,
        # deoarece coordonatele MediaPipe sunt normalizate [0.0 - 1.0]
        small_frame = cv2.resize(frame_bgr, (320, 240))
        frame_rgb = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False
        results = self.hands.process(frame_rgb)

        sign_input_template = {
            "hands_detected": 0,
            "hands": []
        }

        if results.multi_hand_landmarks and results.multi_handedness:
            sign_input_template["hands_detected"] = len(results.multi_hand_landmarks)

            for hand_lms, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                self.mp_draw.draw_landmarks(
                    frame_bgr,
                    hand_lms,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_styles.get_default_hand_landmarks_style(),
                    self.mp_styles.get_default_hand_connections_style(),
                )

                hand_label = handedness.classification[0].label
                landmarks_list = [
                    {
                        "id": idx,
                        "x": float(lm.x),
                        "y": float(lm.y),
                        "z": float(lm.z),
                    }
                    for idx, lm in enumerate(hand_lms.landmark)
                ]

                sign_input_template["hands"].append({
                    "type": hand_label,
                    "landmarks": landmarks_list
                })

        return frame_bgr, sign_input_template