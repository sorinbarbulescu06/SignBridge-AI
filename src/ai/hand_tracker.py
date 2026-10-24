import cv2
import mediapipe as mp


class HandTracker:
    """
    Reprezintă blocul [AI] -> [sign input (template)] din arhitectură.
    Primește un frame BGR, desenează scheletul mâinilor și returnează
    datele standardizate (template-ul) cu cele 21 de coordonate (x, y, z).
    """

    def __init__(self, max_hands=2, detection_conf=0.7, tracking_conf=0.7):
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_styles = mp.solutions.drawing_styles

        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_hands,
            min_detection_confidence=detection_conf,
            min_tracking_confidence=tracking_conf,
        )

    def process_frame(self, frame_bgr):
        """
        Procesează cadrul curent și returnează:
        1. frame_bgr (cu scheletul desenat pe el pentru afișare în UI)
        2. sign_input_template (dicționar standardizat cu coordonatele mâinilor)
        """
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)

        # Formatul generic "sign input (template)" pe care îl trimitem mai departe
        sign_input_template = {
            "hands_detected": 0,
            "hands": []  # Fiecare element va avea: label ("Left"/"Right") și lista de 21 puncte
        }

        if results.multi_hand_landmarks and results.multi_handedness:
            sign_input_template["hands_detected"] = len(results.multi_hand_landmarks)

            for hand_lms, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                # 1. Desenăm oasele și încheieturile direct pe imagine
                self.mp_draw.draw_landmarks(
                    frame_bgr,
                    hand_lms,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_styles.get_default_hand_landmarks_style(),
                    self.mp_styles.get_default_hand_connections_style(),
                )

                # 2. Extragem coordonatele matematice în template-ul generic
                hand_label = handedness.classification[0].label  # "Left" sau "Right"
                landmarks_list = []

                for idx, lm in enumerate(hand_lms.landmark):
                    landmarks_list.append({
                        "id": idx,
                        "x": round(lm.x, 4),
                        "y": round(lm.y, 4),
                        "z": round(lm.z, 4),
                    })

                sign_input_template["hands"].append({
                    "type": hand_label,
                    "landmarks": landmarks_list
                })

        return frame_bgr, sign_input_template