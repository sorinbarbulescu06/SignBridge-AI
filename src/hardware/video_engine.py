import cv2
from PIL import Image
from ai.hand_tracker import HandTracker


class VideoEngine:
    """Gestionează exclusiv fluxul OpenCV (cameră web sau fișier video)."""

    def __init__(self):
        self.cap = None
        self.is_running = False
        self.is_live = False
        self.hand_tracker = HandTracker()

    def start(self, source, is_live=True):
        self.stop()
        self.is_live = is_live

        if is_live:
            self.cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
            if not self.cap.isOpened():
                self.cap = cv2.VideoCapture(source)
        else:
            self.cap = cv2.VideoCapture(source)

        if not self.cap.isOpened():
            self.cap = None
            self.is_running = False
            return False

        self.is_running = True
        return True

    def get_frame(self):
        """Returnează (succes, imagine_PIL_cu_schelet, sign_input_template)."""
        if not self.is_running or self.cap is None:
            return False, None, None

        ret, frame = self.cap.read()
        if not ret:
            return False, None, None

        if self.is_live:
            frame = cv2.flip(frame, 1)

        # Trimitem cadrul prin AI pentru desenarea scheletului și extragerea template-ului
        frame_annotated, sign_template = self.hand_tracker.process_frame(frame)

        frame_rgb = cv2.cvtColor(frame_annotated, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(frame_rgb)
        return True, img_pil, sign_template

    def stop(self):
        self.is_running = False
        if self.cap is not None:
            self.cap.release()
            self.cap = None
