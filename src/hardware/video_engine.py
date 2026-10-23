import cv2
from PIL import Image


class VideoEngine:
    """Gestionează exclusiv fluxul OpenCV (cameră web sau fișier video)."""

    def __init__(self):
        self.cap = None
        self.is_running = False
        self.is_live = False

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
        """Returnează un tuplu (succes, imagine_PIL)."""
        if not self.is_running or self.cap is None:
            return False, None

        ret, frame = self.cap.read()
        if not ret:
            return False, None

        if self.is_live:
            frame = cv2.flip(frame, 1)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(frame_rgb)
        return True, img_pil

    def stop(self):
        self.is_running = False
        if self.cap is not None:
            self.cap.release()
            self.cap = None
