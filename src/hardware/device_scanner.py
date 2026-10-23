import cv2
import sounddevice as sd


class DeviceScanner:
    """Scanează și returnează dicționarele cu camerele, microfoanele și ieșirile audio."""

    @staticmethod
    def scan_all():
        camera_devices = {}
        mic_devices = {}
        speaker_devices = {}

        # 1. Scanare Camere (verificăm primii 4 indecși rapid cu CAP_DSHOW pe Windows)
        cam_idx = 0
        for i in range(4):
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
            if not cap.isOpened():
                cap = cv2.VideoCapture(i)
            if cap.isOpened():
                camera_devices[f"[{cam_idx}] Camera {i}"] = i
                cam_idx += 1
                cap.release()

        if not camera_devices:
            camera_devices["[0] Default Camera"] = 0

        # 2. Scanare Microfoane și Boxe (cu indexare vizuală separată de la 0)
        try:
            devices = sd.query_devices()
            hostapis = sd.query_hostapis()

            preferred_api = 0
            for idx, api in enumerate(hostapis):
                if "MME" in api["name"] or "DirectSound" in api["name"]:
                    preferred_api = idx
                    break

            mic_idx = 0
            spk_idx = 0

            for i, dev in enumerate(devices):
                if dev["hostapi"] == preferred_api:
                    name_clean = dev["name"][:32]
                    if dev["max_input_channels"] > 0:
                        mic_devices[f"[{mic_idx}] {name_clean}"] = i
                        mic_idx += 1
                    if dev["max_output_channels"] > 0:
                        speaker_devices[f"[{spk_idx}] {name_clean}"] = i
                        spk_idx += 1

            # Fallback dacă filtrul API nu a găsit nimic
            if not mic_devices or not speaker_devices:
                for i, dev in enumerate(devices):
                    name_clean = dev["name"][:32]
                    if dev["max_input_channels"] > 0 and i not in mic_devices.values():
                        mic_devices[f"[{mic_idx}] {name_clean}"] = i
                        mic_idx += 1
                    if dev["max_output_channels"] > 0 and i not in speaker_devices.values():
                        speaker_devices[f"[{spk_idx}] {name_clean}"] = i
                        spk_idx += 1
        except Exception:
            pass

        if not mic_devices:
            mic_devices["[0] Default Microphone"] = None
        if not speaker_devices:
            speaker_devices["[0] Default Output"] = None

        return camera_devices, mic_devices, speaker_devices
