import customtkinter as ctk
from tkinter import filedialog, Canvas
import cv2
from PIL import Image
import sounddevice as sd
import numpy as np
import math

# Configurare temă generală (Dark Mode + accente albastre)
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")


class SignLanguageApp(ctk.CTk):

    def __init__(self):
        super().__init__()

        # Configurare fereastră principală
        self.title("AI Sign Language Translator")
        self.geometry("1080x720")
        self.minsize(900, 600)

        # Imagine goală 1x1 (Workaround pentru bug-ul CustomTkinter unde image=None nu șterge frame-ul)
        self.empty_image = ctk.CTkImage(
            light_image=Image.new("RGBA", (1, 1), (0, 0, 0, 0)),
            dark_image=Image.new("RGBA", (1, 1), (0, 0, 0, 0)),
            size=(1, 1),
        )

        # Variabile de stare pentru Hardware & Loop-uri
        self.cap = None
        self.camera_running = False
        self.is_live_camera = False
        self.video_loop_id = None

        self.audio_stream = None
        self.mic_running = False
        self.wave_loop_id = None
        self.current_volume = 0.0
        self.wave_phase = 0.0

        # Dicționare pentru maparea dispozitivelor (Nume afișat -> Index hardware)
        self.camera_devices = {}
        self.mic_devices = {}
        self.speaker_devices = {}

        # Scanăm dispozitivele la pornire
        self.scan_hardware_devices()

        # Layout principal (Grid 2x1: Bara de sus pentru opțiuni + Zona de lucru jos)
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=1)

        # ==========================================
        # ZONA VERDE: Header & Selector de Opțiune
        # ==========================================
        self.header_frame = ctk.CTkFrame(self, height=80, corner_radius=0)
        self.header_frame.grid(row=0, column=0, sticky="ew")
        self.header_frame.grid_columnconfigure(0, weight=1)

        self.title_label = ctk.CTkLabel(
            self.header_frame,
            text="AI Sign Language Hub",
            font=ctk.CTkFont(size=22, weight="bold"),
        )
        self.title_label.grid(row=0, column=0, pady=(15, 5))

        # Selectorul "option" din schiță: sign -> voice vs voice -> sign
        self.mode_selector = ctk.CTkSegmentedButton(
            self.header_frame,
            values=["Sign -> Voice", "Voice -> Sign"],
            command=self.switch_mode,
            font=ctk.CTkFont(size=14, weight="bold"),
            height=38,
            width=320,
        )
        self.mode_selector.grid(row=1, column=0, pady=(5, 15))
        self.mode_selector.set("Sign -> Voice")

        # ==========================================
        # CONTAINERE PENTRU CELE DOUĂ FLUXURI
        # ==========================================
        self.sign_to_voice_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.voice_to_sign_frame = ctk.CTkFrame(self, fg_color="transparent")

        self.setup_sign_to_voice_ui()
        self.setup_voice_to_sign_ui()

        # Afișăm primul mod la pornire
        self.switch_mode("Sign -> Voice")

    # ==========================================
    # SCANARE HARDWARE (Camere, Microfoane, Boxe)
    # ==========================================
    def scan_hardware_devices(self):
        """Detectează camerele video și dispozitivele audio de intrare/ieșire disponibile"""
        self.camera_devices.clear()
        self.mic_devices.clear()
        self.speaker_devices.clear()

        # 1. Scanare Camere (verificăm primii 4 indecși rapid cu CAP_DSHOW pe Windows)
        cam_idx = 0
        for i in range(4):
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
            if not cap.isOpened():
                cap = cv2.VideoCapture(i)
            if cap.isOpened():
                self.camera_devices[f"[{cam_idx}] Camera {i}"] = i
                cam_idx += 1
                cap.release()

        if not self.camera_devices:
            self.camera_devices["[0] Default Camera"] = 0

        # 2. Scanare Microfoane și Boxe (cu indexare vizuală separată de la 0)
        try:
            devices = sd.query_devices()
            hostapis = sd.query_hostapis()

            # Preferăm MME sau Windows DirectSound pentru a evita duplicatele din listă
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
                        self.mic_devices[f"[{mic_idx}] {name_clean}"] = i
                        mic_idx += 1
                    if dev["max_output_channels"] > 0:
                        self.speaker_devices[f"[{spk_idx}] {name_clean}"] = i
                        spk_idx += 1

            # Fallback dacă filtrul API nu a găsit nimic
            if not self.mic_devices or not self.speaker_devices:
                for i, dev in enumerate(devices):
                    name_clean = dev["name"][:32]
                    if dev["max_input_channels"] > 0 and i not in self.mic_devices.values():
                        self.mic_devices[f"[{mic_idx}] {name_clean}"] = i
                        mic_idx += 1
                    if dev["max_output_channels"] > 0 and i not in self.speaker_devices.values():
                        self.speaker_devices[f"[{spk_idx}] {name_clean}"] = i
                        spk_idx += 1
        except Exception:
            pass

        if not self.mic_devices:
            self.mic_devices["[0] Default Microphone"] = None
        if not self.speaker_devices:
            self.speaker_devices["[0] Default Output"] = None

    def refresh_device_lists(self):
        """Re-scanează hardware-ul și actualizează meniurile dropdown"""
        self.scan_hardware_devices()

        cam_list = list(self.camera_devices.keys())
        self.cam_dropdown.configure(values=cam_list)
        self.cam_dropdown.set(cam_list[0])

        spk_list = list(self.speaker_devices.keys())
        self.speaker_dropdown.configure(values=spk_list)
        self.speaker_dropdown.set(spk_list[0])

        mic_list = list(self.mic_devices.keys())
        self.mic_dropdown.configure(values=mic_list)
        self.mic_dropdown.set(mic_list[0])

    # ==========================================
    # HELPER PENTRU PĂSTRAREA ASPECT RATIO-ULUI
    # ==========================================
    def fit_image_to_container(self, img_pil, container_widget, padding=16):
        """
        Calculează dimensiunile optime (w, h) ca imaginea să încapă maxim în container
        fără să își piardă proporțiile originale (fără stretch).
        """
        cont_w = container_widget.winfo_width() - padding
        cont_h = container_widget.winfo_height() - padding

        if cont_w <= 50 or cont_h <= 50:
            cont_w, cont_h = 400, 300

        orig_w, orig_h = img_pil.size
        if orig_w == 0 or orig_h == 0:
            return ctk.CTkImage(light_image=img_pil, dark_image=img_pil, size=(cont_w, cont_h))

        # Alegem factorul de scalare mai mic pentru a păstra aspect ratio-ul intact
        scale = min(cont_w / orig_w, cont_h / orig_h)
        new_w = max(1, int(orig_w * scale))
        new_h = max(1, int(orig_h * scale))

        return ctk.CTkImage(light_image=img_pil, dark_image=img_pil, size=(new_w, new_h))

    # ==========================================
    # UI: RAMURA STÂNGĂ (Sign -> Voice)
    # ==========================================
    def setup_sign_to_voice_ui(self):
        self.sign_to_voice_frame.grid_columnconfigure(0, weight=1, uniform="group1")
        self.sign_to_voice_frame.grid_columnconfigure(1, weight=1, uniform="group1")
        self.sign_to_voice_frame.grid_rowconfigure(0, weight=1)

        # --- Panoul Stâng: INPUT (Camera / Video File) ---
        input_panel = ctk.CTkFrame(self.sign_to_voice_frame, corner_radius=15)
        input_panel.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")
        input_panel.grid_rowconfigure(2, weight=1)
        input_panel.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            input_panel,
            text="SIGN INPUT SOURCE",
            font=ctk.CTkFont(size=16, weight="bold"),
        ).grid(row=0, column=0, pady=(15, 5))

        # Selector Cameră
        cam_select_frame = ctk.CTkFrame(input_panel, fg_color="transparent")
        cam_select_frame.grid(row=1, column=0, padx=20, pady=(0, 5), sticky="ew")

        ctk.CTkLabel(cam_select_frame, text="Camera:", font=ctk.CTkFont(size=13)).pack(side="left", padx=(0, 8))
        self.cam_dropdown = ctk.CTkOptionMenu(
            cam_select_frame,
            values=list(self.camera_devices.keys()),
            command=self.on_camera_device_change,
            width=200,
            height=28,
        )
        self.cam_dropdown.pack(side="left", fill="x", expand=True)

        ctk.CTkButton(
            cam_select_frame,
            text="⟳",
            width=32,
            height=28,
            fg_color="#3a3a3a",
            hover_color="#505050",
            command=self.refresh_device_lists,
        ).pack(side="left", padx=(8, 0))

        # Container fix pentru video (grid_propagate(False) previne deformarea layout-ului)
        self.video_container = ctk.CTkFrame(input_panel, fg_color="#1a1a1a", corner_radius=10)
        self.video_container.grid(row=2, column=0, padx=20, pady=10, sticky="nsew")
        self.video_container.grid_propagate(False)
        self.video_container.grid_rowconfigure(0, weight=1)
        self.video_container.grid_columnconfigure(0, weight=1)

        self.video_display = ctk.CTkLabel(
            self.video_container,
            text="[ Camera / Video Feed ]\n\nWaiting for input...",
            image=self.empty_image,
            fg_color="transparent",
            font=ctk.CTkFont(size=14),
        )
        self.video_display.grid(row=0, column=0, sticky="nsew")

        # Butoane pentru selectarea sursei
        btn_frame = ctk.CTkFrame(input_panel, fg_color="transparent")
        btn_frame.grid(row=3, column=0, pady=15)

        self.btn_camera = ctk.CTkButton(
            btn_frame,
            text="Open Camera",
            command=self.toggle_camera,
            width=150,
            height=35,
        )
        self.btn_camera.pack(side="left", padx=10)

        self.btn_video = ctk.CTkButton(
            btn_frame,
            text="Upload Video File",
            command=self.load_video_file,
            fg_color="#2b5b84",
            width=150,
            height=35,
        )
        self.btn_video.pack(side="left", padx=10)

        # --- Panoul Drept: OUTPUT TEXT & AUDIO OUT ---
        output_panel = ctk.CTkFrame(self.sign_to_voice_frame, corner_radius=15)
        output_panel.grid(row=0, column=1, padx=20, pady=20, sticky="nsew")
        output_panel.grid_rowconfigure(2, weight=1)
        output_panel.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            output_panel,
            text="OUTPUT TEXT & VOICE",
            font=ctk.CTkFont(size=16, weight="bold"),
        ).grid(row=0, column=0, pady=(15, 5))

        # Selector Dispozitiv Ieșire Audio (Speaker / Căști)
        spk_select_frame = ctk.CTkFrame(output_panel, fg_color="transparent")
        spk_select_frame.grid(row=1, column=0, padx=20, pady=(0, 5), sticky="ew")

        ctk.CTkLabel(spk_select_frame, text="Audio Out:", font=ctk.CTkFont(size=13)).pack(side="left", padx=(0, 8))
        self.speaker_dropdown = ctk.CTkOptionMenu(
            spk_select_frame,
            values=list(self.speaker_devices.keys()),
            width=220,
            height=28,
        )
        self.speaker_dropdown.pack(side="left", fill="x", expand=True)

        self.text_output = ctk.CTkTextbox(
            output_panel,
            font=ctk.CTkFont(size=16),
            corner_radius=10,
            wrap="word",
        )
        self.text_output.grid(row=2, column=0, padx=20, pady=10, sticky="nsew")
        self.text_output.insert("0.0", "Translated text will appear here\n")

        self.btn_clear_text = ctk.CTkButton(
            output_panel,
            text="Delete Text",
            fg_color="#4a4a4a",
            hover_color="#333333",
            command=lambda: self.text_output.delete("0.0", "end"),
        )
        self.btn_clear_text.grid(row=3, column=0, pady=15)

    # ==========================================
    # UI: RAMURA DREAPTĂ (Voice -> Sign)
    # ==========================================
    def setup_voice_to_sign_ui(self):
        self.voice_to_sign_frame.grid_columnconfigure(0, weight=1, uniform="group1")
        self.voice_to_sign_frame.grid_columnconfigure(1, weight=1, uniform="group1")
        self.voice_to_sign_frame.grid_rowconfigure(0, weight=1)

        # --- Panoul Stâng: INPUT (Microphone / Audio File) ---
        input_panel = ctk.CTkFrame(self.voice_to_sign_frame, corner_radius=15)
        input_panel.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")
        input_panel.grid_rowconfigure(2, weight=1)
        input_panel.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            input_panel,
            text="VOICE INPUT SOURCE",
            font=ctk.CTkFont(size=16, weight="bold"),
        ).grid(row=0, column=0, pady=(15, 5))

        # Selector Microfon
        mic_select_frame = ctk.CTkFrame(input_panel, fg_color="transparent")
        mic_select_frame.grid(row=1, column=0, padx=20, pady=(0, 5), sticky="ew")

        ctk.CTkLabel(mic_select_frame, text="Mic In:", font=ctk.CTkFont(size=13)).pack(side="left", padx=(0, 8))
        self.mic_dropdown = ctk.CTkOptionMenu(
            mic_select_frame,
            values=list(self.mic_devices.keys()),
            command=self.on_mic_device_change,
            width=200,
            height=28,
        )
        self.mic_dropdown.pack(side="left", fill="x", expand=True)

        ctk.CTkButton(
            mic_select_frame,
            text="⟳",
            width=32,
            height=28,
            fg_color="#3a3a3a",
            hover_color="#505050",
            command=self.refresh_device_lists,
        ).pack(side="left", padx=(8, 0))

        # Container pentru Canvas-ul undei audio (grid_propagate(False) pentru redimensionare curată)
        self.wave_container = ctk.CTkFrame(input_panel, fg_color="#1a1a1a", corner_radius=10)
        self.wave_container.grid(row=2, column=0, padx=20, pady=10, sticky="nsew")
        self.wave_container.grid_propagate(False)
        self.wave_container.grid_rowconfigure(0, weight=1)
        self.wave_container.grid_columnconfigure(0, weight=1)

        self.wave_canvas = Canvas(
            self.wave_container,
            bg="#1a1a1a",
            highlightthickness=0
        )
        self.wave_canvas.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

        self.audio_status_overlay = ctk.CTkLabel(
            self.wave_container,
            text="[ Status Audio / Waveform ]\n\nWaiting for input...",
            fg_color="#1a1a1a",
            font=ctk.CTkFont(size=14),
        )
        self.audio_status_overlay.place(relx=0.5, rely=0.5, anchor="center")

        btn_frame = ctk.CTkFrame(input_panel, fg_color="transparent")
        btn_frame.grid(row=3, column=0, pady=15)

        self.btn_mic = ctk.CTkButton(
            btn_frame,
            text="Use Microphone",
            command=self.toggle_mic,
            fg_color="#c93b2b",
            hover_color="#9e2a1b",
            width=150,
            height=35,
        )
        self.btn_mic.pack(side="left", padx=10)

        self.btn_audio_file = ctk.CTkButton(
            btn_frame,
            text="Upload Audio File",
            command=self.load_audio_file,
            fg_color="#2b5b84",
            width=150,
            height=35,
        )
        self.btn_audio_file.pack(side="left", padx=10)

        # --- Panoul Drept: OUTPUT 3D MODEL ---
        output_panel = ctk.CTkFrame(self.voice_to_sign_frame, corner_radius=15)
        output_panel.grid(row=0, column=1, padx=20, pady=20, sticky="nsew")
        output_panel.grid_rowconfigure(1, weight=1)
        output_panel.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            output_panel,
            text="OUTPUT 3D MODEL",
            font=ctk.CTkFont(size=16, weight="bold"),
        ).grid(row=0, column=0, pady=15)

        # Container fix cu păstrarea proporțiilor pentru viitorul randator 3D / Video Avatar
        self.model_3d_container = ctk.CTkFrame(output_panel, fg_color="#1a1a1a", corner_radius=10)
        self.model_3d_container.grid(row=1, column=0, padx=20, pady=10, sticky="nsew")
        self.model_3d_container.grid_propagate(False)
        self.model_3d_container.grid_rowconfigure(0, weight=1)
        self.model_3d_container.grid_columnconfigure(0, weight=1)

        self.model_3d_display = ctk.CTkLabel(
            self.model_3d_container,
            text="[ Viewport Model 3D ]\n\nAspect-Ratio Locked Ready",
            image=self.empty_image,
            fg_color="transparent",
            font=ctk.CTkFont(size=14),
        )
        self.model_3d_display.grid(row=0, column=0, sticky="nsew")

        self.recognized_voice_label = ctk.CTkLabel(
            output_panel,
            text="No Word Detected",
            font=ctk.CTkFont(size=14, slant="italic"),
        )
        self.recognized_voice_label.grid(row=2, column=0, pady=15)

    def update_3d_viewport_frame(self, img_pil):
        """
        Funcție pregătită pentru când vom randa modelul 3D: primește o imagine PIL
        și o afișează în viewport-ul 3D păstrând aspect ratio-ul fără stretch.
        """
        ctk_img = self.fit_image_to_container(img_pil, self.model_3d_container)
        self.model_3d_display.configure(image=ctk_img, text="")
        self.model_3d_display.image = ctk_img

    # ==========================================
    # LOGICA DE ROUTING & HARD RESET
    # ==========================================
    def clear_video_display(self, message="[ Camera / Video Feed ]\n\nWaiting for input..."):
        """Curăță garantat ultimul frame rămas pe ecran folosind imaginea 1x1 transparentă"""
        self.video_display.configure(image=self.empty_image, text=message)
        self.video_display.image = self.empty_image

    def clear_3d_display(self, message="[ Viewport Model 3D ]\n\n"):
        """Curăță garantat viewport-ul 3D"""
        self.model_3d_display.configure(image=self.empty_image, text=message)
        self.model_3d_display.image = self.empty_image

    def reset_all_states(self):
        """Oprește toate procesele active și șterge complet datele din ambele moduri"""
        # 1. Oprire Cameră / Video + Ștergere frame
        self.stop_camera_feed()
        self.clear_video_display("[ Camera / Video Feed ]\n\nWaiting for input...")
        self.btn_camera.configure(text="Open Camera", fg_color=["#3B8ED0", "#1F6AA5"])

        # 2. Ștergere Text Tradus
        self.text_output.delete("0.0", "end")
        self.text_output.insert("0.0", "Translated text will appear here\n")

        # 3. Oprire Microfon & Resetare Undă
        self.stop_mic_feed()
        self.wave_canvas.delete("all")
        self.audio_status_overlay.configure(text="[ Status Audio / Waveform ]\n\nWaiting for input...")
        self.audio_status_overlay.place(relx=0.5, rely=0.5, anchor="center")
        self.btn_mic.configure(text="Use Microphone", fg_color="#c93b2b")

        # 4. Resetare Output 3D
        self.recognized_voice_label.configure(text="No Word Detected")
        self.clear_3d_display("[ Viewport Model 3D ]\n\n")

    def switch_mode(self, value):
        self.reset_all_states()

        if value == "Sign -> Voice":
            self.voice_to_sign_frame.grid_forget()
            self.sign_to_voice_frame.grid(row=1, column=0, sticky="nsew")
        else:
            self.sign_to_voice_frame.grid_forget()
            self.voice_to_sign_frame.grid(row=1, column=0, sticky="nsew")

    # ==========================================
    # LOGICA PENTRU CAMERĂ & VIDEO (Sign -> Voice)
    # ==========================================
    def on_camera_device_change(self, selected_cam_name):
        """Dacă utilizatorul schimbă camera din dropdown în timp ce camera merge, comutăm live"""
        if self.camera_running and self.is_live_camera:
            cam_index = self.camera_devices.get(selected_cam_name, 0)
            self.text_output.insert("end", f"\n[INFO] Switched to {selected_cam_name}...")
            self.start_video_capture(cam_index, is_live=True)

    def toggle_camera(self):
        if self.camera_running:
            self.stop_camera_feed()
            self.clear_video_display("[ Camera / Video Feed ]\n\nCamera stopped.")
            self.btn_camera.configure(text="Open Camera", fg_color=["#3B8ED0", "#1F6AA5"])
        else:
            selected_cam = self.cam_dropdown.get()
            cam_index = self.camera_devices.get(selected_cam, 0)
            self.start_video_capture(cam_index, is_live=True)
            if self.camera_running:
                self.btn_camera.configure(text="Stop Camera", fg_color="#c93b2b")
                self.text_output.insert("end", f"\n[INFO] Started {selected_cam}...")

    def load_video_file(self):
        file_path = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mkv")],
        )
        if file_path:
            nume_fisier = file_path.split("/")[-1]
            self.stop_camera_feed()
            self.btn_camera.configure(text="Open Camera", fg_color=["#3B8ED0", "#1F6AA5"])
            self.text_output.insert("end", f"\n[INFO] Playing video: {nume_fisier}...")
            self.start_video_capture(file_path, is_live=False)

    def start_video_capture(self, source, is_live=True):
        self.stop_camera_feed()
        self.is_live_camera = is_live

        if is_live:
            self.cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
            if not self.cap.isOpened():
                self.cap = cv2.VideoCapture(source)
        else:
            self.cap = cv2.VideoCapture(source)

        if not self.cap.isOpened():
            self.clear_video_display("Error: Could not open video source.")
            return

        self.camera_running = True
        self.video_display.configure(text="")
        self.update_video_frame()

    def update_video_frame(self):
        if not self.camera_running or self.cap is None:
            return

        ret, frame = self.cap.read()
        if ret:
            if self.is_live_camera:
                frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(frame_rgb)

            # Scalăm imaginea păstrând proporțiile originale (Aspect Ratio Locked)
            ctk_img = self.fit_image_to_container(img_pil, self.video_container)

            self.video_display.configure(image=ctk_img, text="")
            self.video_display.image = ctk_img

            self.video_loop_id = self.after(30, self.update_video_frame)
        else:
            self.stop_camera_feed()
            self.clear_video_display("[ Video Finished ]")
            self.btn_camera.configure(text="Open Camera", fg_color=["#3B8ED0", "#1F6AA5"])

    def stop_camera_feed(self):
        self.camera_running = False
        if self.video_loop_id is not None:
            self.after_cancel(self.video_loop_id)
            self.video_loop_id = None
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    # ==========================================
    # LOGICA PENTRU MICROFON & UNDĂ (Voice -> Sign)
    # ==========================================
    def on_mic_device_change(self, selected_mic_name):
        """Dacă utilizatorul schimbă microfonul în timp ce ascultă, repornim fluxul pe noul microfon"""
        if self.mic_running:
            self.start_mic_feed()

    def toggle_mic(self):
        if self.mic_running:
            self.stop_mic_feed()
            self.wave_canvas.delete("all")
            self.audio_status_overlay.configure(text="[ Status Audio / Waveform ]\n\nMicrophone stopped.")
            self.audio_status_overlay.place(relx=0.5, rely=0.5, anchor="center")
            self.btn_mic.configure(text="Use Microphone", fg_color="#c93b2b")
            self.recognized_voice_label.configure(text="No Word Detected")
        else:
            self.start_mic_feed()

    def audio_callback(self, indata, frames, time, status):
        rms = np.sqrt(np.mean(indata**2))
        self.current_volume = min(rms * 35.0, 1.0)

    def start_mic_feed(self):
        self.stop_mic_feed()
        selected_mic_name = self.mic_dropdown.get()
        mic_index = self.mic_devices.get(selected_mic_name, None)

        try:
            self.audio_stream = sd.InputStream(
                device=mic_index,
                callback=self.audio_callback,
                channels=1,
                samplerate=44100,
                blocksize=1024
            )
            self.audio_stream.start()
            self.mic_running = True
            self.audio_status_overlay.place_forget()
            self.btn_mic.configure(text="Stop Microphone", fg_color="#4a4a4a")
            self.recognized_voice_label.configure(text=f"Listening on {selected_mic_name}...")
            self.animate_waveform()
        except Exception as e:
            self.audio_status_overlay.configure(text=f"Mic Error:\n{e}")
            self.audio_status_overlay.place(relx=0.5, rely=0.5, anchor="center")

    def animate_waveform(self):
        if not self.mic_running:
            return

        self.wave_canvas.delete("all")
        w = self.wave_canvas.winfo_width()
        h = self.wave_canvas.winfo_height()
        if w < 50 or h < 50:
            w, h = 400, 300

        mid_y = h / 2
        max_amp = (h / 2) - 15
        amplitude = 3 + (self.current_volume * max_amp)

        self.wave_phase += 0.15 + (self.current_volume * 0.35)

        if self.current_volume > 0.75:
            color = "#ff3b30"
            line_width = 4
        elif self.current_volume > 0.08:
            color = "#30d158"
            line_width = 3
        else:
            color = "#3B8ED0"
            line_width = 2

        points = []
        for x in range(0, w, 4):
            norm_x = x / w
            envelope = math.sin(norm_x * math.pi)
            y_offset = (
                math.sin((x * 0.03) + self.wave_phase) * 0.7 +
                math.sin((x * 0.07) - self.wave_phase * 1.5) * 0.3
            )
            y = mid_y + (y_offset * amplitude * envelope)
            points.extend([x, y])

        if len(points) >= 4:
            self.wave_canvas.create_line(points, fill=color, width=line_width, smooth=True)

        self.wave_loop_id = self.after(30, self.animate_waveform)

    def stop_mic_feed(self):
        self.mic_running = False
        if self.wave_loop_id is not None:
            self.after_cancel(self.wave_loop_id)
            self.wave_loop_id = None
        if self.audio_stream is not None:
            self.audio_stream.stop()
            self.audio_stream.close()
            self.audio_stream = None
        self.current_volume = 0.0

    def load_audio_file(self):
        file_path = filedialog.askopenfilename(
            title="Select Audio File",
            filetypes=[("Audio Files", "*.wav *.mp3 *.m4a")],
        )
        if file_path:
            self.stop_mic_feed()
            self.wave_canvas.delete("all")
            self.btn_mic.configure(text="Use Microphone", fg_color="#c93b2b")

            nume_fisier = file_path.split("/")[-1]
            self.audio_status_overlay.configure(text=f"Audio File Loaded:\n{nume_fisier}")
            self.audio_status_overlay.place(relx=0.5, rely=0.5, anchor="center")
            self.recognized_voice_label.configure(text=f"Ready to process: {nume_fisier}")

    def on_closing(self):
        """Curățare hardware când închizi aplicația de la X"""
        self.reset_all_states()
        self.destroy()


if __name__ == "__main__":
    app = SignLanguageApp()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()