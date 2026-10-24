import customtkinter as ctk
from tkinter import filedialog, Canvas

from translation.sign_translator import SignTranslator
from hardware.device_scanner import DeviceScanner
from hardware.video_engine import VideoEngine
from hardware.audio_engine import AudioEngine
from utils.ui_helpers import (
    create_empty_image,
    fit_image_to_container,
    compute_waveform_Render_data,
)

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")


class SignLanguageApp(ctk.CTk):

    def __init__(self):
        super().__init__()

        self.title("AI Sign Language Translator")
        self.geometry("1080x720")
        self.minsize(900, 600)

        self.empty_image = create_empty_image()

        # Motoare Hardware decuplate (SoC)
        self.sign_translator = SignTranslator()
        self.video_engine = VideoEngine()
        self.audio_engine = AudioEngine()

        self.video_loop_id = None
        self.wave_loop_id = None
        self.wave_phase = 0.0

        # Scanăm dispozitivele prin modulul dedicat
        self.camera_devices, self.mic_devices, self.speaker_devices = DeviceScanner.scan_all()

        # Layout principal
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=1)

        self._setup_header()

        self.sign_to_voice_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.voice_to_sign_frame = ctk.CTkFrame(self, fg_color="transparent")

        self.setup_sign_to_voice_ui()
        self.setup_voice_to_sign_ui()

        self.switch_mode("Sign -> Voice")

    def _setup_header(self):
        self.header_frame = ctk.CTkFrame(self, height=80, corner_radius=0)
        self.header_frame.grid(row=0, column=0, sticky="ew")
        self.header_frame.grid_columnconfigure(0, weight=1)

        self.title_label = ctk.CTkLabel(
            self.header_frame,
            text="AI Sign Language Hub",
            font=ctk.CTkFont(size=22, weight="bold"),
        )
        self.title_label.grid(row=0, column=0, pady=(15, 5))

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

    def refresh_device_lists(self):
        self.camera_devices, self.mic_devices, self.speaker_devices = DeviceScanner.scan_all()

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
    # UI: RAMURA STÂNGĂ (Sign -> Voice)
    # ==========================================
    def setup_sign_to_voice_ui(self):
        self.sign_to_voice_frame.grid_columnconfigure(0, weight=1, uniform="group1")
        self.sign_to_voice_frame.grid_columnconfigure(1, weight=1, uniform="group1")
        self.sign_to_voice_frame.grid_rowconfigure(0, weight=1)

        input_panel = ctk.CTkFrame(self.sign_to_voice_frame, corner_radius=15)
        input_panel.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")
        input_panel.grid_rowconfigure(2, weight=1)
        input_panel.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            input_panel, text="SIGN INPUT SOURCE", font=ctk.CTkFont(size=16, weight="bold")
        ).grid(row=0, column=0, pady=(15, 5))

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
            cam_select_frame, text="⟳", width=32, height=28,
            fg_color="#3a3a3a", hover_color="#505050", command=self.refresh_device_lists
        ).pack(side="left", padx=(8, 0))

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

        btn_frame = ctk.CTkFrame(input_panel, fg_color="transparent")
        btn_frame.grid(row=3, column=0, pady=15)

        self.btn_camera = ctk.CTkButton(
            btn_frame, text="Open Camera", command=self.toggle_camera, width=150, height=35
        )
        self.btn_camera.pack(side="left", padx=10)

        self.btn_video = ctk.CTkButton(
            btn_frame, text="Upload Video File", command=self.load_video_file,
            fg_color="#2b5b84", width=150, height=35
        )
        self.btn_video.pack(side="left", padx=10)

        output_panel = ctk.CTkFrame(self.sign_to_voice_frame, corner_radius=15)
        output_panel.grid(row=0, column=1, padx=20, pady=20, sticky="nsew")
        output_panel.grid_rowconfigure(2, weight=1)
        output_panel.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            output_panel, text="OUTPUT TEXT & VOICE", font=ctk.CTkFont(size=16, weight="bold")
        ).grid(row=0, column=0, pady=(15, 5))

        spk_select_frame = ctk.CTkFrame(output_panel, fg_color="transparent")
        spk_select_frame.grid(row=1, column=0, padx=20, pady=(0, 5), sticky="ew")

        ctk.CTkLabel(spk_select_frame, text="Audio Out:", font=ctk.CTkFont(size=13)).pack(side="left", padx=(0, 8))
        self.speaker_dropdown = ctk.CTkOptionMenu(
            spk_select_frame, values=list(self.speaker_devices.keys()), width=220, height=28
        )
        self.speaker_dropdown.pack(side="left", fill="x", expand=True)

        self.text_output = ctk.CTkTextbox(
            output_panel, font=ctk.CTkFont(size=16), corner_radius=10, wrap="word"
        )
        self.text_output.grid(row=2, column=0, padx=20, pady=10, sticky="nsew")
        self.text_output.insert("0.0", "Translated text will appear here\n")

        self.btn_clear_text = ctk.CTkButton(
            output_panel, text="Delete Text", fg_color="#4a4a4a", hover_color="#333333",
            command=lambda: self.text_output.delete("0.0", "end")
        )
        self.btn_clear_text.grid(row=3, column=0, pady=15)

    # ==========================================
    # UI: RAMURA DREAPTĂ (Voice -> Sign)
    # ==========================================
    def setup_voice_to_sign_ui(self):
        self.voice_to_sign_frame.grid_columnconfigure(0, weight=1, uniform="group1")
        self.voice_to_sign_frame.grid_columnconfigure(1, weight=1, uniform="group1")
        self.voice_to_sign_frame.grid_rowconfigure(0, weight=1)

        input_panel = ctk.CTkFrame(self.voice_to_sign_frame, corner_radius=15)
        input_panel.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")
        input_panel.grid_rowconfigure(2, weight=1)
        input_panel.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            input_panel, text="VOICE INPUT SOURCE", font=ctk.CTkFont(size=16, weight="bold")
        ).grid(row=0, column=0, pady=(15, 5))

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
            mic_select_frame, text="⟳", width=32, height=28,
            fg_color="#3a3a3a", hover_color="#505050", command=self.refresh_device_lists
        ).pack(side="left", padx=(8, 0))

        self.wave_container = ctk.CTkFrame(input_panel, fg_color="#1a1a1a", corner_radius=10)
        self.wave_container.grid(row=2, column=0, padx=20, pady=10, sticky="nsew")
        self.wave_container.grid_propagate(False)
        self.wave_container.grid_rowconfigure(0, weight=1)
        self.wave_container.grid_columnconfigure(0, weight=1)

        self.wave_canvas = Canvas(self.wave_container, bg="#1a1a1a", highlightthickness=0)
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
            btn_frame, text="Use Microphone", command=self.toggle_mic,
            fg_color="#c93b2b", hover_color="#9e2a1b", width=150, height=35
        )
        self.btn_mic.pack(side="left", padx=10)

        self.btn_audio_file = ctk.CTkButton(
            btn_frame, text="Upload Audio File", command=self.load_audio_file,
            fg_color="#2b5b84", width=150, height=35
        )
        self.btn_audio_file.pack(side="left", padx=10)

        output_panel = ctk.CTkFrame(self.voice_to_sign_frame, corner_radius=15)
        output_panel.grid(row=0, column=1, padx=20, pady=20, sticky="nsew")
        output_panel.grid_rowconfigure(1, weight=1)
        output_panel.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            output_panel, text="OUTPUT 3D MODEL", font=ctk.CTkFont(size=16, weight="bold")
        ).grid(row=0, column=0, pady=15)

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
            output_panel, text="No Word Detected", font=ctk.CTkFont(size=14, slant="italic")
        )
        self.recognized_voice_label.grid(row=2, column=0, pady=15)

    def update_3d_viewport_frame(self, img_pil):
        ctk_img = fit_image_to_container(img_pil, self.model_3d_container)
        self.model_3d_display.configure(image=ctk_img, text="")
        self.model_3d_display.image = ctk_img

    # ==========================================
    # LOGICA DE ROUTING & HARD RESET
    # ==========================================
    def clear_video_display(self, message="[ Camera / Video Feed ]\n\nWaiting for input..."):
        self.video_display.configure(image=self.empty_image, text=message)
        self.video_display.image = self.empty_image

    def clear_3d_display(self, message="[ Viewport Model 3D ]\n\n"):
        self.model_3d_display.configure(image=self.empty_image, text=message)
        self.model_3d_display.image = self.empty_image

    def reset_all_states(self):
        self.stop_camera_feed()
        self.clear_video_display("[ Camera / Video Feed ]\n\nWaiting for input...")
        self.btn_camera.configure(text="Open Camera", fg_color=["#3B8ED0", "#1F6AA5"])

        self.text_output.delete("0.0", "end")
        self.text_output.insert("0.0", "Translated text will appear here\n")

        self.stop_mic_feed()
        self.wave_canvas.delete("all")
        self.audio_status_overlay.configure(text="[ Status Audio / Waveform ]\n\nWaiting for input...")
        self.audio_status_overlay.place(relx=0.5, rely=0.5, anchor="center")
        self.btn_mic.configure(text="Use Microphone", fg_color="#c93b2b")

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
    # CONTROL VIDEO (Deleagă către VideoEngine)
    # ==========================================
    def on_camera_device_change(self, selected_cam_name):
        if self.video_engine.is_running and self.video_engine.is_live:
            cam_index = self.camera_devices.get(selected_cam_name, 0)
            self.text_output.insert("end", f"\n[INFO] Switched to {selected_cam_name}...")
            self.start_video_capture(cam_index, is_live=True)

    def toggle_camera(self):
        if self.video_engine.is_running:
            self.stop_camera_feed()
            self.clear_video_display("[ Camera / Video Feed ]\n\nCamera stopped.")
            self.btn_camera.configure(text="Open Camera", fg_color=["#3B8ED0", "#1F6AA5"])
        else:
            selected_cam = self.cam_dropdown.get()
            cam_index = self.camera_devices.get(selected_cam, 0)
            self.start_video_capture(cam_index, is_live=True)
            if self.video_engine.is_running:
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
        if not self.video_engine.start(source, is_live=is_live):
            self.clear_video_display("Error: Could not open video source.")
            return
        self.video_display.configure(text="")
        self.update_video_frame()

    def update_video_frame(self):
        if not self.video_engine.is_running:
            return

        ret, img_pil, sign_template = self.video_engine.get_frame()
        if ret and img_pil is not None:
            ctk_img = fit_image_to_container(img_pil, self.video_container)
            self.video_display.configure(image=ctk_img, text="")
            self.video_display.image = ctk_img
            # Trimitem template-ul generic către Zona Albastră (SignTranslator)
            if sign_template is not None:
                translated_word = self.sign_translator.translate(sign_template)
                if translated_word:
                    self.text_output.insert("end", f"{translated_word}\n")
                    self.text_output.see("end")  # Auto-scroll jos
            self.video_loop_id = self.after(30, self.update_video_frame)
        else:
            self.stop_camera_feed()
            self.clear_video_display("[ Video Finished ]")
            self.btn_camera.configure(text="Open Camera", fg_color=["#3B8ED0", "#1F6AA5"])


    def stop_camera_feed(self):
        if self.video_loop_id is not None:
            self.after_cancel(self.video_loop_id)
            self.video_loop_id = None
        self.video_engine.stop()

    # ==========================================
    # CONTROL AUDIO (Deleagă către AudioEngine)
    # ==========================================
    def on_mic_device_change(self, selected_mic_name):
        if self.audio_engine.is_running:
            self.start_mic_feed()

    def toggle_mic(self):
        if self.audio_engine.is_running:
            self.stop_mic_feed()
            self.wave_canvas.delete("all")
            self.audio_status_overlay.configure(text="[ Status Audio / Waveform ]\n\nMicrophone stopped.")
            self.audio_status_overlay.place(relx=0.5, rely=0.5, anchor="center")
            self.btn_mic.configure(text="Use Microphone", fg_color="#c93b2b")
            self.recognized_voice_label.configure(text="No Word Detected")
        else:
            self.start_mic_feed()

    def start_mic_feed(self):
        self.stop_mic_feed()
        selected_mic_name = self.mic_dropdown.get()
        mic_index = self.mic_devices.get(selected_mic_name, None)

        try:
            self.audio_engine.start(device_index=mic_index)
            self.audio_status_overlay.place_forget()
            self.btn_mic.configure(text="Stop Microphone", fg_color="#4a4a4a")
            self.recognized_voice_label.configure(text=f"Listening on {selected_mic_name}...")
            self.animate_waveform()
        except Exception as e:
            self.audio_status_overlay.configure(text=f"Mic Error:\n{e}")
            self.audio_status_overlay.place(relx=0.5, rely=0.5, anchor="center")

    def animate_waveform(self):
        if not self.audio_engine.is_running:
            return

        self.wave_canvas.delete("all")
        w = self.wave_canvas.winfo_width()
        h = self.wave_canvas.winfo_height()

        points, color, line_width, self.wave_phase = compute_waveform_Render_data(
            w, h, self.audio_engine.current_volume, self.wave_phase
        )

        if len(points) >= 4:
            self.wave_canvas.create_line(points, fill=color, width=line_width, smooth=True)

        self.wave_loop_id = self.after(30, self.animate_waveform)

    def stop_mic_feed(self):
        if self.wave_loop_id is not None:
            self.after_cancel(self.wave_loop_id)
            self.wave_loop_id = None
        self.audio_engine.stop()

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
        self.reset_all_states()
        self.destroy()
