import os
import re
import json
import cv2
import joblib
import numpy as np
import customtkinter as ctk
from tkinter import filedialog
from PIL import Image
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from ai.hand_tracker import HandTracker
from ai.feature_extractor import SignFeatureExtractor
from hardware.device_scanner import DeviceScanner
from utils.ui_helpers import create_empty_image, fit_image_to_container

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")


class DevTrainerStudio(ctk.CTk):
    """
    Unealtă exclusivă pentru Developer (separată de aplicația finală a utilizatorului).
    Permite:
    1. Înregistrarea unui semn cu camera și etichetarea lui ulterioară ("ce ți-am arătat a fost X").
    2. Importul de fișiere video (ex: 'salut.mp4' -> învață automat cuvântul 'SALUT').
    3. Antrenarea și exportul modelului AI optimizat în `src/data/sign_model.pkl`.
    """

    def __init__(self):
        super().__init__()

        self.title("DEV ONLY - AI Sign Language Trainer Studio")
        self.geometry("1120x740")
        self.minsize(950, 620)

        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.data_dir = os.path.join(self.base_dir, "data")
        os.makedirs(self.data_dir, exist_ok=True)

        self.dataset_path = os.path.join(self.data_dir, "signs_dataset.json")
        self.model_path = os.path.join(self.data_dir, "sign_model.pkl")

        self.hand_tracker = HandTracker()
        self.empty_image = create_empty_image()
        self.camera_devices, _, _ = DeviceScanner.scan_all()

        self.cap = None
        self.camera_running = False
        self.video_loop_id = None

        # Stare pentru înregistrarea secvenței curente
        self.is_recording = False
        self.recorded_Buffer = []  # Păstrează cadrele ultimei execuții până îi dai Label

        self.dataset = self._load_dataset()

        self._build_ui()
        self._refresh_stats()

    def _load_dataset(self):
        if os.path.exists(self.dataset_path):
            try:
                with open(self.dataset_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return {"samples": []}  # Fiecare sample: {"label": "SALUT", "features": [...]}

    def _save_dataset(self):
        with open(self.dataset_path, "w", encoding="utf-8") as f:
            json.dump(self.dataset, f)

    @staticmethod
    def _clean_label_from_filename(filepath):
        """
        Extrage numele cuvântului din numele fișierului:
        'salut.mp4' -> 'SALUT'
        'libelula_2.mp4' -> 'LIBELULA'
        'unt (1).avi' -> 'UNT'
        """
        base = os.path.splitext(os.path.basename(filepath))[0]
        # Eliminăm numerele de indexare de la final (_1, -2, (1))
        cleaned = re.sub(r"[\s_\-]*(\(\d+\)|\d+)$", "", base).strip()
        return cleaned.upper() if cleaned else base.upper()

    def _build_ui(self):
        self.grid_columnconfigure(0, weight=3)
        self.grid_columnconfigure(1, weight=2)
        self.grid_rowconfigure(0, weight=1)

        # ==========================================
        # STÂNGA: CAMERĂ & ÎNREGISTRARE LIVE
        # ==========================================
        left_panel = ctk.CTkFrame(self, corner_radius=15)
        left_panel.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")
        left_panel.grid_rowconfigure(2, weight=1)
        left_panel.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            left_panel,
            text="1. LIVE CAPTURE & POST-LABELING",
            font=ctk.CTkFont(size=17, weight="bold"),
        ).grid(row=0, column=0, pady=(15, 5))

        cam_bar = ctk.CTkFrame(left_panel, fg_color="transparent")
        cam_bar.grid(row=1, column=0, padx=20, pady=5, sticky="ew")

        self.cam_dropdown = ctk.CTkOptionMenu(
            cam_bar, values=list(self.camera_devices.keys()), width=200
        )
        self.cam_dropdown.pack(side="left", padx=(0, 10))

        self.btn_cam = ctk.CTkButton(
            cam_bar, text="Open Camera", command=self.toggle_camera, width=140
        )
        self.btn_cam.pack(side="left")

        self.video_container = ctk.CTkFrame(left_panel, fg_color="#1a1a1a", corner_radius=10)
        self.video_container.grid(row=2, column=0, padx=20, pady=10, sticky="nsew")
        self.video_container.grid_propagate(False)
        self.video_container.grid_rowconfigure(0, weight=1)
        self.video_container.grid_columnconfigure(0, weight=1)

        self.video_display = ctk.CTkLabel(
            self.video_container,
            text="Open camera to record a sign...",
            image=self.empty_image,
            font=ctk.CTkFont(size=14),
        )
        self.video_display.grid(row=0, column=0, sticky="nsew")

        # Controale de Record -> apoi Label
        rec_frame = ctk.CTkFrame(left_panel, fg_color="#242424", corner_radius=10)
        rec_frame.grid(row=3, column=0, padx=20, pady=(5, 15), sticky="ew")
        rec_frame.grid_columnconfigure(1, weight=1)

        self.btn_record = ctk.CTkButton(
            rec_frame,
            text="● Start Recording Gesture",
            fg_color="#c93b2b",
            hover_color="#9e2a1b",
            command=self.toggle_recording,
            width=190,
            height=38,
        )
        self.btn_record.grid(row=0, column=0, padx=12, pady=12)

        self.entry_label = ctk.CTkEntry(
            rec_frame,
            placeholder_text="Ce cuvânt/semn ai arătat? (ex: SALUT, UNT, LIBELULA)",
            height=38,
        )
        self.entry_label.grid(row=0, column=1, padx=8, pady=12, sticky="ew")

        self.btn_save_live = ctk.CTkButton(
            rec_frame,
            text="Save as Label",
            fg_color="#2d8a4e",
            hover_color="#236b3c",
            command=self.save_recorded_gesture,
            width=130,
            height=38,
            state="disabled",
        )
        self.btn_save_live.grid(row=0, column=2, padx=12, pady=12)

        # ==========================================
        # DREAPTA: UPLOAD VIDEO & ANTRENARE AI
        # ==========================================
        right_panel = ctk.CTkFrame(self, corner_radius=15)
        right_panel.grid(row=0, column=1, padx=(0, 20), pady=20, sticky="nsew")
        right_panel.grid_rowconfigure(3, weight=1)
        right_panel.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            right_panel,
            text="2. VIDEO FILE INGESTION (AUTO-LABEL)",
            font=ctk.CTkFont(size=17, weight="bold"),
        ).grid(row=0, column=0, pady=(15, 5))

        ctk.CTkLabel(
            right_panel,
            text="Selectează unul sau mai multe fișiere (ex: salut.mp4, unt.mp4).\nNumele fișierului devine automat cuvântul învățat!",
            font=ctk.CTkFont(size=12),
            text_color="#aaaaaa",
        ).grid(row=1, column=0, padx=20, pady=5)

        self.btn_upload_videos = ctk.CTkButton(
            right_panel,
            text="📁 Import Video Files (Auto-Label)",
            fg_color="#2b5b84",
            height=40,
            command=self.import_video_files,
        )
        self.btn_upload_videos.grid(row=2, column=0, padx=20, pady=10, sticky="ew")

        # Jurnal și Statistici Dataset
        self.log_box = ctk.CTkTextbox(right_panel, font=ctk.CTkFont(size=13), wrap="word")
        self.log_box.grid(row=3, column=0, padx=20, pady=10, sticky="nsew")

        # Buton de ștergere a unui cuvânt greșit + Butonul principal de Train
        bottom_bar = ctk.CTkFrame(right_panel, fg_color="transparent")
        bottom_bar.grid(row=4, column=0, padx=20, pady=(5, 15), sticky="ew")
        bottom_bar.grid_columnconfigure(0, weight=1)

        self.btn_train = ctk.CTkButton(
            bottom_bar,
            text="⚡ TRAIN & EXPORT AI MODEL",
            font=ctk.CTkFont(size=15, weight="bold"),
            fg_color="#d97706",
            hover_color="#b45309",
            height=46,
            command=self.train_and_export_model,
        )
        self.btn_train.grid(row=0, column=0, pady=(0, 8), sticky="ew")

        self.btn_clear_db = ctk.CTkButton(
            bottom_bar,
            text="Reset Dataset",
            fg_color="#4a4a4a",
            hover_color="#333333",
            height=30,
            command=self.clear_dataset,
        )
        self.btn_clear_db.grid(row=1, column=0, sticky="ew")

    def log(self, message):
        self.log_box.insert("end", f"{message}\n")
        self.log_box.see("end")
        self.update_idletasks()

    def _refresh_stats(self):
        self.log_box.delete("0.0", "end")
        samples = self.dataset.get("samples", [])
        counts = {}
        for s in samples:
            lbl = s["label"]
            counts[lbl] = counts.get(lbl, 0) + 1

        self.log("=== DATASET SUMMARY ===")
        self.log(f"Total înregistrări (cu augmentare 9x): {len(samples)}")
        self.log(f"Cuvinte/Semne unice în dicționar: {len(counts)}\n")
        for lbl, cnt in sorted(counts.items()):
            self.log(f"  • {lbl}: {cnt} vectori de antrenament")
        self.log("========================\n")

    # ==========================================
    # FLUXUL 1: CAMERĂ LIVE & POST-LABELING
    # ==========================================
    def toggle_camera(self):
        if self.camera_running:
            self.stop_camera()
            self.video_display.configure(image=self.empty_image, text="Camera stopped.")
            self.btn_cam.configure(text="Open Camera")
        else:
            cam_name = self.cam_dropdown.get()
            cam_idx = self.camera_devices.get(cam_name, 0)
            self.cap = cv2.VideoCapture(cam_idx, cv2.CAP_DSHOW)
            if not self.cap.isOpened():
                self.cap = cv2.VideoCapture(cam_idx)
            if self.cap.isOpened():
                self.camera_running = True
                self.btn_cam.configure(text="Stop Camera")
                self.update_camera_loop()

    def update_camera_loop(self):
        if not self.camera_running or self.cap is None:
            return

        ret, frame = self.cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            annotated, sign_template = self.hand_tracker.process_frame(frame)

            if self.is_recording:
                feat = SignFeatureExtractor.extract_frame_features(sign_template)
                if feat is not None:
                    self.recorded_Buffer.append(feat)
                # Indicator vizual REC pe ecran
                cv2.putText(
                    annotated,
                    f"REC ({len(self.recorded_Buffer)} frames)",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 0, 255),
                    2,
                )

            rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
            ctk_img = fit_image_to_container(pil_img, self.video_container)
            self.video_display.configure(image=ctk_img, text="")
            self.video_display.image = ctk_img

        self.video_loop_id = self.after(30, self.update_camera_loop)

    def stop_camera(self):
        self.camera_running = False
        if self.video_loop_id:
            self.after_cancel(self.video_loop_id)
            self.video_loop_id = None
        if self.cap:
            self.cap.release()
            self.cap = None

    def toggle_recording(self):
        if not self.camera_running:
            self.log("[!] Pornește camera înainte să înregistrezi un semn!")
            return

        if not self.is_recording:
            self.recorded_Buffer.clear()
            self.is_recording = True
            self.btn_record.configure(text="■ Stop Recording", fg_color="#d97706")
            self.btn_save_live.configure(state="disabled")
            self.log("[REC] Execută semnul acum în fața camerei...")
        else:
            self.is_recording = False
            self.btn_record.configure(text="● Start Recording Gesture", fg_color="#c93b2b")
            if len(self.recorded_Buffer) >= 3:
                self.btn_save_live.configure(state="normal")
                self.log(
                    f"[OK] S-au capturat {len(self.recorded_Buffer)} cadre cu mâna. "
                    f"Scrie în căsuță ce cuvânt a fost și apasă 'Save as Label'!"
                )
            else:
                self.log("[EROARE] Nu s-a detectat mâna în timpul înregistrării. Încearcă din nou!")

    def save_recorded_gesture(self):
        label = self.entry_label.get().strip().upper()
        if not label:
            self.log("[!] Scrie numele cuvântului/literei în căsuță înainte de salvare!")
            return

        if len(self.recorded_Buffer) < 3:
            self.log("[!] Nu există o înregistrare validă în memorie.")
            return

        # Generăm vectorul original + 8 variații sintetice (rotații + micro-jitter)
        aug_vectors = SignFeatureExtractor.augment_sequence(self.recorded_Buffer, num_augmentations=8)
        for vec in aug_vectors:
            self.dataset["samples"].append({
                "label": label,
                "features": [round(float(x), 5) for x in vec],
            })

        # Dacă înregistrarea e mai lungă, extragem și sub-ferestre glisante
        if len(self.recorded_Buffer) >= 18:
            mid = len(self.recorded_Buffer) // 2
            for sub_window in [self.recorded_Buffer[:mid], self.recorded_Buffer[mid:]]:
                for vec in SignFeatureExtractor.augment_sequence(sub_window, num_augmentations=4):
                    self.dataset["samples"].append({
                        "label": label,
                        "features": [round(float(x), 5) for x in vec],
                    })

        self._save_dataset()
        self.recorded_Buffer.clear()
        self.btn_save_live.configure(state="disabled")
        self.entry_label.delete(0, "end")
        self._refresh_stats()
        self.log(f"[SALVAT] Semnul '{label}' a fost adăugat în baza de date!")

    # ==========================================
    # FLUXUL 2: IMPORT FIȘIERE VIDEO (salut.mp4 -> SALUT)
    # ==========================================
    def import_video_files(self):
        file_paths = filedialog.askopenfilenames(
            title="Select Training Video Files",
            filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mkv")],
        )
        if not file_paths:
            return

        added_total = 0
        for path in file_paths:
            label = self._clean_label_from_filename(path)
            filename = os.path.basename(path)
            self.log(f"[VIDEO] Procesez '{filename}' -> Label detectat: '{label}'...")

            cap = cv2.VideoCapture(path)
            frames_feats = []
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                _, sign_template = self.hand_tracker.process_frame(frame)
                feat = SignFeatureExtractor.extract_frame_features(sign_template)
                if feat is not None:
                    frames_feats.append(feat)
            cap.release()

            if len(frames_feats) < 3:
                self.log(f"  [!] Atenție: În '{filename}' nu s-au detectat mâini clare. Ignorat.")
                continue

            # 1. Întregul videoclip ca o secvență completă + augmentări
            for vec in SignFeatureExtractor.augment_sequence(frames_feats, num_augmentations=10):
                self.dataset["samples"].append({
                    "label": label,
                    "features": [round(float(x), 5) for x in vec],
                })
                added_total += 1

            # 2. Ferestre glisante (Sliding Windows) de 20 de cadre din interiorul videoclipului
            win_size = min(20, len(frames_feats))
            step = max(3, win_size // 3)
            for start_i in range(0, max(1, len(frames_feats) - win_size + 1), step):
                sub_seq = frames_feats[start_i : start_i + win_size]
                if len(sub_seq) >= 5:
                    for vec in SignFeatureExtractor.augment_sequence(sub_seq, num_augmentations=4):
                        self.dataset["samples"].append({
                            "label": label,
                            "features": [round(float(x), 5) for x in vec],
                        })
                        added_total += 1

        self._save_dataset()
        self._refresh_stats()
        self.log(f"[FINALIZAT] S-au extras {added_total} secvențe de antrenament din fișierele video!")

    # ==========================================
    # FLUXUL 3: ANTRENAREA SI EXPORTUL CREIERULUI AI
    # ==========================================
    def train_and_export_model(self):
        samples = self.dataset.get("samples", [])
        if len(samples) < 5:
            self.log("[EROARE] Adaugă cel puțin 2 semne/cuvinte diferite înainte de antrenare!")
            return

        X = np.array([s["features"] for s in samples], dtype=np.float32)
        y = np.array([s["label"] for s in samples])

        unique_classes = np.unique(y)
        if len(unique_classes) < 2:
            self.log(
                f"[EROARE] Ai doar cuvântul '{unique_classes[0]}' în baza de date. "
                f"Un clasificator AI are nevoie de minim 2 cuvinte diferite pentru a învăța să le deosebească!"
            )
            return

        self.log(f"[AI] Antrenez modelul ExtraTrees (300 arbori) pe {len(unique_classes)} cuvinte...")

        # Pipeline cu StandardScaler + ExtraTreesClassifier (ideal pentru separarea fină în 390 dimensiuni)
        model = make_pipeline(
            StandardScaler(),
            ExtraTreesClassifier(
                n_estimators=80,
                max_depth=None,
                min_samples_split=2,
                class_weight="balanced",
                random_state=42,
                n_jobs=1,
            ),
        )
        model.fit(X, y)

        bundle = {
            "model": model,
            "classes": unique_classes.tolist(),
        }
        joblib.dump(bundle, self.model_path)
        self.log(f"[SUCCES] Modelul AI a fost salvat în:\n{self.model_path}")
        self.log(f"Cuvinte cunoscute acum de aplicația finală: {', '.join(unique_classes)}")

    def clear_dataset(self):
        self.dataset = {"samples": []}
        self._save_dataset()
        if os.path.exists(self.model_path):
            os.remove(self.model_path)
        self._refresh_stats()
        self.log("[INFO] Baza de date și modelul au fost resetate complet.")

    def on_closing(self):
        self.stop_camera()
        self.destroy()


if __name__ == "__main__":
    studio = DevTrainerStudio()
    studio.protocol("WM_DELETE_WINDOW", studio.on_closing)
    studio.mainloop()