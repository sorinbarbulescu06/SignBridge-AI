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

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class BulkDatasetModal(ctk.CTkToplevel):
    """
    Fereastră user-friendly care scanează un folder de dataset (cu subfoldere sau fișiere directe),
    afișează toate cuvintele/semnele găsite și îl lasă pe developer să bifeze exact ce vrea să învețe AI-ul.
    """

    def __init__(self, parent, folder_path, grouped_files, on_confirm_callback):
        super().__init__(parent)
        self.title("Bulk Dataset Manager - Selectează ce vrei să învețe AI-ul")
        self.geometry("720x640")
        self.minsize(620, 520)
        self.transient(parent)
        self.grab_set()

        self.folder_path = folder_path
        self.grouped_files = grouped_files  # { "SALUT": [path1, path2, ...], ... }
        self.on_confirm_callback = on_confirm_callback
        self.checkbox_vars = {}
        self.checkbox_widgets = {}

        self._build_modal_ui()

    def _build_modal_ui(self):
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(3, weight=1)

        # Titlu și cale
        header = ctk.CTkFrame(self, fg_color="transparent")
        header.grid(row=0, column=0, padx=20, pady=(15, 5), sticky="ew")

        ctk.CTkLabel(
            header,
            text=f"📦 Dataset găsit: {len(self.grouped_files)} cuvinte / categorii detectate",
            font=ctk.CTkFont(size=17, weight="bold"),
        ).pack(anchor="w")

        ctk.CTkLabel(
            header,
            text=f"Sursă: {self.folder_path}",
            font=ctk.CTkFont(size=12),
            text_color="#999999",
        ).pack(anchor="w")

        # Bara de căutare și selecție rapidă
        toolbar = ctk.CTkFrame(self, corner_radius=10)
        toolbar.grid(row=1, column=0, padx=20, pady=8, sticky="ew")
        toolbar.grid_columnconfigure(0, weight=1)

        self.search_entry = ctk.CTkEntry(
            toolbar,
            placeholder_text="🔍 Caută un cuvânt sau o literă (ex: SALUT, A, WATER)...",
            height=34,
        )
        self.search_entry.grid(row=0, column=0, padx=10, pady=10, sticky="ew")
        self.search_entry.bind("<KeyRelease>", self._filter_list)

        ctk.CTkButton(
            toolbar, text="Select All", width=95, height=34, command=self.select_all
        ).grid(row=0, column=1, padx=5, pady=10)

        ctk.CTkButton(
            toolbar,
            text="Deselect All",
            width=95,
            height=34,
            fg_color="#4a4a4a",
            hover_color="#333333",
            command=self.deselect_all,
        ).grid(row=0, column=2, padx=(5, 10), pady=10)

        # Opțiune de limitare a numărului de fișiere per cuvânt (pentru viteză)
        limit_frame = ctk.CTkFrame(self, fg_color="transparent")
        limit_frame.grid(row=2, column=0, padx=20, pady=(0, 5), sticky="ew")

        ctk.CTkLabel(
            limit_frame,
            text="Maxim fișiere procesate per cuvânt (pentru import rapid):",
            font=ctk.CTkFont(size=13),
        ).pack(side="left", padx=(0, 10))

        self.max_files_menu = ctk.CTkOptionMenu(
            limit_frame,
            values=["10 fișiere / cuvânt", "25 fișiere / cuvânt", "50 fișiere / cuvânt", "Toate fișierele"],
            width=170,
        )
        self.max_files_menu.set("25 fișiere / cuvânt")
        self.max_files_menu.pack(side="left")

        # Lista scrollabilă cu Checkboxes pentru fiecare cuvânt/fișier
        self.scroll_frame = ctk.CTkScrollableFrame(
            self, label_text="Bifează cuvintele/fișierele pe care vrei să le adaugi (Append) în Dataset:"
        )
        self.scroll_frame.grid(row=3, column=0, padx=20, pady=8, sticky="nsew")
        self.scroll_frame.grid_columnconfigure(0, weight=1)

        for idx, (label, paths) in enumerate(sorted(self.grouped_files.items())):
            var = ctk.BooleanVar(value=True)
            self.checkbox_vars[label] = var

            vids = sum(1 for p in paths if os.path.splitext(p)[1].lower() in VIDEO_EXTS)
            imgs = sum(1 for p in paths if os.path.splitext(p)[1].lower() in IMAGE_EXTS)
            details = []
            if vids:
                details.append(f"{vids} video")
            if imgs:
                details.append(f"{imgs} imagini")
            info_str = ", ".join(details)

            cb = ctk.CTkCheckBox(
                self.scroll_frame,
                text=f"{label}   ({info_str})",
                variable=var,
                font=ctk.CTkFont(size=13, weight="bold"),
            )
            cb.grid(row=idx, column=0, padx=10, pady=5, sticky="w")
            self.checkbox_widgets[label] = cb

        # Butoanele de acțiune jos
        footer = ctk.CTkFrame(self, fg_color="transparent")
        footer.grid(row=4, column=0, padx=20, pady=15, sticky="ew")
        footer.grid_columnconfigure(0, weight=1)

        self.btn_confirm = ctk.CTkButton(
            footer,
            text="🚀 Append Selected to Dataset (Learn)",
            font=ctk.CTkFont(size=15, weight="bold"),
            fg_color="#2d8a4e",
            hover_color="#236b3c",
            height=44,
            command=self._confirm_selection,
        )
        self.btn_confirm.grid(row=0, column=0, sticky="ew")

    def _filter_list(self, _event=None):
        query = self.search_entry.get().strip().upper()
        row_idx = 0
        for label, cb in self.checkbox_widgets.items():
            if not query or query in label:
                cb.grid(row=row_idx, column=0, padx=10, pady=5, sticky="w")
                row_idx += 1
            else:
                cb.grid_forget()

    def select_all(self):
        query = self.search_entry.get().strip().upper()
        for label, var in self.checkbox_vars.items():
            if not query or query in label:
                var.set(True)

    def deselect_all(self):
        query = self.search_entry.get().strip().upper()
        for label, var in self.checkbox_vars.items():
            if not query or query in label:
                var.set(False)

    def _confirm_selection(self):
        limit_str = self.max_files_menu.get()
        if "10" in limit_str:
            max_per_class = 10
        elif "25" in limit_str:
            max_per_class = 25
        elif "50" in limit_str:
            max_per_class = 50
        else:
            max_per_class = None

        selected_map = {}
        for label, var in self.checkbox_vars.items():
            if var.get():
                files = self.grouped_files[label]
                selected_map[label] = files[:max_per_class] if max_per_class else files

        self.destroy()
        if selected_map:
            self.on_confirm_callback(selected_map)


class DevTrainerStudio(ctk.CTk):
    """
    Unealtă exclusivă pentru Developer (separată de aplicația finală).
    Toate metodele de învățare (Live, Single/Multi File și Bulk Folder) fac APPEND
    la baza de date existentă (`signs_dataset.json`), fără să șteargă ce a învățat anterior.
    """

    def __init__(self):
        super().__init__()

        self.title("DEV ONLY - AI Sign Language Trainer Studio")
        self.geometry("1180x780")
        self.minsize(980, 660)

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

        self.is_recording = False
        self.recorded_Buffer = []

        self.dataset = self._load_dataset()

        self._build_ui()
        self._refresh_stats()

    def _load_dataset(self):
        if os.path.exists(self.dataset_path):
            try:
                with open(self.dataset_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if "samples" in data and isinstance(data["samples"], list):
                        return data
            except Exception:
                pass
        return {"samples": []}

    def _save_dataset(self):
        with open(self.dataset_path, "w", encoding="utf-8") as f:
            json.dump(self.dataset, f)

    @staticmethod
    def _clean_label_from_filename(filepath):
        """
        Extrage numele cuvântului din numele fișierului:
        'salut.mp4' -> 'SALUT', 'libelula_2.mp4' -> 'LIBELULA'
        """
        base = os.path.splitext(os.path.basename(filepath))[0]
        cleaned = re.sub(r"[\s_\-]*(\(\d+\)|\d+)$", "", base).strip()
        return cleaned.upper() if cleaned else base.upper()

    def _build_ui(self):
        self.grid_columnconfigure(0, weight=3)
        self.grid_columnconfigure(1, weight=2)
        self.grid_rowconfigure(0, weight=1)

        # ==========================================
        # STÂNGA: CAMERĂ & ÎNREGISTRARE LIVE MANUALĂ
        # ==========================================
        left_panel = ctk.CTkFrame(self, corner_radius=15)
        left_panel.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")
        left_panel.grid_rowconfigure(2, weight=1)
        left_panel.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            left_panel,
            text="1. LIVE CAPTURE & MANUAL LABELING",
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
            placeholder_text="Ce cuvânt/semn ai arătat? (ex: SALUT, UNT, A)",
            height=38,
        )
        self.entry_label.grid(row=0, column=1, padx=8, pady=12, sticky="ew")

        self.btn_save_live = ctk.CTkButton(
            rec_frame,
            text="Append Label",
            fg_color="#2d8a4e",
            hover_color="#236b3c",
            command=self.save_recorded_gesture,
            width=130,
            height=38,
            state="disabled",
        )
        self.btn_save_live.grid(row=0, column=2, padx=12, pady=12)

        # ==========================================
        # DREAPTA: BULK DATASET & FILE INGESTION (APPEND)
        # ==========================================
        right_panel = ctk.CTkFrame(self, corner_radius=15)
        right_panel.grid(row=0, column=1, padx=(0, 20), pady=20, sticky="nsew")
        right_panel.grid_rowconfigure(4, weight=1)
        right_panel.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            right_panel,
            text="2. BULK & FILE DATASET INGESTION (APPEND)",
            font=ctk.CTkFont(size=17, weight="bold"),
        ).grid(row=0, column=0, pady=(15, 5))

        ctk.CTkLabel(
            right_panel,
            text="Adaugă semne noi peste ce știe deja din fișiere individuale\nsau din foldere întregi cu dataset-uri (Video & Imagini).",
            font=ctk.CTkFont(size=12),
            text_color="#aaaaaa",
        ).grid(row=1, column=0, padx=20, pady=2)

        ingest_btns = ctk.CTkFrame(right_panel, fg_color="transparent")
        ingest_btns.grid(row=2, column=0, padx=20, pady=8, sticky="ew")
        ingest_btns.grid_columnconfigure(0, weight=1)

        # Butonul nou de BULK READ cu selector user-friendly
        self.btn_bulk_folder = ctk.CTkButton(
            ingest_btns,
            text="📦 Open Bulk Dataset Folder (Select & Learn)",
            font=ctk.CTkFont(size=13, weight="bold"),
            fg_color="#6d28d9",
            hover_color="#5b21b6",
            height=40,
            command=self.open_bulk_dataset_wizard,
        )
        self.btn_bulk_folder.grid(row=0, column=0, pady=(0, 6), sticky="ew")

        # Butonul de selecție directă de fișiere (cu posibilitate de label manual sau auto)
        self.btn_upload_files = ctk.CTkButton(
            ingest_btns,
            text="📁 Select Specific Files (Video / Images)",
            fg_color="#2b5b84",
            height=36,
            command=self.import_selected_files,
        )
        self.btn_upload_files.grid(row=1, column=0, sticky="ew")

        # Bară de progres pentru importurile Bulk
        self.progress_bar = ctk.CTkProgressBar(right_panel, height=10)
        self.progress_bar.grid(row=3, column=0, padx=20, pady=(2, 4), sticky="ew")
        self.progress_bar.set(0)

        # Jurnal și Statistici Dataset
        self.log_box = ctk.CTkTextbox(right_panel, font=ctk.CTkFont(size=13), wrap="word")
        self.log_box.grid(row=4, column=0, padx=20, pady=8, sticky="nsew")

        # Management per cuvânt + Butonul de Train
        bottom_bar = ctk.CTkFrame(right_panel, fg_color="transparent")
        bottom_bar.grid(row=5, column=0, padx=20, pady=(5, 15), sticky="ew")
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
        self.btn_train.grid(row=0, column=0, columnspan=2, pady=(0, 8), sticky="ew")

        # Ștergere individuală a unei singure etichete (fără a pierde restul bazei de date)
        del_row = ctk.CTkFrame(bottom_bar, fg_color="transparent")
        del_row.grid(row=1, column=0, columnspan=2, sticky="ew")
        del_row.grid_columnconfigure(0, weight=1)

        self.delete_label_menu = ctk.CTkOptionMenu(
            del_row, values=["(Niciun cuvânt)"], height=30
        )
        self.delete_label_menu.grid(row=0, column=0, padx=(0, 6), sticky="ew")

        self.btn_delete_one = ctk.CTkButton(
            del_row,
            text="Șterge Cuvântul Selectat",
            fg_color="#7f1d1d",
            hover_color="#991b1b",
            height=30,
            width=155,
            command=self.delete_single_label,
        )
        self.btn_delete_one.grid(row=0, column=1)

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

        sorted_labels = sorted(counts.keys())
        if sorted_labels:
            self.delete_label_menu.configure(values=sorted_labels)
            self.delete_label_menu.set(sorted_labels[0])
        else:
            self.delete_label_menu.configure(values=["(Niciun cuvânt)"])
            self.delete_label_menu.set("(Niciun cuvânt)")

        self.log("=== DATASET SUMMARY (APPEND MODE) ===")
        self.log(f"Total secvențe în baza de date: {len(samples)}")
        self.log(f"Cuvinte / Litere învățate: {len(counts)}\n")
        for lbl in sorted_labels:
            self.log(f"  • {lbl}: {counts[lbl]} vectori")
        self.log("=====================================\n")

    # ==========================================
    # PROCESATOR UNITAR PENTRU VIDEO ȘI IMAGINI
    # ==========================================
    def _ingest_single_media_file(self, filepath, label):
        """
        Extrage coordonatele mâinii dintr-un fișier video (.mp4/.avi) sau dintr-o imagine (.jpg/.png)
        și îi dă APPEND în `self.dataset["samples"]` sub eticheta `label`.
        Returnează numărul de vectori adăugați.
        """
        ext = os.path.splitext(filepath)[1].lower()
        added = 0

        # CAZUL 1: Fișier Imagine (.jpg, .png) - util pentru dataset-uri Kaggle cu litere/semne
        if ext in IMAGE_EXTS:
            frame = cv2.imread(filepath)
            if frame is None:
                return 0
            _, sign_template = self.hand_tracker.process_frame(frame)
            feat = SignFeatureExtractor.extract_frame_features(sign_template)
            if feat is None:
                return 0

            # Simulăm o secvență statică stabilă din acea imagine + augmentări
            static_seq = [feat for _ in range(6)]
            for vec in SignFeatureExtractor.augment_sequence(static_seq, num_augmentations=6):
                self.dataset["samples"].append({
                    "label": label,
                    "features": [round(float(x), 5) for x in vec],
                })
                added += 1
            return added

        # CAZUL 2: Fișier Video (.mp4, .avi, .mov)
        if ext in VIDEO_EXTS:
            cap = cv2.VideoCapture(filepath)
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

            if len(frames_feats) < 2:
                return 0

            # Întregul clip + variații
            for vec in SignFeatureExtractor.augment_sequence(frames_feats, num_augmentations=8):
                self.dataset["samples"].append({
                    "label": label,
                    "features": [round(float(x), 5) for x in vec],
                })
                added += 1

            # Ferestre scurte glisante pentru recunoaștere rapidă
            win_size = min(8, len(frames_feats))
            step = max(2, win_size // 2)
            for start_i in range(0, max(1, len(frames_feats) - win_size + 1), step):
                sub_seq = frames_feats[start_i : start_i + win_size]
                if len(sub_seq) >= 2:
                    for vec in SignFeatureExtractor.augment_sequence(sub_seq, num_augmentations=3):
                        self.dataset["samples"].append({
                            "label": label,
                            "features": [round(float(x), 5) for x in vec],
                        })
                        added += 1

        return added

    # ==========================================
    # FLUXUL BULK FOLDER (CU SELECTOR VIZUAL)
    # ==========================================
    def open_bulk_dataset_wizard(self):
        folder_path = filedialog.askdirectory(
            title="Selectează Folderul cu Dataset-ul (poate conține subfoldere sau fișiere directe)"
        )
        if not folder_path:
            return

        self.log(f"[SCAN] Scanez folderul '{os.path.basename(folder_path)}'...")
        grouped_files = {}
        norm_root = os.path.normpath(folder_path)

        for current_root, _, files in os.walk(norm_root):
            for file in files:
                ext = os.path.splitext(file)[1].lower()
                if ext not in VIDEO_EXTS and ext not in IMAGE_EXTS:
                    continue

                full_path = os.path.join(current_root, file)
                # Dacă fișierul se află într-un subfolder (ex: dataset/SALUT/001.mp4),
                # luăm numele subfolderului ca etichetă. Dacă e direct în rădăcină, luăm numele fișierului.
                if os.path.normpath(current_root) != norm_root:
                    label = os.path.basename(current_root).strip().upper()
                else:
                    label = self._clean_label_from_filename(full_path)

                grouped_files.setdefault(label, []).append(full_path)

        if not grouped_files:
            self.log("[!] Nu s-au găsit fișiere video sau imagini în folderul selectat.")
            return

        # Deschidem fereastra prietenoasă unde alegi exact ce cuvinte vrei să învețe
        BulkDatasetModal(
            self,
            folder_path=folder_path,
            grouped_files=grouped_files,
            on_confirm_callback=self._process_bulk_selection,
        )

    def _process_bulk_selection(self, selected_map):
        total_files = sum(len(paths) for paths in selected_map.values())
        if total_files == 0:
            return

        self.log(
            f"[BULK LEARN] Încep procesarea a {len(selected_map)} cuvinte ({total_files} fișiere)..."
        )
        self.progress_bar.set(0)

        processed = 0
        added_vectors = 0
        learned_words = []

        for label, file_list in selected_map.items():
            word_vectors = 0
            for filepath in file_list:
                cnt = self._ingest_single_media_file(filepath, label)
                word_vectors += cnt
                added_vectors += cnt
                processed += 1
                self.progress_bar.set(processed / total_files)
                self.update_idletasks()

            if word_vectors > 0:
                learned_words.append(f"{label} (+{word_vectors})")
                self.log(f"  ✓ '{label}': adăugați {word_vectors} vectori în dataset.")
            else:
                self.log(f"  ⚠ '{label}': nu s-au detectat mâini clare în fișiere.")

        self._save_dataset()
        self.progress_bar.set(1.0)
        self._refresh_stats()
        self.log(
            f"[BULK COMPLET] S-au adăugat (Append) {added_vectors} vectori noi! "
            f"Apasă '⚡ TRAIN & EXPORT AI MODEL' pentru a actualiza creierul AI."
        )

    # ==========================================
    # FLUXUL PENTRU FIȘIERE SELECTATE MANUAL
    # ==========================================
    def import_selected_files(self):
        file_paths = filedialog.askopenfilenames(
            title="Selectează Fișiere Video sau Imagini pentru Învățare",
            filetypes=[
                ("Media Files", "*.mp4 *.avi *.mov *.mkv *.webm *.jpg *.jpeg *.png"),
                ("Video Files", "*.mp4 *.avi *.mov *.mkv *.webm"),
                ("Image Files", "*.jpg *.jpeg *.png *.bmp"),
            ],
        )
        if not file_paths:
            return

        # Dacă utilizatorul a scris manual un cuvânt în căsuța de jos, îl întrebăm
        # sau folosim automat numele fiecărui fișier
        manual_override = self.entry_label.get().strip().upper()

        total = len(file_paths)
        added_total = 0
        self.progress_bar.set(0)

        for idx, path in enumerate(file_paths, start=1):
            label = manual_override if manual_override else self._clean_label_from_filename(path)
            filename = os.path.basename(path)
            self.log(f"[{idx}/{total}] Procesez '{filename}' -> Cuvânt: '{label}'...")

            cnt = self._ingest_single_media_file(path, label)
            if cnt > 0:
                added_total += cnt
            else:
                self.log(f"  [!] În '{filename}' nu s-a detectat nicio mână.")

            self.progress_bar.set(idx / total)
            self.update_idletasks()

        self._save_dataset()
        self._refresh_stats()
        self.log(
            f"[APPEND COMPLET] S-au adăugat {added_total} vectori noi peste baza de date existentă!"
        )

    # ==========================================
    # FLUXUL LIVE CAMERA & POST-LABELING
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
            if len(self.recorded_Buffer) >= 2:
                self.btn_save_live.configure(state="normal")
                self.log(
                    f"[OK] S-au capturat {len(self.recorded_Buffer)} cadre. "
                    f"Scrie în căsuță ce cuvânt a fost și apasă 'Append Label'!"
                )
            else:
                self.log("[EROARE] Nu s-a detectat mâna în timpul înregistrării.")

    def save_recorded_gesture(self):
        label = self.entry_label.get().strip().upper()
        if not label:
            self.log("[!] Scrie numele cuvântului/literei în căsuță înainte de salvare!")
            return

        if len(self.recorded_Buffer) < 2:
            self.log("[!] Nu există o înregistrare validă în memorie.")
            return

        aug_vectors = SignFeatureExtractor.augment_sequence(self.recorded_Buffer, num_augmentations=8)
        for vec in aug_vectors:
            self.dataset["samples"].append({
                "label": label,
                "features": [round(float(x), 5) for x in vec],
            })

        if len(self.recorded_Buffer) >= 12:
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
        self.log(f"[APPEND] Semnul '{label}' a fost adăugat în baza de date!")

    # ==========================================
    # ANTRENARE & MANAGEMENT SELECTIV
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
                f"Adaugă cel puțin încă un cuvânt diferit!"
            )
            return

        self.log(f"[AI] Antrenez modelul rapid pe {len(unique_classes)} cuvinte ({len(samples)} vectori)...")

        model = make_pipeline(
            StandardScaler(),
            ExtraTreesClassifier(
                n_estimators=80,
                max_depth=None,
                min_samples_split=2,
                class_weight="balanced",
                random_state=42,
                n_jobs=1,  # Critic pentru latență 0 la rularea live pe Windows
            ),
        )
        model.fit(X, y)

        bundle = {
            "model": model,
            "classes": unique_classes.tolist(),
        }
        joblib.dump(bundle, self.model_path)
        self.log(f"[SUCCES] Modelul AI actualizat a fost salvat în:\n{self.model_path}")
        self.log(f"Total cuvinte active în aplicație ({len(unique_classes)}): {', '.join(unique_classes)}")

    def delete_single_label(self):
        target = self.delete_label_menu.get()
        if not target or target == "(Niciun cuvânt)":
            return

        before_count = len(self.dataset.get("samples", []))
        self.dataset["samples"] = [
            s for s in self.dataset.get("samples", []) if s["label"] != target
        ]
        after_count = len(self.dataset["samples"])
        removed = before_count - after_count

        self._save_dataset()
        self._refresh_stats()
        self.log(
            f"[ȘTERS] Cuvântul '{target}' ({removed} vectori) a fost eliminat. "
            f"Restul bazei de date a rămas intact!"
        )

    def on_closing(self):
        self.stop_camera()
        self.destroy()


if __name__ == "__main__":
    studio = DevTrainerStudio()
    studio.protocol("WM_DELETE_WINDOW", studio.on_closing)
    studio.mainloop()