import numpy as np


class SignFeatureExtractor:
    """
    Transformă coordonatele brute din `sign_input_template` într-un vector
    invariant la poziție, distanță și rotație:
    - 88 trăsături anatomice per cadru (coordonate 3D + distanțe între vârfuri + unghiuri 3D)
    - 320 trăsături spatio-temporale per secvență (Start, Mid, End, Velocity/Traiectorie, Std)
    """

    FINGERTIPS = [4, 8, 12, 16, 20]
    ANGLE_TRIPLETS = [
        (0, 1, 2), (1, 2, 3), (2, 3, 4),       # Degetul mare (Thumb)
        (0, 5, 6), (5, 6, 8),                  # Arătător (Index)
        (0, 9, 10), (9, 10, 12),               # Mijlociu (Middle)
        (0, 13, 14), (13, 14, 16),             # Inelar (Ring)
        (0, 17, 18),                           # Degetul mic (Pinky)
    ]

    @staticmethod
    def _angle_3d(a, b, c):
        """Calculează cosinusul unghiului ABC în spațiul 3D (invariant la orientarea mâinii)."""
        ba = a - b
        bc = c - b
        norm_ba = np.linalg.norm(ba)
        norm_bc = np.linalg.norm(bc)
        if norm_ba < 1e-6 or norm_bc < 1e-6:
            return 0.0
        cos_val = np.dot(ba, bc) / (norm_ba * norm_bc)
        return float(np.clip(cos_val, -1.0, 1.0))

    @classmethod
    def extract_frame_features(cls, sign_input_template):
        """
        Extrage vectorul de 88 de trăsături dintr-un singur cadru.
        Returnează None dacă nu este detectată nicio mână.
        """
        if not sign_input_template or sign_input_template.get("hands_detected", 0) == 0:
            return None

        hand = sign_input_template["hands"][0]
        lms = hand["landmarks"]

        coords = np.array([[lm["x"], lm["y"], lm["z"]] for lm in lms], dtype=np.float32)

        # Dacă e mâna stângă, oglindim axa X relativă pentru ca un semn făcut cu
        # stânga sau cu dreapta să aibă aceeași amprentă anatomică
        is_left = hand.get("type", "Right") == "Left"

        # 1. Translație în originea încheieturii (Punctul 0)
        wrist = coords[0].copy()
        rel_coords = coords - wrist
        if is_left:
            rel_coords[:, 0] = -rel_coords[:, 0]

        # 2. Normalizare la dimensiunea palmei (distanța 3D Încheietură -> Baza Mijlociului)
        palm_size = np.linalg.norm(rel_coords[9])
        if palm_size < 1e-4:
            palm_size = 1.0
        norm_coords = rel_coords / palm_size  # 21 x 3 = 63 trăsături

        # 3. Toate cele 15 distanțe pereche între încheietură și cele 5 vârfuri ale degetelor
        key_points = [0] + cls.FINGERTIPS
        pairwise_dists = []
        for i in range(len(key_points)):
            for j in range(i + 1, len(key_points)):
                dist = np.linalg.norm(norm_coords[key_points[i]] - norm_coords[key_points[j]])
                pairwise_dists.append(dist)  # 15 trăsături

        # 4. Cele 10 unghiuri articulare 3D ale oaselor
        joint_angles = [
            cls._angle_3d(norm_coords[a], norm_coords[b], norm_coords[c])
            for (a, b, c) in cls.ANGLE_TRIPLETS
        ]  # 10 trăsături

        # Total: 63 + 15 + 10 = 88 trăsături pe cadru
        feature_vec = np.concatenate([
            norm_coords.flatten(),
            np.array(pairwise_dists, dtype=np.float32),
            np.array(joint_angles, dtype=np.float32),
        ])
        return feature_vec

    @classmethod
    def extract_sequence_features(cls, frame_features_list):
        """
        Transformă o listă de cadre (de la un video sau o înregistrare live)
        într-un vector spatio-temporal de dimensiune fixă (320 trăsături)
        care surprinde atât forma mâinii, cât și traiectoria mișcării.
        """
        if not frame_features_list:
            return None

        seq = np.array(frame_features_list, dtype=np.float32)
        n = len(seq)

        # Împărțim secvența în 3 segmente temporale (Începutul, Mijlocul și Finalul gestului)
        if n == 1:
            start_pose = mid_pose = end_pose = seq[0]
            std_pose = np.zeros(63, dtype=np.float32)
        elif n == 2:
            start_pose = seq[0]
            mid_pose = np.mean(seq, axis=0)
            end_pose = seq[1]
            std_pose = np.std(seq[:, :63], axis=0)
        else:
            t1 = max(1, n // 3)
            t2 = max(t1 + 1, (2 * n) // 3)
            start_pose = np.mean(seq[:t1], axis=0)      # 88 trăsături
            mid_pose = np.mean(seq[t1:t2], axis=0)      # 88 trăsături
            end_pose = np.mean(seq[t2:], axis=0)        # 88 trăsături
            std_pose = np.std(seq[:, :63], axis=0)      # 63 trăsături (intensitatea mișcării)

        # Vectorul de traiectorie/deplasare între începutul și sfârșitul semnului (38 trăsături)
        # Esențial pentru diferențierea semnelor în mișcare (ex. J, Z, Salut, Unt vs Libelulă)
        delta_trajectory = end_pose[:63] - start_pose[:63]

        # Total: 88*3 + 63 + 63 = 390 trăsături spatio-temporale
        return np.concatenate([
            start_pose,
            mid_pose,
            end_pose,
            delta_trajectory,
            std_pose,
        ])

    @classmethod
    def augment_sequence(cls, frame_features_list, num_augmentations=8):
        """
        Generează variații sintetice (rotații 3D în planul camerei + scalare + jitter)
        pentru ca AI-ul să învețe robust chiar și dintr-un singur videoclip sau o singură înregistrare.
        """
        augmented_vectors = []
        base_vec = cls.extract_sequence_features(frame_features_list)
        if base_vec is None:
            return []

        augmented_vectors.append(base_vec)
        seq = np.array(frame_features_list, dtype=np.float32)

        for _ in range(num_augmentations):
            aug_seq = seq.copy()
            # Unghi mic de rotație între -12 și +12 grade
            theta = np.random.uniform(-0.21, 0.21)
            cos_t, sin_t = np.cos(theta), np.sin(theta)

            for f_idx in range(len(aug_seq)):
                coords = aug_seq[f_idx, :63].reshape(21, 3)
                x_new = coords[:, 0] * cos_t - coords[:, 1] * sin_t
                y_new = coords[:, 0] * sin_t + coords[:, 1] * cos_t
                coords[:, 0] = x_new
                coords[:, 1] = y_new

                # Adăugăm micro-zgomot gaussian pentru toleranță la tremuratul mâinii
                coords += np.random.normal(0, 0.008, coords.shape)
                aug_seq[f_idx, :63] = coords.flatten()

            aug_vec = cls.extract_sequence_features(list(aug_seq))
            if aug_vec is not None:
                augmented_vectors.append(aug_vec)

        return augmented_vectors