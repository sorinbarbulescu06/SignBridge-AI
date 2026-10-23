import math
import customtkinter as ctk
from PIL import Image


def create_empty_image():
    """Returnează o imagine transparentă 1x1 pentru ștergerea garantată a frame-urilor din CTkLabel."""
    return ctk.CTkImage(
        light_image=Image.new("RGBA", (1, 1), (0, 0, 0, 0)),
        dark_image=Image.new("RGBA", (1, 1), (0, 0, 0, 0)),
        size=(1, 1),
    )


def fit_image_to_container(img_pil, container_widget, padding=16):
    """Scalează imaginea PIL în container păstrând aspect ratio-ul original (fără stretch)."""
    cont_w = container_widget.winfo_width() - padding
    cont_h = container_widget.winfo_height() - padding

    if cont_w <= 50 or cont_h <= 50:
        cont_w, cont_h = 400, 300

    orig_w, orig_h = img_pil.size
    if orig_w == 0 or orig_h == 0:
        return ctk.CTkImage(light_image=img_pil, dark_image=img_pil, size=(cont_w, cont_h))

    scale = min(cont_w / orig_w, cont_h / orig_h)
    new_w = max(1, int(orig_w * scale))
    new_h = max(1, int(orig_h * scale))

    return ctk.CTkImage(light_image=img_pil, dark_image=img_pil, size=(new_w, new_h))


def compute_waveform_Render_data(width, height, volume, phase):
    """Calculează punctele, culoarea, grosimea liniei și noua fază pentru unda audio."""
    if width < 50 or height < 50:
        width, height = 400, 300

    mid_y = height / 2
    max_amp = (height / 2) - 15
    amplitude = 3 + (volume * max_amp)

    new_phase = phase + 0.15 + (volume * 0.35)

    if volume > 0.75:
        color = "#ff3b30"
        line_width = 4
    elif volume > 0.08:
        color = "#30d158"
        line_width = 3
    else:
        color = "#3B8ED0"
        line_width = 2

    points = []
    for x in range(0, width, 4):
        norm_x = x / width
        envelope = math.sin(norm_x * math.pi)
        y_offset = (
            math.sin((x * 0.03) + new_phase) * 0.7
            + math.sin((x * 0.07) - new_phase * 1.5) * 0.3
        )
        y = mid_y + (y_offset * amplitude * envelope)
        points.extend([x, y])

    return points, color, line_width, new_phase
