import os
import os.path as path
import numpy as np
import cv2
from PIL import ImageFont
from hashlib import md5

# Color schemes (RGB) mapping to (Box/Accent Color, Text Contrast Color)
COLOR_NAME_TO_RGB = {
    "navy": ((0, 38, 63), (119, 193, 250)),
    "blue": ((0, 120, 210), (173, 220, 252)),
    "aqua": ((115, 221, 252), (0, 76, 100)),
    "teal": ((15, 205, 202), (0, 0, 0)),
    "olive": ((52, 153, 114), (25, 58, 45)),
    "green": ((0, 204, 84), (15, 64, 31)),
    "lime": ((1, 255, 127), (0, 102, 53)),
    "yellow": ((255, 216, 70), (103, 87, 28)),
    "orange": ((255, 125, 57), (104, 48, 19)),
    "red": ((255, 47, 65), (131, 0, 17)),
    "maroon": ((135, 13, 75), (239, 117, 173)),
    "fuchsia": ((246, 0, 184), (103, 0, 78)),
    "purple": ((179, 17, 193), (241, 167, 244)),
    "black": ((24, 24, 24), (220, 220, 220)),
    "gray": ((168, 168, 168), (0, 0, 0)),
    "silver": ((220, 220, 220), (0, 0, 0)),
}


from hashlib import md5
import os
import os.path as path
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

COLOR_NAME_TO_RGB = {
    "navy": ((0, 38, 63), (119, 193, 250)),
    "blue": ((0, 120, 210), (173, 220, 252)),
    "aqua": ((115, 221, 252), (0, 76, 100)),
    "teal": ((15, 205, 202), (0, 0, 0)),
    "olive": ((52, 153, 114), (25, 58, 45)),
    "green": ((0, 204, 84), (15, 64, 31)),
    "lime": ((1, 255, 127), (0, 102, 53)),
    "yellow": ((255, 216, 70), (103, 87, 28)),
    "orange": ((255, 125, 57), (104, 48, 19)),
    "red": ((255, 47, 65), (131, 0, 17)),
    "maroon": ((135, 13, 75), (239, 117, 173)),
    "fuchsia": ((246, 0, 184), (103, 0, 78)),
    "purple": ((179, 17, 193), (241, 167, 244)),
    "black": ((24, 24, 24), (220, 220, 220)),
    "gray": ((168, 168, 168), (0, 0, 0)),
    "silver": ((220, 220, 220), (0, 0, 0)),
}


from hashlib import md5
import cv2
import numpy as np

COLOR_NAME_TO_RGB = {
    "navy": ((0, 38, 63), (119, 193, 250)),
    "blue": ((0, 120, 210), (173, 220, 252)),
    "aqua": ((115, 221, 252), (0, 76, 100)),
    "teal": ((15, 205, 202), (0, 0, 0)),
    "olive": ((52, 153, 114), (25, 58, 45)),
    "green": ((0, 204, 84), (15, 64, 31)),
    "lime": ((1, 255, 127), (0, 102, 53)),
    "yellow": ((255, 216, 70), (103, 87, 28)),
    "orange": ((255, 125, 57), (104, 48, 19)),
    "red": ((255, 47, 65), (131, 0, 17)),
    "maroon": ((135, 13, 75), (239, 117, 173)),
    "fuchsia": ((246, 0, 184), (103, 0, 78)),
    "purple": ((179, 17, 193), (241, 167, 244)),
    "black": ((24, 24, 24), (220, 220, 220)),
    "gray": ((168, 168, 168), (0, 0, 0)),
    "silver": ((220, 220, 220), (0, 0, 0)),
}


class BoundingBox:

    def __init__(
        self,
        box_thickness=3,
        font_scale=1.2,
        font_weight=3,
        fill_alpha=0.3,
        default_color="green",
    ):
        """
        Args:
            box_thickness (int): Thickness of the box border.
            font_scale (float): Size of text label (e.g. 1.0, 1.5, 2.5). Increase this to make text bigger!
            font_weight (int): Boldness of text font (e.g., 2, 3, 4).
            fill_alpha (float): Transparency for box interior overlay (0.0 to 1.0).
            default_color (str): Default color name.
        """
        self.box_thickness = box_thickness
        self.font_scale = font_scale
        self.font_weight = max(1, font_weight)
        self.fill_alpha = fill_alpha
        self.default_color = default_color
        self.color_names = list(COLOR_NAME_TO_RGB.keys())

    @staticmethod
    def _rgb_to_bgr(color):
        return tuple(reversed(color))

    def add(
        self,
        image,
        left,
        top,
        right,
        bottom,
        label=None,
        color=None,
        fill_alpha=None,
        box_thickness=None,
        font_scale=None,
    ):
        if not isinstance(image, np.ndarray):
            raise TypeError("'image' parameter must be a numpy.ndarray")

        img_h, img_w = image.shape[:2]
        left, top, right, bottom = (
            int(max(0, left)),
            int(max(0, top)),
            int(min(img_w, right)),
            int(min(img_h, bottom)),
        )

        alpha = fill_alpha if fill_alpha is not None else self.fill_alpha
        thickness = (
            box_thickness if box_thickness is not None else self.box_thickness
        )
        scale = font_scale if font_scale is not None else self.font_scale

        # Dynamic Color Selection
        if label and not color:
            hex_digest = md5(label.encode()).hexdigest()
            color_index = int(hex_digest, 16) % len(self.color_names)
            color = self.color_names[color_index]

        if not color or color not in COLOR_NAME_TO_RGB:
            color = self.default_color

        bgr_box_color = self._rgb_to_bgr(COLOR_NAME_TO_RGB[color][0])
        bgr_text_color = self._rgb_to_bgr(COLOR_NAME_TO_RGB[color][1])

        # 1. Semi-transparent Interior Fill
        if alpha > 0.0 and (right > left) and (bottom > top):
            sub_img = image[top:bottom, left:right]
            colored_rect = np.full_like(sub_img, bgr_box_color, dtype=np.uint8)
            blended = cv2.addWeighted(
                sub_img, 1.0 - alpha, colored_rect, alpha, 0
            )
            image[top:bottom, left:right] = blended

        # 2. Outer Bounding Box
        cv2.rectangle(
            image, (left, top), (right, bottom), bgr_box_color, thickness
        )

        # 3. Text Badge Rendering (Pure OpenCV)
        if label:
            font = cv2.FONT_HERSHEY_SIMPLEX
            (text_w, text_h), baseline = cv2.getTextSize(
                label, font, scale, self.font_weight
            )

            padding = int(8 * scale)
            badge_w = text_w + (padding * 2)
            badge_h = text_h + baseline + (padding * 2)

            # Position badge: default above top border, move inside if off screen
            badge_top = top - badge_h
            if badge_top < 0:
                badge_top = top

            badge_left = left
            badge_bottom = badge_top + badge_h
            badge_right = badge_left + badge_w

            # Draw Solid Background Badge for Text
            cv2.rectangle(
                image,
                (badge_left, badge_top),
                (badge_right, badge_bottom),
                bgr_box_color,
                -1,
            )

            # Draw Bold Text inside Badge
            text_x = badge_left + padding
            text_y = badge_bottom - padding - baseline
            cv2.putText(
                image,
                label,
                (text_x, text_y),
                font,
                scale,
                bgr_text_color,
                self.font_weight,
                lineType=cv2.LINE_AA,
            )

        return image