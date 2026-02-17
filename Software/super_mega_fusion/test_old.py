from super_mega_fusion import hsv_mask, VisionLabelIntervals, Labels, ChannelIntervals, image_to_labels, flatten_labels
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import RangeSlider, Button, Slider


class HSVRangeGUI:
    """Interactive HSV mask GUI using double-handle RangeSliders.

    Usage:
        gui = HSVRangeGUI(image_path)
        gui.run()
    """

    def __init__(self, image_path, figsize=(10, 6)):
        self.image_path = Path(image_path)
        self.figsize = figsize

        image = cv2.imread(str(self.image_path))
        if image is None:
            raise FileNotFoundError(f"Could not read image: {self.image_path}")

        # prepare images
        self.image_bgr = image.copy()
        self.image_rgb = cv2.cvtColor(self.image_bgr, cv2.COLOR_BGR2RGB)
        self.image_hsv = cv2.cvtColor(self.image_bgr, cv2.COLOR_BGR2HSV)

        # initial ranges
        self.h_low0, self.s_low0, self.v_low0 = 0, 0, 0
        self.h_high0, self.s_high0, self.v_high0 = 179, 255, 255

        # build UI
        self._build_ui()

    def _build_ui(self):
        self.fig = plt.figure(figsize=self.figsize)
        self.ax_img = self.fig.add_subplot(1, 2, 1)
        self.ax_mask = self.fig.add_subplot(1, 2, 2)

        self.ax_img.imshow(self.image_rgb)
        self.ax_img.set_title('Image')
        self.ax_img.axis('off')

        mask = self._compute_mask(self.h_low0, self.h_high0,
                      self.s_low0, self.s_high0,
                      self.v_low0, self.v_high0)
        mask_rgb = np.dstack([mask, mask, mask])
        self.mask_im = self.ax_mask.imshow(mask_rgb, cmap='gray')
        self.ax_mask.set_title('HSV Mask')
        self.ax_mask.axis('off')

        # sliders
        slider_height = 0.04
        spacing = 0.02
        start_y = 0.12

        # H uses two independent sliders (low/high) so values may cross (wrap-around)
        ax_hhigh = self.fig.add_axes([0.15, start_y + 1.7 * (slider_height + spacing) - 0.01, 0.7, slider_height])
        ax_hlow = self.fig.add_axes([0.15, start_y + 2 * (slider_height + spacing), 0.7, slider_height])
        ax_srange = self.fig.add_axes([0.15, start_y + 1 * (slider_height + spacing), 0.7, slider_height])
        ax_vrange = self.fig.add_axes([0.15, start_y + 0 * (slider_height + spacing), 0.7, slider_height])

        self.s_h_low = Slider(ax_hlow, 'H low', 0, 179, valinit=self.h_low0, valstep=1)
        self.s_h_high = Slider(ax_hhigh, 'H high', 0, 179, valinit=self.h_high0, valstep=1)
        self.r_s = RangeSlider(ax_srange, 'S', 0, 255, valinit=(self.s_low0, self.s_high0), valstep=1)
        self.r_v = RangeSlider(ax_vrange, 'V', 0, 255, valinit=(self.v_low0, self.v_high0), valstep=1)

        # connect updates
        self.s_h_low.on_changed(self._update)
        self.s_h_high.on_changed(self._update)
        self.r_s.on_changed(self._update)
        self.r_v.on_changed(self._update)

        # reset button
        ax_reset = self.fig.add_axes([0.85, 0.02, 0.1, 0.04])
        btn = Button(ax_reset, 'Reset')
        btn.on_clicked(self._reset)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    def _update(self, _=None):
        hl = int(self.s_h_low.val)
        hh = int(self.s_h_high.val)
        sl, sh = map(int, self.r_s.val)
        vl, vh = map(int, self.r_v.val)

        m = self._compute_mask(hl, hh, sl, sh, vl, vh)
        m_rgb = np.dstack([m, m, m])
        self.mask_im.set_data(m_rgb)
        self.fig.canvas.draw_idle()

    def _compute_mask(self, hl, hh, sl, sh, vl, vh):
        """Compute mask handling hue wrap-around.

        If hl <= hh: select H in [hl, hh]. If hl > hh: select H >= hl or H <= hh.
        S and V are selected between their ranges normally.
        Returns an 8-bit single-channel mask (0 or 255).
        """
        hsv = self.image_hsv
        h = hsv[:, :, 0]
        s = hsv[:, :, 1]
        v = hsv[:, :, 2]

        if hl <= hh:
            mask_h = (h >= hl) & (h <= hh)
        else:
            # wrap-around: outside the interval between hl and hh
            mask_h = (h >= hl) | (h <= hh)

        mask_s = (s >= sl) & (s <= sh)
        mask_v = (v >= vl) & (v <= vh)

        mask = mask_h & mask_s & mask_v
        return (mask.astype('uint8') * 255)

    def _reset(self, event=None):
        self.s_h_low.set_val(self.h_low0)
        self.s_h_high.set_val(self.h_high0)
        self.r_s.set_val((self.s_low0, self.s_high0))
        self.r_v.set_val((self.v_low0, self.v_high0))

    def run(self):
        plt.show()


if __name__ == '__main__':
    image_path = Path(__file__).parent / 'test.png'
    image_bgr = cv2.imread(str(image_path))
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    
    if False: # Range finder
        gui = HSVRangeGUI(image_path)
        gui.run()
    
    if False: # Show results of found range
        # Params
        hl, hh = (33, 72)
        sl, sh = (157, 255)
        vl, vh = (47, 255)

        # Load image
        image_hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        # Normalize HSV to 0..1 floats for hsv_mask
        hsv_float = image_hsv.astype('float32')
        hsv_float[:, :, 0] = hsv_float[:, :, 0] / 179.0
        hsv_float[:, :, 1] = hsv_float[:, :, 1] / 255.0
        hsv_float[:, :, 2] = hsv_float[:, :, 2] / 255.0
        # Compute mask (boolean)
        mask_bool = hsv_mask(hsv_float, (hl / 179, hh / 179), (sl / 255, sh / 255), (vl / 255, vh / 255))
        # Convert boolean mask to uint8 0/255 for display and OpenCV operations
        mask = (mask_bool.astype('uint8') * 255)
        # Get masked RGB image using the mask
        masked_rgb = cv2.bitwise_and(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB),
                                    cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB),
                                    mask=mask)
        
        # Show results (original, mask, masked image)
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 3, 1)
        plt.imshow(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
        plt.title('Original Image')
        plt.axis('off')
        plt.subplot(1, 3, 2)
        plt.imshow(mask, cmap='gray')
        plt.title('HSV Mask')
        plt.axis('off')
        plt.subplot(1, 3, 3)
        plt.imshow(masked_rgb)
        plt.title('Masked Image')
        plt.axis('off')
        plt.tight_layout()
        plt.show()


    # Apply to super_mega_fusion
    # Define intervals
    vision_label_intervals: VisionLabelIntervals = {
        Labels.ENEMY:      ChannelIntervals(h=(162/179, 24/179), s=(200/255, 1.0), v=(15/255, 1.0)),
        Labels.WALL_GREEN: ChannelIntervals(h=(33/179, 72/179),  s=(157/255, 1.0), v=(47/255, 1.0)),
        Labels.WALL_RED:   ChannelIntervals(h=(162/179, 24/179), s=(200/255, 1.0), v=(15/255, 1.0)),
        Labels.BACKGROUND: ChannelIntervals(h=(0.0, 1.0),        s=(0.0, 1.0),     v=(0.0, 1.0)),
    }

    # Get labels
    labels_image = image_to_labels(image_rgb, band_y_min=0, band_y_max=image_bgr.shape[0], label_intervals=vision_label_intervals)
    
    # Show labels as color-coded image
    label_colors = {
        Labels.ENEMY.value: (255, 0, 0),       # Red
        Labels.WALL_GREEN.value: (0, 255, 0),  # Green
        Labels.WALL_RED.value: (255, 0, 0),    # Red
        Labels.BACKGROUND.value: (0, 0, 0),    # Black
    }
    label_image = np.zeros((labels_image.shape[0], labels_image.shape[1], 3), dtype=np.uint8)
    for label_value, color in label_colors.items():
        label_image[labels_image == label_value] = color
    plt.figure(figsize=(6, 6))
    plt.imshow(label_image)
    plt.title('Label Image')
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    # Save image
    cv2.imwrite(str(Path(__file__).parent / 'label_image_full.png'), cv2.cvtColor(label_image, cv2.COLOR_RGB2BGR))


    # Get 1D labels (flattened)
    labels_1d = flatten_labels(labels_image)

    # Show labels as color-coded image
    label_colors = {
        Labels.ENEMY.value: (255, 0, 0),       # Red
        Labels.WALL_GREEN.value: (0, 255, 0),  # Green
        Labels.WALL_RED.value: (255, 0, 0),    # Red
        Labels.BACKGROUND.value: (0, 0, 0),    # Black
    }
    label_image = np.zeros((labels_1d.shape[0], 1, 3), dtype=np.uint8)
    for label_value, color in label_colors.items():
        label_image[labels_1d == label_value] = color
    # Transpose to (width, 1) for easier indexing
    label_image = label_image.transpose(1, 0, 2)  # Now shape is (1, width, 3)
    plt.figure(figsize=(4, 6))
    plt.imshow(label_image)
    plt.title('Label Image (1D)')
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    # Save image
    cv2.imwrite(str(Path(__file__).parent / 'label_image.png'), cv2.cvtColor(label_image, cv2.COLOR_RGB2BGR))