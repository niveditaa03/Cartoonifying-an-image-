import cv2
import numpy as np
from sklearn.cluster import KMeans

class ImageProcessor:
    def __init__(self, line_size=7, blur_value=7, n_colors=7):
        self.line_size = line_size
        self.blur_value = blur_value
        self.n_colors = n_colors

    def create_edge_mask(self, image):
        """Create edge mask using adaptive thresholding"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_blur = cv2.medianBlur(gray, self.blur_value)
        edges = cv2.adaptiveThreshold(
            gray_blur, 255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY,
            self.line_size,
            self.blur_value
        )
        return edges

    def apply_color_quantization(self, image):
        """Apply color quantization using KMeans clustering"""
        data = image.reshape((-1, 3))
        kmeans = KMeans(n_clusters=self.n_colors, random_state=42)
        kmeans.fit(data)
        quantized = kmeans.cluster_centers_[kmeans.labels_].reshape(image.shape)
        return quantized.astype(np.uint8)

    def apply_bilateral_filter(self, image, d=7, sigma_color=200, sigma_space=200):
        """Apply bilateral filter for edge-preserving smoothing"""
        return cv2.bilateralFilter(image, d=d, sigmaColor=sigma_color, sigmaSpace=sigma_space)

    def cartoonify(self, image, bilateral_filter_params=None):
        """Apply cartoonification effect to the image"""
        if bilateral_filter_params is None:
            bilateral_filter_params = {}

        # Create edge mask
        edges = self.create_edge_mask(image)

        # Apply color quantization
        color_quantized = self.apply_color_quantization(image)

        # Apply bilateral filter
        smoothed = self.apply_bilateral_filter(color_quantized, **bilateral_filter_params)

        # Combine edge mask with color image
        cartoon = cv2.bitwise_and(smoothed, smoothed, mask=edges)

        return cartoon

    def adjust_brightness_contrast(self, image, brightness=0, contrast=0):
        """Adjust brightness and contrast of the image"""
        if brightness != 0:
            if brightness > 0:
                shadow = brightness
                highlight = 255
            else:
                shadow = 0
                highlight = 255 + brightness
            alpha_b = (highlight - shadow)/255
            gamma_b = shadow
            image = cv2.addWeighted(image, alpha_b, image, 0, gamma_b)

        if contrast != 0:
            f = 131*(contrast + 127)/(127*(131-contrast))
            alpha_c = f
            gamma_c = 127*(1-f)
            image = cv2.addWeighted(image, alpha_c, image, 0, gamma_c)

        return image

    def enhance_edges(self, image, kernel_size=3):
        """Enhance edges using unsharp masking"""
        blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
        return cv2.addWeighted(image, 1.5, blurred, -0.5, 0)