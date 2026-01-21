# image_processor.py
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
        if len(image.shape) == 3 and image.shape[2] == 3: # Check if image is color
             gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif len(image.shape) == 2: # Check if image is already grayscale
             gray = image
        else:
             # Handle unexpected image format, maybe raise an error or return None
             # For now, let's try converting assuming it might be BGR or BGRA
             try:
                 gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
             except cv2.error:
                 # If conversion fails, return a black mask as a fallback
                 print("Warning: Could not convert image to grayscale for edge detection. Input shape:", image.shape)
                 return np.zeros(image.shape[:2], dtype=np.uint8)

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
        # Ensure image is 3-channel BGR before reshaping
        if len(image.shape) != 3 or image.shape[2] != 3:
            print("Warning: Color quantization requires a 3-channel BGR image.")
            # Optionally convert grayscale back to BGR or return original
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            else:
                return image # Return original if format is unexpected

        data = image.reshape((-1, 3))
        # Ensure data is float32 for KMeans
        data = np.float32(data)
        kmeans = KMeans(n_clusters=self.n_colors, random_state=42, n_init=10) # Explicitly set n_init
        kmeans.fit(data)
        # Ensure cluster centers are uint8 before assignment
        centers = np.uint8(kmeans.cluster_centers_)
        quantized = centers[kmeans.labels_].reshape(image.shape)
        return quantized # Already uint8

    def apply_bilateral_filter(self, image, d=7, sigma_color=200, sigma_space=200):
        """Apply bilateral filter for edge-preserving smoothing"""
        # Bilateral filter works on color (3-channel) or grayscale images
        return cv2.bilateralFilter(image, d=d, sigmaColor=sigma_color, sigmaSpace=sigma_space)

    def cartoonify(self, image, bilateral_filter_params=None):
        """Apply cartoonification effect to the image"""
        if bilateral_filter_params is None:
            bilateral_filter_params = {}

        # Create edge mask (works on grayscale representation)
        edges = self.create_edge_mask(image)

        # Apply color quantization (requires BGR)
        # If input is grayscale, cartoonify might not look right.
        # Consider converting grayscale to BGR first if needed, or handle differently.
        if len(image.shape) == 2:
            image_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            image_bgr = image

        color_quantized = self.apply_color_quantization(image_bgr)

        # Apply bilateral filter
        smoothed = self.apply_bilateral_filter(color_quantized, **bilateral_filter_params)

        # Combine edge mask with color image
        # Ensure mask is single channel
        if len(edges.shape) == 3:
             edges = cv2.cvtColor(edges, cv2.COLOR_BGR2GRAY) # Ensure mask is 1-channel

        cartoon = cv2.bitwise_and(smoothed, smoothed, mask=edges)

        return cartoon

    def adjust_brightness_contrast(self, image, brightness=0, contrast=0):
        """Adjust brightness and contrast of the image"""
        # Ensure image is uint8 before operations
        image = np.uint8(image)
        
        # --- Brightness Adjustment ---
        if brightness != 0:
            if brightness > 0:
                shadow = brightness
                highlight = 255
            else:
                shadow = 0
                highlight = 255 + brightness
            alpha_b = (highlight - shadow)/255
            gamma_b = shadow
            # Use addWeighted for controlled blending
            image = cv2.addWeighted(image, alpha_b, np.zeros_like(image), 0, gamma_b)
            # Ensure result stays within uint8 range
            image = np.clip(image, 0, 255).astype(np.uint8)


        # --- Contrast Adjustment ---
        if contrast != 0:
            # Original formula might be sensitive, alternative approach:
            # Adjust contrast by scaling pixel values around the midpoint (128)
            alpha_c = float(131 * (contrast + 127)) / (127 * (131 - contrast)) # Factor
            gamma_c = 127 * (1 - alpha_c) # Offset

            # Apply the transformation: new_pixel = alpha_c * old_pixel + gamma_c
            image = cv2.addWeighted(image, alpha_c, np.zeros_like(image), 0, gamma_c)
            # Ensure result stays within uint8 range
            image = np.clip(image, 0, 255).astype(np.uint8)

        return image

    def enhance_edges(self, image, kernel_size=3):
        """Enhance edges using unsharp masking"""
        # Ensure image is uint8
        image = np.uint8(image)
        # GaussianBlur works on BGR or grayscale
        blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
        # Use addWeighted for sharpening: image + (image - blurred) * weight
        # Equivalent to cv2.addWeighted(image, 1 + weight, blurred, -weight, 0)
        sharpened = cv2.addWeighted(image, 1.5, blurred, -0.5, 0)
        # Ensure result stays within uint8 range
        sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
        return sharpened

    # --- NEW METHODS ---

    def apply_grayscale(self, image):
        """Convert image to grayscale."""
        if len(image.shape) == 3 and image.shape[2] == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # Return as 3-channel gray for consistency if needed downstream
            return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        elif len(image.shape) == 2: # Already grayscale
             return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) # Convert to 3-channel
        else:
             print("Warning: Could not convert image to grayscale. Input shape:", image.shape)
             return image # Return original if format is unexpected

    def apply_negative(self, image):
        """Apply negative effect to the image."""
        # Works directly on BGR or grayscale uint8 images
        return cv2.bitwise_not(image)

    def shift_hue(self, image, shift_value=30):
        """Shift the hue of the image. shift_value typically 0-179 for OpenCV HSV."""
        if len(image.shape) != 3 or image.shape[2] != 3:
            print("Warning: Hue shift requires a 3-channel BGR image.")
            return image # Return original if not BGR

        try:
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            # Ensure hue values are treated correctly (uint8) and wrap around
            # Convert H channel to int16 to prevent overflow during addition
            h = hsv[:, :, 0].astype(np.int16)
            h = (h + shift_value) % 180 # OpenCV Hue range is 0-179
            # Convert back to uint8
            hsv[:, :, 0] = h.astype(np.uint8)

            shifted_image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            return shifted_image
        except cv2.error as e:
            print(f"Error during hue shift: {e}")
            return image # Return original on error

    def get_edges_only(self, image):
        """Return only the edge mask as a BGR image."""
        edges = self.create_edge_mask(image)
        # Convert single-channel mask to 3-channel BGR for consistent output
        return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
