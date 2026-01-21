# app.py
import os
import time
from flask import Flask, request, render_template, send_file, jsonify
import cv2
import numpy as np
# from PIL import Image # PIL not needed for these changes
import imageio
import io
from image_processor import ImageProcessor # Assuming image_processor.py is in the same directory
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Configure upload folder and settings
UPLOAD_FOLDER = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
# Added gif/bmp to allowed extensions based on HTML form accept attribute
app.config['ALLOWED_EXTENSIONS'] = {'.png', '.jpg', '.jpeg', '.gif', '.bmp'}

# Create upload folder if it doesn't exist and set permissions
if not os.path.exists(UPLOAD_FOLDER):
    try:
        os.makedirs(UPLOAD_FOLDER)
        os.chmod(UPLOAD_FOLDER, 0o755) # Set read/write/execute for owner, read/execute for group/others
    except OSError as e:
        app.logger.error(f"Error creating or setting permissions for upload folder '{UPLOAD_FOLDER}': {e}")
        # Depending on the severity, you might want to exit or handle this differently
        # For now, we'll log and continue, but uploads might fail.

def cleanup_old_files():
    """Remove files older than 1 hour from the upload folder"""
    current_time = time.time()
    try:
        for filename in os.listdir(UPLOAD_FOLDER):
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            if os.path.isfile(file_path):
                # Check file modification time
                if current_time - os.path.getmtime(file_path) > 3600: # 3600 seconds = 1 hour
                    try:
                        os.remove(file_path)
                        app.logger.info(f'Cleaned up old file: {filename}')
                    except Exception as e:
                        app.logger.error(f'Error cleaning up file {filename}: {str(e)}')
    except FileNotFoundError:
        app.logger.warning(f"Upload folder '{UPLOAD_FOLDER}' not found during cleanup.")
    except Exception as e:
        app.logger.error(f"Error during file cleanup process: {str(e)}")

# Initialize image processor
try:
    processor = ImageProcessor()
except NameError:
    # This might happen if ImageProcessor class definition has an error or isn't imported
    app.logger.error("Failed to initialize ImageProcessor. Check image_processor.py.")
    processor = None # Set to None to handle potential errors later
except Exception as e:
    app.logger.error(f"An unexpected error occurred during ImageProcessor initialization: {str(e)}")
    processor = None

def create_animation(image):
    """Creates a simple zoom animation centered on a black background."""
    frames = []
    # Use the already processed image (should be BGR format)
    base_image = image

    # Ensure base_image is BGR format before proceeding
    if len(base_image.shape) == 2: # If grayscale, convert to BGR
        base_image = cv2.cvtColor(base_image, cv2.COLOR_GRAY2BGR)
    elif len(base_image.shape) == 3 and base_image.shape[2] == 4: # If BGRA, convert to BGR
        base_image = cv2.cvtColor(base_image, cv2.COLOR_BGRA2BGR)
    elif not (len(base_image.shape) == 3 and base_image.shape[2] == 3):
        app.logger.error(f"Cannot create animation: Unexpected image shape {base_image.shape}")
        return None # Cannot animate this format

    # Create a zooming effect
    for scale in np.linspace(0.8, 1.2, 10): # 10 frames zooming from 80% to 120%
        # Calculate new dimensions while maintaining aspect ratio
        height, width = base_image.shape[:2]
        new_width = int(width * scale)
        new_height = int(height * scale)
        # Ensure dimensions are at least 1x1
        new_width = max(1, new_width)
        new_height = max(1, new_height)
        dim = (new_width, new_height)

        # Resize the image
        try:
            resized = cv2.resize(base_image, dim, interpolation=cv2.INTER_LINEAR)
        except cv2.error as e:
             app.logger.error(f"Error resizing frame for animation: {e}")
             return None # Stop animation generation if resize fails

        # Create background with same dimensions as original image (filled with black)
        background = np.zeros_like(base_image)
        orig_height, orig_width = background.shape[:2]

        # Calculate offsets to center the resized image
        y_offset = max(0, (orig_height - new_height) // 2)
        x_offset = max(0, (orig_width - new_width) // 2)

        # Calculate the slicing indices for the resized image and the background
        bg_y_start, bg_y_end = y_offset, y_offset + new_height
        bg_x_start, bg_x_end = x_offset, x_offset + new_width

        res_y_start, res_y_end = 0, new_height
        res_x_start, res_x_end = 0, new_width

        # Adjust if resized image is larger than background (cropping effect)
        if new_height > orig_height:
             res_y_start = (new_height - orig_height) // 2
             res_y_end = res_y_start + orig_height
             bg_y_start, bg_y_end = 0, orig_height

        if new_width > orig_width:
             res_x_start = (new_width - orig_width) // 2
             res_x_end = res_x_start + orig_width
             bg_x_start, bg_x_end = 0, orig_width

        # Ensure slicing indices are within bounds
        bg_y_end = min(bg_y_end, orig_height)
        bg_x_end = min(bg_x_end, orig_width)
        res_y_end = res_y_start + (bg_y_end - bg_y_start)
        res_x_end = res_x_start + (bg_x_end - bg_x_start)

        # Place the potentially cropped resized image onto the background
        try:
            background[bg_y_start:bg_y_end, bg_x_start:bg_x_end] = \
                resized[res_y_start:res_y_end, res_x_start:res_x_end]
            frames.append(background)
        except ValueError as e:
            app.logger.error(f"Error placing resized image onto background: {e}")
            app.logger.error(f"Background shape: {background.shape}, Slice: [{bg_y_start}:{bg_y_end}, {bg_x_start}:{bg_x_end}]")
            app.logger.error(f"Resized shape: {resized.shape}, Slice: [{res_y_start}:{res_y_end}, {res_x_start}:{res_x_end}]")
            return None # Stop animation generation

    if not frames:
        app.logger.error("No frames generated for animation.")
        return None

    # Ensure frames are in RGB format for imageio GIF saving
    try:
        rgb_frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames]
        return rgb_frames
    except cv2.error as e:
        app.logger.error(f"Error converting animation frames to RGB: {e}")
        return None


@app.route('/')
def home():
    """Serves the main HTML page."""
    return render_template('index.html')

@app.route('/cartoonify', methods=['POST'])
def cartoonify_route(): # Renamed function slightly to avoid conflict with method name
    """Handles image upload, processing, and returns the result."""
    if processor is None:
         return jsonify({'error': 'Image processor is not available. Please check server logs.'}), 500

    try:
        cleanup_old_files()

        if 'file' not in request.files:
            return jsonify({'error': 'No file part in the request'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        filename = secure_filename(file.filename)
        file_ext = os.path.splitext(filename)[1].lower() if filename else ''
        if not filename or file_ext not in app.config['ALLOWED_EXTENSIONS']:
            allowed_ext_str = ", ".join(app.config['ALLOWED_EXTENSIONS'])
            return jsonify({'error': f'Invalid file type or filename. Allowed types: {allowed_ext_str}'}), 400

        # Get parameters from request form
        try:
            # --- Primary Effect ---
            effect_type = request.form.get('effect_type', 'cartoon') # Default to cartoon

            # --- Optional Adjustments ---
            apply_animation = request.form.get('apply_animation', 'false').lower() == 'true'
            brightness = int(request.form.get('brightness', 0))
            contrast = int(request.form.get('contrast', 0))
            enhance_edges = request.form.get('enhance_edges', 'false').lower() == 'true'

            # Validate ranges (optional but good practice)
            brightness = max(-100, min(100, brightness))
            contrast = max(-100, min(100, contrast))

        except ValueError:
             return jsonify({'error': 'Invalid parameter value (brightness/contrast must be integers).'}), 400

        # Read and decode image
        img_stream = file.read()
        img_array = np.frombuffer(img_stream, np.uint8)
        image = cv2.imdecode(img_array, cv2.IMREAD_COLOR) # Read as BGR

        if image is None:
            return jsonify({'error': 'Could not decode image. Invalid image file format or content.'}), 400

        # --- Image Processing Pipeline ---
        processed_image = image.copy() # Start with a copy

        try:
            # 1. Apply the selected primary effect
            if effect_type == 'cartoon':
                processed_image = processor.cartoonify(processed_image)
            elif effect_type == 'grayscale':
                processed_image = processor.apply_grayscale(processed_image)
            elif effect_type == 'negative':
                processed_image = processor.apply_negative(processed_image)
            elif effect_type == 'hue_shift':
                # You could add a slider for shift_value later if needed
                processed_image = processor.shift_hue(processed_image, shift_value=30)
            elif effect_type == 'edges':
                processed_image = processor.get_edges_only(processed_image)
                # Disable enhance_edges and animation if primary effect is just edges?
                enhance_edges = False # Makes sense to disable sharpening on an edge mask
                # apply_animation = False # Animation might look weird on just edges
            elif effect_type == 'none':
                pass # Keep original for adjustments only
            else:
                # Fallback or error if unknown effect type
                app.logger.warning(f"Unknown effect_type received: {effect_type}")
                processed_image = image # Return original if effect is unknown

            # 2. Apply optional adjustments (if not disabled)
            if brightness != 0 or contrast != 0:
                processed_image = processor.adjust_brightness_contrast(processed_image, brightness, contrast)

            if enhance_edges: # Check flag which might have been modified above
                processed_image = processor.enhance_edges(processed_image)

        except AttributeError as e:
             app.logger.error(f"AttributeError during image processing (likely missing method in ImageProcessor): {str(e)}")
             return jsonify({'error': 'A required image processing function is missing or invalid.'}), 500
        except Exception as e:
            app.logger.error(f"Error during image processing: {str(e)}", exc_info=True)
            return jsonify({'error': f'Error during image processing: {str(e)}'}), 500


        # --- Output Generation ---
        try:
            output_filename_base = effect_type # Use effect type for base name
            if apply_animation:
                # Create animation frames (expects BGR, returns RGB)
                # Pass the *final* processed image to the animation function
                frames = create_animation(processed_image)
                if not frames:
                     # create_animation logs errors, return a generic message
                     return jsonify({'error': 'Failed to create animation frames.'}), 500

                # Save as GIF
                output = io.BytesIO()
                # Use duration=100ms (0.1s) per frame for 10fps animation
                imageio.mimsave(output, frames, format='GIF', duration=100, loop=0) # loop=0 means infinite loop
                output.seek(0)
                return send_file(
                    output,
                    mimetype='image/gif',
                    as_attachment=True,
                    download_name=f'{output_filename_base}_animation.gif'
                )
            else:
                # Convert final processed image (should be BGR or 3-channel Gray) to PNG bytes
                is_success, buffer = cv2.imencode('.png', processed_image)
                if not is_success:
                     return jsonify({'error': 'Failed to encode processed image to PNG.'}), 500

                output = io.BytesIO(buffer)
                return send_file(
                    output,
                    mimetype='image/png',
                    as_attachment=True,
                    download_name=f'{output_filename_base}_image.png'
                )
        except Exception as e:
            app.logger.error(f"Error saving or sending processed image/animation: {str(e)}", exc_info=True)
            return jsonify({'error': 'Error generating the final output file.'}), 500

    except Exception as e:
        app.logger.error(f"An unexpected error occurred in /cartoonify route: {str(e)}", exc_info=True)
        return jsonify({'error': 'An internal server error occurred. Please try again later.'}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000) # Keep debug=True for development
