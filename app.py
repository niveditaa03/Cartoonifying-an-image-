import os
import time
from flask import Flask, request, render_template, send_file, jsonify
import cv2
import numpy as np
from PIL import Image
import imageio
import io
from image_processor import ImageProcessor
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Configure upload folder and settings
UPLOAD_FOLDER = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['ALLOWED_EXTENSIONS'] = {'.png', '.jpg', '.jpeg', '.gif', '.bmp'}

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
    os.chmod(UPLOAD_FOLDER, 0o755)

def cleanup_old_files():
    """Remove files older than 1 hour from the upload folder"""
    current_time = time.time()
    for filename in os.listdir(UPLOAD_FOLDER):
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        if os.path.isfile(file_path):
            if current_time - os.path.getmtime(file_path) > 3600:
                try:
                    os.remove(file_path)
                except Exception as e:
                    app.logger.error(f'Error cleaning up file {filename}: {str(e)}')

# Initialize image processor
processor = ImageProcessor()

def create_animation(image):
    frames = []
    # Create a zooming effect
    for scale in np.linspace(0.8, 1.2, 10):
        # Calculate new dimensions while maintaining aspect ratio
        width = int(image.shape[1] * scale)
        height = int(image.shape[0] * scale)
        dim = (width, height)
        
        # Resize the image
        resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
        
        # Create background with same dimensions as original image
        background = np.zeros_like(image)
        
        # Calculate offsets to center the resized image
        y_offset = max(0, (image.shape[0] - resized.shape[0]) // 2)
        x_offset = max(0, (image.shape[1] - resized.shape[1]) // 2)
        
        # Ensure we don't exceed background dimensions
        y_end = min(y_offset + resized.shape[0], background.shape[0])
        x_end = min(x_offset + resized.shape[1], background.shape[1])
        resized_y = min(resized.shape[0], background.shape[0])
        resized_x = min(resized.shape[1], background.shape[1])
        
        # Place resized image onto background
        background[y_offset:y_end, x_offset:x_end] = resized[:resized_y, :resized_x]
        frames.append(background)
    
    return frames

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/cartoonify', methods=['POST'])
def cartoonify():
    try:
        cleanup_old_files()
        
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
            
        # Validate file type
        file_ext = os.path.splitext(file.filename)[1].lower()
        if file_ext not in app.config['ALLOWED_EXTENSIONS']:
            return jsonify({'error': 'Invalid file type. Only image files are allowed'}), 400
            
        # Secure the filename
        filename = secure_filename(file.filename)
        
        # Get parameters from request
        apply_cartoon = request.form.get('apply_cartoon', 'false') == 'true'
        apply_animation = request.form.get('apply_animation', 'false') == 'true'
        brightness = int(request.form.get('brightness', 0))
        contrast = int(request.form.get('contrast', 0))
        enhance_edges = request.form.get('enhance_edges', 'false') == 'true'
        
        # Read and process image
        img_stream = file.read()
        img_array = np.frombuffer(img_stream, np.uint8)
        image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({'error': 'Invalid image file'}), 400
        
        processed_image = image.copy()
        
        # Apply cartoonification if requested
        if apply_cartoon:
            try:
                processed_image = processor.cartoonify(processed_image)
                
                # Apply additional effects if requested
                if brightness != 0 or contrast != 0:
                    processed_image = processor.adjust_brightness_contrast(processed_image, brightness, contrast)
                
                if enhance_edges:
                    processed_image = processor.enhance_edges(processed_image)
            except Exception as e:
                return jsonify({'error': f'Error during image processing: {str(e)}'}), 500
        
        try:
            if apply_animation:
                # Create animation frames
                frames = create_animation(processed_image)
                # Save as GIF
                output = io.BytesIO()
                imageio.mimsave(output, frames, format='GIF', duration=0.1)
                output.seek(0)
                return send_file(output, mimetype='image/gif', as_attachment=True, download_name='cartoon.gif')
            else:
                # Convert to PNG
                _, buffer = cv2.imencode('.png', processed_image)
                output = io.BytesIO(buffer)
                return send_file(output, mimetype='image/png', as_attachment=True, download_name='cartoon.png')
        except Exception as e:
            return jsonify({'error': f'Error saving processed image: {str(e)}'}), 500
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)