import os
import io
import requests
from flask import Flask, request, send_file, jsonify
from PIL import Image
import numpy as np
import math
import tempfile
from dotenv import load_dotenv
from typing import Tuple  
from flask_cors import CORS

load_dotenv()

app = Flask(__name__)

CORS(app)
# Unsplash API configuration
UNSPLASH_ACCESS_KEY = os.getenv("UNSPLASH_ACCESS_KEY")
UNSPLASH_API_URL = "https://api.unsplash.com/photos/random"

if not UNSPLASH_ACCESS_KEY:
    raise ValueError("UNSPLASH_ACCESS_KEY not found in environment variables")

TEMP_DIR = "temp"
os.makedirs(TEMP_DIR, exist_ok=True)

def get_random_image_from_unsplash():
    """Fetch a random image from Unsplash API."""
    headers = {"Authorization": f"Client-ID {UNSPLASH_ACCESS_KEY}"}
    params = {"query": "abstract", "orientation": "squarish"}

    try:
        response = requests.get(UNSPLASH_API_URL, headers=headers, params=params)
        response.raise_for_status()
        image_url = response.json()["urls"]["regular"]
        image_response = requests.get(image_url)
        image_response.raise_for_status()
        return Image.open(io.BytesIO(image_response.content))
    except Exception as e:
        raise Exception(f"Failed to fetch image from Unsplash: {str(e)}")

def save_upload_file_temporarily(upload_file):
    """Save uploaded file to temporary directory."""
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    upload_file.save(temp_file.name)
    return temp_file.name

def file_to_binary(file_path: str) -> bytes:
    """Convert file to binary data."""
    with open(file_path, 'rb') as f:
        return f.read()

def calculate_image_size(binary_data: bytes) -> tuple:
    """Calculate required image size for binary data."""
    total_bits = len(binary_data) * 8
    min_pixels = math.ceil(total_bits / 3)
    side = math.ceil(math.sqrt(min_pixels))
    return (side, side)

def binary_to_image(binary_data: bytes, template_image: Image.Image, original_extension: str) -> Image.Image:
    """Convert binary data to image."""
    width, height = calculate_image_size(binary_data)
    template_image = template_image.resize((width, height), Image.LANCZOS)
    pixels = np.array(template_image).flatten()

    bit_array = np.unpackbits(np.frombuffer(binary_data, dtype=np.uint8))

    if len(bit_array) > len(pixels):
        raise ValueError("File too large for image encoding")

    # Embed data in LSB
    pixels[:len(bit_array)] = (pixels[:len(bit_array)] & 0xFE) | bit_array

    ext_bytes = original_extension.encode('utf-8')
    ext_length = len(ext_bytes)
    pixels[-8:] = ext_length
    pixels[-8-ext_length:-8] = np.frombuffer(ext_bytes, dtype=np.uint8)

    return Image.fromarray(pixels.reshape(height, width, 3).astype(np.uint8))

def image_to_binary(image: Image.Image) -> Tuple[bytes, str]:
    """Extract binary data and extension from image."""
    pixels = np.array(image).flatten()

    ext_length = int(pixels[-8])
    ext_bytes = pixels[-8-ext_length:-8].astype(np.uint8).tobytes()
    extension = ext_bytes.decode('utf-8')

    bit_array = pixels[:-8-ext_length] & 1
    byte_data = np.packbits(bit_array).tobytes()

    return byte_data, extension

@app.route("/encode", methods=["POST"])
def encode_file():
    """Encode a file into an image."""
    file = request.files.get('file')
    template_image = request.files.get('template_image')

    if not file:
        return jsonify({"error": "File is required"}), 400

    try:
        file_path = save_upload_file_temporarily(file)
        binary_data = file_to_binary(file_path)
        original_extension = os.path.splitext(file.filename)[1][1:]

        if template_image:
            template_img = Image.open(io.BytesIO(template_image.read())).convert('RGB')
        else:
            template_img = get_random_image_from_unsplash()

        encoded_image = binary_to_image(binary_data, template_img, original_extension)

        output_buffer = io.BytesIO()
        encoded_image.save(output_buffer, format='PNG')
        output_buffer.seek(0)

        return send_file(
            output_buffer,
            mimetype='image/png',
            as_attachment=True,
            download_name=f"encoded_{original_extension}.png"
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/decode", methods=["POST"])
def decode_image():
    """Decode an image back to the original file."""
    image = request.files.get('image')

    if not image:
        return jsonify({"error": "Image is required"}), 400

    try:
        img = Image.open(io.BytesIO(image.read()))

        binary_data, extension = image_to_binary(img)

        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=f".{extension}")
        temp_file.write(binary_data)
        temp_file.close()

        return send_file(
            temp_file.name,
            as_attachment=True,
            download_name=f"decoded_file.{extension}"
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
