import os
import io
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from flask import Flask, request, jsonify, render_template, send_from_directory
import numpy as np
import base64

app = Flask(__name__)

# ─── Model Configuration ───────────────────────────────────────────────
CLASS_NAMES = ['Ikan segar biasa', 'ikan sangat segar', 'ikan tidak segar']
IMG_SIZE = 224
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model_insang_ikan_pytorch.pth')

# ─── Validation Configuration ──────────────────────────────────────────
CONFIDENCE_THRESHOLD = 55.0       # Minimum confidence (%) to show result without warning
MIN_GILL_COLOR_RATIO = 0.12       # Minimum ratio of red/brown/pink pixels (12%)

# ─── Load Model ────────────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_model():
    model = models.mobilenet_v2(weights=None)
    num_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(num_features, len(CLASS_NAMES))
    )
    state_dict = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

model = load_model()
print(f"Model loaded on {device}")

# ─── Image Preprocessing ───────────────────────────────────────────────
val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ─── Fish Gill Color Validation ────────────────────────────────────────
def validate_fish_gill(image_bytes):
    """
    Analyze if the image likely contains fish gill based on color distribution.
    Fish gills are characterized by red, dark red, brown-red, and pink hues.
    
    Returns: (is_valid, color_ratio_percent, detail_message)
    """
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    # Resize for faster processing
    img_small = img.resize((150, 150))
    
    # Convert to HSV for better color analysis
    img_hsv = img_small.convert('HSV')
    hsv_arr = np.array(img_hsv)
    
    h = hsv_arr[:, :, 0]  # PIL HSV: H = 0-255
    s = hsv_arr[:, :, 1]  # S = 0-255
    v = hsv_arr[:, :, 2]  # V = 0-255
    
    total_pixels = h.shape[0] * h.shape[1]
    
    # ── Red hues (H wraps around 0/255 in PIL) ──
    # Deep red / bright red: H < 18 or H > 235 (in PIL's 0-255 scale)
    is_red = ((h < 18) | (h > 235)) & (s > 35) & (v > 30)
    
    # ── Brown / orange-brown hues ──
    # Brown-red to orange: H 18-45
    is_brown = ((h >= 18) & (h <= 45)) & (s > 30) & (v > 25)
    
    # ── Dark red / maroon ──
    # Dark tones with red hue
    is_dark_red = ((h < 22) | (h > 230)) & (s > 20) & (v > 15) & (v < 120)
    
    # ── Pink hues ──
    # Pink: reddish with moderate saturation
    is_pink = ((h < 15) | (h > 240)) & (s > 25) & (s < 180) & (v > 140)
    
    # Combine all gill-like colors
    gill_pixels = is_red | is_brown | is_dark_red | is_pink
    gill_ratio = np.sum(gill_pixels) / total_pixels
    ratio_pct = round(gill_ratio * 100, 2)
    
    is_valid = gill_ratio >= MIN_GILL_COLOR_RATIO
    
    if is_valid:
        detail = f"Warna insang terdeteksi ({ratio_pct}% piksel cocok)"
    else:
        detail = f"Gambar tidak memiliki ciri warna insang ikan ({ratio_pct}% piksel cocok, minimum {MIN_GILL_COLOR_RATIO*100}%)"
    
    return is_valid, ratio_pct, detail


def predict_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    tensor = val_transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)[0]
        conf, pred_idx = torch.max(probs, 0)
    
    label = CLASS_NAMES[pred_idx.item()]
    confidence = conf.item() * 100
    all_probs = {CLASS_NAMES[i]: round(probs[i].item() * 100, 2) for i in range(len(CLASS_NAMES))}
    return label, confidence, all_probs

# ─── Routes ────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'Tidak ada file yang diunggah'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Nama file kosong'}), 400

    allowed = {'png', 'jpg', 'jpeg', 'webp', 'bmp'}
    ext = file.filename.rsplit('.', 1)[-1].lower()
    if ext not in allowed:
        return jsonify({'error': f'Format file tidak didukung: .{ext}'}), 400

    image_bytes = file.read()

    # ── Step 1: Validate fish gill color ──
    color_valid, color_ratio, color_detail = validate_fish_gill(image_bytes)
    
    if not color_valid:
        return jsonify({
            'rejected': True,
            'reason': 'NOT_FISH_GILL',
            'message': 'Gambar yang diunggah tidak terdeteksi sebagai insang ikan.',
            'detail': color_detail,
            'color_ratio': color_ratio,
            'tips': [
                'Pastikan foto menunjukkan bagian insang ikan secara jelas',
                'Gunakan pencahayaan yang cukup agar warna insang terlihat',
                'Hindari foto dengan latar belakang yang mendominasi',
                'Foto insang sebaiknya diambil dari jarak dekat (close-up)'
            ]
        })

    # ── Step 2: Run prediction ──
    label, confidence, all_probs = predict_image(image_bytes)

    # Map label to display info
    label_map = {
        'ikan sangat segar': {
            'display': 'Sangat Segar',
            'emoji': '🐟',
            'color': '#10b981',
            'badge': 'SANGAT SEGAR',
            'desc': 'Insang berwarna merah cemerlang tanpa lendir. Ikan dalam kondisi prima dan aman dikonsumsi.',
            'score': 'A+'
        },
        'Ikan segar biasa': {
            'display': 'Segar',
            'emoji': '🐠',
            'color': '#3b82f6',
            'badge': 'SEGAR',
            'desc': 'Insang berwarna merah sedikit kusam dengan sedikit lendir. Ikan masih layak dan aman dikonsumsi.',
            'score': 'B'
        },
        'ikan tidak segar': {
            'display': 'Tidak Segar',
            'emoji': '⚠️',
            'color': '#ef4444',
            'badge': 'TIDAK SEGAR',
            'desc': 'Insang berubah warna kecoklatan dengan lendir tebal. Ikan mengalami kemunduran mutu dan tidak disarankan untuk dikonsumsi.',
            'score': 'C'
        }
    }

    info = label_map.get(label, {
        'display': label, 'emoji': '❓', 'color': '#6b7280',
        'badge': label.upper(), 'desc': '', 'score': '-'
    })

    # ── Step 3: Check confidence threshold ──
    warning = None
    if confidence < CONFIDENCE_THRESHOLD:
        warning = {
            'message': f'Tingkat kepercayaan model rendah ({confidence:.1f}%). Hasil mungkin kurang akurat.',
            'suggestion': 'Coba upload foto insang ikan yang lebih jelas dengan pencahayaan yang baik.'
        }

    response = {
        'rejected': False,
        'label': label,
        'display': info['display'],
        'emoji': info['emoji'],
        'color': info['color'],
        'badge': info['badge'],
        'description': info['desc'],
        'score': info['score'],
        'confidence': round(confidence, 2),
        'probabilities': all_probs,
        'color_ratio': color_ratio,
        'validation': {
            'color_check': 'passed',
            'confidence_check': 'passed' if confidence >= CONFIDENCE_THRESHOLD else 'warning'
        }
    }
    
    if warning:
        response['warning'] = warning

    return jsonify(response)

@app.route('/health')
def health():
    return jsonify({'status': 'ok', 'model': 'MobileNetV2', 'device': str(device)})

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)
