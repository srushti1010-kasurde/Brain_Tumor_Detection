from flask import Flask, request, render_template, jsonify
import torch
from torchvision import transforms, models
from PIL import Image
import os
import io
import base64
import numpy as np
from datetime import datetime
from dotenv import load_dotenv
import google.generativeai as genai

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
load_dotenv()

app = Flask(__name__)
app.jinja_env.globals.update(enumerate=enumerate)  # allow enumerate in templates

# ── Model ─────────────────────────────────────────────────────────────────────
model = models.resnet18(pretrained=True)
model.fc = torch.nn.Linear(model.fc.in_features, 4)
model.load_state_dict(torch.load('multi_tumor_processed.pth', map_location='cpu'))
model.eval()

classes = ['🧠 Glioma', '🎯 Meningioma', '⚡ Pituitary', '✅ No Tumor']

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ── Report Data Tables ────────────────────────────────────────────────────────

TUMOR_INFO = {
    'Glioma': {
        'region':      'Cerebral Hemisphere (Parietal Lobe)',
        'grade_range': 'Grade II – IV',
        'description': (
            'Gliomas arise from glial cells that support and protect neurons. '
            'They are the most common primary brain tumor and can range from slow-growing '
            '(low-grade) to highly aggressive (glioblastoma, Grade IV). '
            'Location often determines which neurological functions are affected.'
        ),
        'symptoms': [
            'Persistent or worsening headaches',
            'Seizures or convulsions',
            'Cognitive changes (memory, concentration)',
            'Weakness or numbness on one side of the body',
            'Speech or language difficulties',
        ],
        'severity_base': 7.5,
    },
    'Meningioma': {
        'region':      'Meninges (Spinal Cord Lining)',
        'grade_range': 'Grade I – III',
        'description': (
            'Meningiomas develop from the meninges — the protective membranes surrounding the brain. '
            'Most are benign (Grade I) and slow-growing; many are found incidentally. '
            'When large enough to compress surrounding tissue, they can produce significant symptoms.'
        ),
        'symptoms': [
            'Headaches, often dull and persistent',
            'Vision problems or loss',
            'Hearing loss or ringing in the ears',
            'Weakness in arms or legs',
            'Seizures (less common than glioma)',
        ],
        'severity_base': 4.5,
    },
    'Pituitary': {
        'region':      'Pituitary Gland (Sella Turcica)',
        'grade_range': 'Grade I – II',
        'description': (
            'Pituitary tumors (adenomas) form in the pituitary gland, which regulates key hormones. '
            'Most are benign and non-invasive. Depending on whether they secrete hormones, '
            'they can disrupt the entire endocrine system even when small.'
        ),
        'symptoms': [
            'Bitemporal hemianopia (tunnel vision)',
            'Headaches behind the eyes',
            'Hormonal imbalances (fatigue, weight changes, infertility)',
            'Nausea and vomiting',
            'Galactorrhea or menstrual irregularities',
        ],
        'severity_base': 3.5,
    },
}

RECOMMENDATIONS = {
    'High': [
        'Immediate referral to a neurosurgeon or neuro-oncologist',
        'Advanced imaging recommended (contrast MRI / CT / PET)',
        'Multidisciplinary oncology team consultation',
        'Begin treatment planning without delay',
        'Psychological support for patient and family',
    ],
    'Medium': [
        'Urgent neurologist consultation within 1–2 weeks',
        'Consider contrast-enhanced MRI for clearer staging',
        'Biopsy may be required for definitive diagnosis',
        'Discuss surgical and non-surgical treatment options',
        'Monitor for symptom progression closely',
    ],
    'Low': [
        'Schedule a follow-up MRI in 3–6 months',
        'Consult a neurologist for clinical evaluation',
        'Keep a symptom journal to track any changes',
        'Maintain a healthy lifestyle (sleep, diet, low stress)',
    ],
}


# ── Report Generator ──────────────────────────────────────────────────────────

def build_report(tumor_label: str, confidence_pct: float, img_array: np.ndarray) -> dict:
    """
    Build a detailed clinical report from the model's predicted class,
    confidence score, and basic image statistics.
    """
    # strip emoji from class name  e.g. '🧠 Glioma' -> 'Glioma'
    clean_label = tumor_label.replace('🧠', '').replace('🎯', '').replace('⚡', '').replace('✅', '').strip()
    info = TUMOR_INFO[clean_label]

    # size estimate: image contrast (std) + confidence as proxy
    std_val = float(img_array.std())
    size_mm = round(12 + (std_val / 255) * 35 + (confidence_pct / 100) * 18, 1)

    # severity driven by tumor type base + confidence
    raw_score = info['severity_base'] + (confidence_pct - 50) / 100 * 2.5
    severity_score = round(min(max(raw_score, 1.0), 10.0), 1)

    if severity_score >= 7.0:
        severity = 'High'
    elif severity_score >= 4.5:
        severity = 'Medium'
    else:
        severity = 'Low'

    affected_area = round(min((size_mm / 80) * 100, 40.0), 1)

    return {
        'tumor_type':      clean_label,
        'region':          info['region'],
        'grade_range':     info['grade_range'],
        'description':     info['description'],
        'symptoms':        info['symptoms'],
        'recommendations': RECOMMENDATIONS[severity],
        'size_mm':         size_mm,
        'severity':        severity,
        'severity_score':  severity_score,
        'affected_area':   affected_area,
        'scan_date':       datetime.now().strftime('%B %d, %Y — %H:%M'),
        'report_id':       f'NSA-{datetime.now().strftime("%Y%m%d%H%M%S")}',
    }


def generate_heatmap(img_pil: Image.Image):
    """Blend a synthetic anomaly heatmap onto the MRI. Returns base64 PNG or None."""
    try:
        from scipy.ndimage import gaussian_filter
        img = img_pil.convert('RGB').resize((300, 300))
        arr = np.array(img, dtype=np.float32)
        gray = arr.mean(axis=2)
        smoothed = gaussian_filter(gray, sigma=22)
        norm = (smoothed - smoothed.min()) / (smoothed.max() - smoothed.min() + 1e-8)

        heatmap = np.zeros((300, 300, 3), dtype=np.float32)
        heatmap[:, :, 0] = norm * 255
        heatmap[:, :, 1] = (1 - norm) * 160
        heatmap[:, :, 2] = 30

        blend = (arr * 0.55 + heatmap * 0.45).clip(0, 255).astype(np.uint8)
        out = Image.fromarray(blend)
        buf = io.BytesIO()
        out.save(buf, format='PNG')
        return base64.b64encode(buf.getvalue()).decode()
    except Exception:
        return None


def img_to_b64(img_pil: Image.Image) -> str:
    buf = io.BytesIO()
    img_pil.resize((300, 300)).save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode()


# ── Gemini Chat ───────────────────────────────────────────────────────────────

def get_brain_health_response(message):
    try:
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            print('ERROR: GEMINI_API_KEY not found in environment')
            return None

        genai.configure(api_key=api_key)
        gemini_model = genai.GenerativeModel('gemini-1.5-flash')

        prompt = f"""You are Neuro Assistant, a warm and knowledgeable brain health guide built into a brain tumor detection app. You help patients and their families understand brain health topics and you give them answers of all their questions warmly explaining everything in detail like causes, lifestyle tips or anything they ask

Your expertise covers:
- Brain tumor types: glioma, meningioma, pituitary tumors, and what "no tumor" means
- Symptoms: headaches, seizures, vision changes, memory issues, balance problems
- Diagnosis: how MRI scans work, what contrast dye does, reading scan results
- Treatment options: surgery, radiation, chemotherapy, watchful waiting
- Risk factors and prevention tips
- What to expect after diagnosis (emotional support, next steps)
- General brain health and neurology basics

Behavior rules:
- If the user says hi/hello or greets you, greet them back warmly and briefly explain what you can help with
- Always answer in plain, compassionate language a patient can understand — avoid heavy jargon
- Keep answers concise (3-5 sentences) unless the topic needs more detail
- NEVER mention AI, machine learning, models, or technology
- NEVER give a specific diagnosis or say someone definitely has/doesn't have a tumor
- If asked something outside brain health (weather, news, coding, etc.), politely redirect: "I'm specialized in brain health topics — feel free to ask me about brain tumors, symptoms, or MRI scans!"
- End EVERY response with: "⚠️ This is educational information only, not medical advice. Please consult a qualified neurologist."

User message: {message}"""

        response = gemini_model.generate_content(prompt)
        return response.text.strip()

    except Exception as e:
        print(f'Gemini API error: {e}')
        return None


def get_fallback_response(message):
    msg = message.lower().strip()

    greetings = ['hi', 'hello', 'hey', 'good morning', 'good evening', 'good afternoon', 'howdy', 'hiya']
    if any(msg == g or msg.startswith(g + ' ') for g in greetings):
        return ("👋 Hello! I'm Neuro Assistant. I'm here to help you understand brain health topics — "
                "feel free to ask me about brain tumor types, symptoms, MRI scans, treatments, or what to expect after a diagnosis. "
                "⚠️ This is educational information only, not medical advice. Please consult a qualified neurologist.")

    if any(w in msg for w in ['what is a tumor', 'what are tumors', 'define tumor', 'what is tumor']):
        return ("A brain tumor is an abnormal growth of cells in or around the brain. "
                "Tumors can be benign (non-cancerous) or malignant (cancerous), and they vary widely in type, location, and how fast they grow. "
                "Common types include glioma, meningioma, and pituitary tumors. "
                "⚠️ This is educational information only, not medical advice. Please consult a qualified neurologist.")

    if 'glioma' in msg:
        return ("Gliomas are tumors that originate in the glial cells, which support and protect neurons in the brain. "
                "They range from low-grade (slow-growing, less aggressive) to high-grade (fast-growing, like glioblastoma). "
                "Symptoms depend on the tumor's location and may include headaches, seizures, or cognitive changes. "
                "⚠️ This is educational information only, not medical advice. Please consult a qualified neurologist.")

    if 'meningioma' in msg:
        return ("Meningiomas grow from the meninges — the protective membranes surrounding the brain and spinal cord. "
                "Most are benign and slow-growing, and many people live with them for years without symptoms. "
                "When they press on nearby brain tissue, they can cause headaches, vision problems, or weakness. "
                "⚠️ This is educational information only, not medical advice. Please consult a qualified neurologist.")

    if 'pituitary' in msg:
        return ("Pituitary tumors form in the pituitary gland at the base of the brain, which controls many hormones. "
                "Most are benign and may cause symptoms like vision problems, headaches, or hormonal imbalances (weight changes, fatigue, infertility). "
                "Treatment often involves medication, surgery, or radiation. "
                "⚠️ This is educational information only, not medical advice. Please consult a qualified neurologist.")

    if any(w in msg for w in ['symptom', 'sign', 'headache', 'seizure', 'vision', 'dizzy', 'nausea', 'weakness', 'memory']):
        return ("Common brain tumor symptoms include persistent headaches (especially in the morning), seizures, blurred or double vision, "
                "nausea, difficulty with balance, memory problems, or personality changes. "
                "These symptoms can also have many other causes, so it's important not to self-diagnose. "
                "⚠️ This is educational information only, not medical advice. Please consult a qualified neurologist.")

    if any(w in msg for w in ['mri', 'scan', 'imaging', 'contrast', 'xray', 'x-ray']):
        return ("MRI (Magnetic Resonance Imaging) is the gold standard for detecting brain tumors. "
                "It uses magnetic fields and radio waves to produce detailed images of brain tissue. "
                "A contrast dye (gadolinium) is often injected to make tumors more visible. "
                "⚠️ This is educational information only, not medical advice. Please consult a qualified neurologist.")

    if any(w in msg for w in ['treatment', 'surgery', 'chemo', 'radiation', 'cure', 'therapy', 'medicine', 'operation']):
        return ("Treatment depends on the tumor type, size, location, and the patient's overall health. "
                "Options include surgery to remove the tumor, radiation therapy, chemotherapy, targeted drug therapy, or watchful waiting for slow-growing tumors. "
                "A multidisciplinary team of neurosurgeons, oncologists, and radiologists work together to plan care. "
                "⚠️ This is educational information only, not medical advice. Please consult a qualified neurologist.")

    if any(w in msg for w in ['cause', 'risk', 'why', 'prevent', 'avoid', 'genetic', 'hereditary']):
        return ("The exact cause of most brain tumors is unknown. Known risk factors include radiation exposure, certain genetic syndromes, "
                "and a family history of brain tumors. Most brain tumors are not directly inherited. "
                "There's no guaranteed prevention, but maintaining general health may reduce overall cancer risk. "
                "⚠️ This is educational information only, not medical advice. Please consult a qualified neurologist.")

    return ("I'm here to help with brain health topics! You can ask me about brain tumor types (glioma, meningioma, pituitary), "
            "symptoms to watch for, how MRI scans work, treatment options, or what to do after a diagnosis. "
            "⚠️ This is educational information only, not medical advice. Please consult a qualified neurologist.")


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route('/')
def home():
    return render_template('home.html')


@app.route('/scan', methods=['GET', 'POST'])
def scan():
    if request.method == 'POST':
        if 'image' not in request.files:
            return render_template('scan.html', error='No image selected')

        file = request.files['image']
        if file.filename == '':
            return render_template('scan.html', error='No image selected')

        try:
            img_pil = Image.open(file.stream).convert('RGB')
            img_array = np.array(img_pil.resize((300, 300)))

            # ── Inference ────────────────────────────────────────────────────
            img_tensor = transform(img_pil).unsqueeze(0)
            with torch.no_grad():
                output = model(img_tensor)
                prob = torch.softmax(output, dim=1)
                pred = torch.argmax(output, dim=1).item()
                confidence = prob[0][pred].item() * 100

            tumor_type = classes[pred]        # e.g. '🧠 Glioma'
            has_tumor  = (pred != 3)          # index 3 = No Tumor

            if confidence > 60:
                status = 'confident'
            elif confidence > 40:
                status = 'uncertain'
            else:
                status = 'low_confidence'

            result = f"{tumor_type} ({confidence:.1f}%)"
            if status == 'low_confidence':
                result = f"❓ UNCERTAIN — {tumor_type} ({confidence:.1f}%)"

            # ── Build report only for tumor predictions ───────────────────────
            report       = None
            heatmap_b64  = None
            original_b64 = img_to_b64(img_pil)

            if has_tumor:
                report      = build_report(tumor_type, confidence, img_array)
                heatmap_b64 = generate_heatmap(img_pil)

            return render_template(
                'result.html',
                result=result,
                confidence=f'{confidence:.1f}%',
                tumor_type=tumor_type,
                status=status,
                has_tumor=has_tumor,
                report=report,
                original_b64=original_b64,
                heatmap_b64=heatmap_b64,
            )

        except Exception as e:
            import traceback
            traceback.print_exc()
            return render_template('scan.html', error=f'Error: {str(e)}')

    return render_template('scan.html')


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/chat', methods=['GET', 'POST'])
def chat():
    if request.method == 'POST':
        if request.is_json:
            data = request.get_json()
            user_msg = data.get('message', '').strip()
        else:
            user_msg = request.form.get('message', '').strip()

        if not user_msg:
            return jsonify({'response': 'Please enter a message about brain health.'})

        emergency_keywords = ['seizure having', 'unconscious', 'stroke', "can't breathe",
                               'emergency', 'collapsing', 'unresponsive']
        if any(keyword in user_msg.lower() for keyword in emergency_keywords):
            return jsonify({
                'response': ('🚨 This sounds like a medical emergency. Please call your local emergency number '
                             '(112 in India / 911 in USA) or go to the nearest hospital emergency department immediately.')
            })

        response = get_brain_health_response(user_msg)
        if not response:
            response = get_fallback_response(user_msg)

        return jsonify({'response': response})

    return render_template('chat.html')


@app.route('/settings')
def settings():
    return render_template('settings.html')


@app.route('/test-gemini')
def test_gemini():
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        return '❌ No API key found in environment'
    try:
        genai.configure(api_key=api_key)
        gemini_model = genai.GenerativeModel('gemini-1.5-flash')
        response = gemini_model.generate_content('Say hello in one sentence.')
        return f'✅ Gemini works: {response.text}'
    except Exception as e:
        return f'❌ Gemini error: {type(e).__name__}: {e}'


if __name__ == '__main__':
    app.run(debug=True, port=5000)