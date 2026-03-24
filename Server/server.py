from flask import Flask, request, jsonify
from transformers import MarianMTModel, MarianTokenizer
import torch
import os

app = Flask(__name__)

# ============= MODEL PATHS =============
EN_HI_PATH = r"C:\Langauge Translator Project\En-Hi language Translator\my_hi_translator"
HI_EN_PATH = r"C:\Langauge Translator Project\Hi-En language Translator\my_en_translator"

# ============= LOAD BOTH MODELS AT STARTUP =============
print("Loading models...")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# English → Hindi
print("Loading En→Hi model...")
en_hi_tokenizer = MarianTokenizer.from_pretrained(EN_HI_PATH, local_files_only=True)
en_hi_model     = MarianMTModel.from_pretrained(EN_HI_PATH, local_files_only=True).to(device)
en_hi_model.eval()
print("En→Hi model loaded ✓")

# Hindi → English
print("Loading Hi→En model...")
hi_en_tokenizer = MarianTokenizer.from_pretrained(HI_EN_PATH, local_files_only=True)
hi_en_model     = MarianMTModel.from_pretrained(HI_EN_PATH, local_files_only=True).to(device)
hi_en_model.eval()
print("Hi→En model loaded ✓")

print("\nServer ready!\n")

# ============= TRANSLATION FUNCTION =============
def translate(text, tokenizer, model, max_length=128, num_beams=4):
    inputs = tokenizer(
        text, return_tensors="pt",
        padding=True, truncation=True,
        max_length=max_length
    ).to(device)

    with torch.no_grad():
        translated = model.generate(
            **inputs,
            max_length=max_length,
            num_beams=num_beams
        )

    return tokenizer.decode(translated[0], skip_special_tokens=True)

# ============= API ENDPOINT =============
@app.route('/translate', methods=['POST'])
def translate_endpoint():
    try:
        data      = request.get_json()
        text      = data.get('text', '').strip()
        direction = data.get('direction', 'en-hi').strip()

        if not text:
            return jsonify({"error": "No text provided"}), 400

        print(f"[Request] direction={direction}  text={text}")

        if direction == "en-hi":
            result = translate(text, en_hi_tokenizer, en_hi_model)
        elif direction == "hi-en":
            result = translate(text, hi_en_tokenizer, hi_en_model)
        else:
            return jsonify({"error": f"Unknown direction: {direction}"}), 400

        print(f"[Result]  {result}\n")
        return jsonify({"translation": result})

    except Exception as e:
        print(f"[ERROR] {e}")
        return jsonify({"error": str(e)}), 500

# ============= HEALTH CHECK =============
@app.route('/ping', methods=['GET'])
def ping():
    return jsonify({"status": "ok", "device": device})

# ============= RUN SERVER =============
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)