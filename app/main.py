from transformers import AutoProcessor, AutoModel
import torch
import scipy.io.wavfile as wavfile
import sys

MODEL_NAME = "suno/bark-small"
OUTPUT_FILE = "./output/audio.wav"

def main():
    if len(sys.argv) < 2:
        print("👉 Escribe el texto a convertir:")
        text = input("> ")
    else:
        text = " ".join(sys.argv[1:])

    print("🔄 Cargando modelo...")
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)

    print("🎤 Generando audio...")
    inputs = processor(text, return_tensors="pt")

    with torch.no_grad():
        audio_array = model.generate(**inputs)

    audio = audio_array[0].cpu().numpy()

    wavfile.write(OUTPUT_FILE, rate=22050, data=audio)

    print(f"✅ Audio generado: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
