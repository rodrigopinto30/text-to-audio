from transformers import AutoProcessor, AutoModel
import torch
import scipy.io.wavfile as wavfile
import os

MODEL_NAME = "suno/bark-small"
OUTPUT_DIR = "./output"

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("🔄 Cargando modelo, esto puede tardar un poco...")
processor = AutoProcessor.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)

print("\n🔊 Sistema de texto a audio iniciado")
print("Escribí un texto (o 'exit' para salir):\n")

counter = 1

while True:
    text = input("> ")

    if text.lower() in ["exit", "salir", "quit"]:
        print("👋 Saliendo...")
        break

    if len(text.strip()) == 0:
        print("⚠️ Texto vacío, intentá de nuevo")
        continue

    print("🎤 Generando audio...")

    inputs = processor(text, return_tensors="pt")

    with torch.no_grad():
        audio_array = model.generate(**inputs)

    audio = audio_array[0].cpu().numpy()

    output_file = f"{OUTPUT_DIR}/audio_{counter}.wav"
    wavfile.write(output_file, rate=22050, data=audio)

    print(f"✅ Audio generado: {output_file}\n")

    counter += 1
