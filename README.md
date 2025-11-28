# 🎙️ Text to Audio (TTS con Python + HuggingFace)

Este proyecto permite convertir texto en audio utilizando modelos de **Text-To-Speech (TTS)** mediante **HuggingFace Transformers**, ejecutado dentro de un entorno Docker.

El sistema corre en modo interactivo por consola: escribís un texto y genera un archivo de audio `.wav`.

---

## 🧠 ¿Qué es TTS?

**TTS (Text To Speech)** es una tecnología que convierte texto escrito en voz sintética generada por inteligencia artificial.


## 🧪 Uso del sistema

```bash
git clone <tu-repo>
cd text-to-audio
```

Ejecutas:
```bash
./start
```

Vas a ver:
```bash
🔊 Sistema de texto a audio iniciado
Escribí un texto (o 'exit' para salir):
```

Ahora solo tenes que ingresar un texto:
```bash
> Hola, soy una inteligencia artificial
```

Y se generará:
```bash
✅ Audio generado: ./output/audio_1.wav
```
