# Vani-TTS (Part of DeepScaleR4)

**Simple. Fast. Efficient.**

> ⚠️ **Note**: The main logic is in `main.py`. This README explains the user-facing app in `app.py`.

Vani-TTS is a fast, GPU-powered Text-to-Speech (TTS) tool for Hindi and Gujarati. It uses Facebook’s MMS-TTS models and is built for speed. On good hardware, it can run as fast as 0.02–0.05 Real-Time Factor (RTF).

---

## 🌟 Features

- **Supports Two Languages**: Hindi and Gujarati TTS.
- **Hardware Acceleration**: Works with NVIDIA GPUs (CUDA), Apple Silicon (MPS), and can fall back to CPU.
- **Real-Time Performance**: Can finish inference in less than 100 ms on suitable hardware.
- **Smart Caching**: Keeps recent audio in memory and on disk for quick re-use.
- **Live Analytics**: Shows RTF, memory use, GPU stats, and system info.
- **Optimized Backend**: Uses PyTorch tricks like mixed precision, cuDNN tuning, TensorFloat-32, dynamic quantization, and smart memory handling.
- **Non-Blocking Pipeline**: Runs inference without freezing the rest of the app.
- **Cross-Platform**: Works on Windows, macOS, and Linux.

---

## 🛠️ Installation

Follow these steps to get Vani-TTS up and running:

### 1. What You Need

- **Python** 3.9 or newer (3.11+ is best).
- At least **8 GB of RAM** (16 GB or more is recommended).
- If you have an **NVIDIA GPU**:
  - CUDA Toolkit 12.6 or newer.
  - Compatible GPU drivers.
- A working **internet connection** to download the models (about 300 MB total) and other packages.

### 2. Install UV (Optional but Recommended)

UV is a faster alternative to pip for installing Python packages.

- **Windows (PowerShell)**:
  ```powershell
  powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
````

* **macOS/Linux (Terminal)**:

  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```
* **If you prefer pip**:

  ```bash
  pip install uv
  ```

  After that, make sure `uv` is in your system PATH.

### 3. Clone the Repo & Set Up a Virtual Environment

1. **Clone the GitHub repository**:

   ```bash
   git clone https://github.com/Adarsh-61/DeepScaleR4.git
   cd DeepScaleR4
   ```

2. **Create a virtual environment** (using UV if you installed it, otherwise use `python -m venv`):

   ```bash
   uv venv
   ```

   If you don’t have UV, run:

   ```bash
   python -m venv .venv
   ```

3. **Activate the virtual environment**:

   * **Windows (PowerShell)**:

     ```powershell
     .venv\Scripts\activate
     ```
   * **Windows (Command Prompt)**:

     ```cmd
     .venv\Scripts\activate.bat
     ```
   * **macOS/Linux (bash or zsh)**:

     ```bash
     source .venv/bin/activate
     ```

### 4. Install Python Dependencies

1. **Install the core requirements**:

   ```bash
   uv pip install -r requirements.txt
   ```

   Or, if you don’t have UV:

   ```bash
   pip install -r requirements.txt
   ```

2. **Install PyTorch (pick one based on your hardware)**:

   * **For NVIDIA GPU users** (best performance):
     Make sure your CUDA Toolkit (12.6+) and drivers are set up, then run:

     ```bash
     uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
     ```
   * **For Apple Silicon (M1/M2/M3/M4)**:

     ```bash
     uv pip install torch torchvision torchaudio
     ```
   * **For CPU-only systems** (slower):

     ```bash
     uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
     ```

3. **(Optional) Install NVIDIA GPU monitoring tools**:
   If you want to see GPU stats in the analytics panel, install these before (or after) the core requirements:

   ```bash
   uv pip install pynvml nvidia-ml-py3
   ```

---

## 🚀 Quick Start

1. **Activate your virtual environment** (if it’s not already active).
2. **Run the app**:

   ```bash
   python app.py
   ```

   On the first run, it will download the Hindi and Gujarati models (about 300 MB). Then it starts a Gradio web interface.
3. **Open your browser** and go to the URL shown in your terminal (usually `http://127.0.0.1:7860`).
4. **Use the interface**:

   * Type your Hindi or Gujarati text.
   * Choose the correct language.
   * Click **Generate Speech**.
   * Listen to or download the WAV audio.

> 🌐 **Note**: By default, `app.py` sets `share=True`. That means you’ll also get a temporary public link like `https://abcdefg.gradio.live`. You can share that link as long as the app is running.

---

## 📖 Detailed Usage Guide

### Web Interface

* **Text Input**:

  * You can type in Devanagari (Hindi) or Gujarati script.
  * Models support up to about 4096 tokens. Very long text might get cut off or cause errors.
* **Language Selection**:

  * **Hindi** uses `facebook/mms-tts-hin`.
  * **Gujarati** uses `facebook/mms-tts-guj`.
* **Audio Output**:

  * Format: WAV, 16 kHz sample rate.
  * You get playback buttons and a download link.

### Performance Analytics & System Info

In the web UI, you’ll find expandable panels for:

* **Performance Analytics**: Real-time stats like RTF, memory usage, GPU usage, cache hits, and a performance score. Click **Refresh Analytics** to update.
* **System Information**: Shows your OS, CPU, RAM, GPU (if any), Python version, and PyTorch version.

---

## ⚙️ Configuration

Most settings are handled automatically in the `UltimateConfig` class (inside `app.py`). You can tweak things if you understand how they work. For example:

```python
# Example settings in app.py:
CACHE_SIZE = 4096              # How many audio clips to keep in memory
USE_MIXED_PRECISION = True     # Use FP16 on compatible GPUs
USE_DYNAMIC_QUANTIZATION = True # Use INT8 quantization on CPU
```

The app will pick CUDA (NVIDIA), MPS (Apple), or CPU depending on what hardware it detects.

---

## 🔧 Troubleshooting

### Common Problems

* **CUDA Out of Memory (OOM)**:

  * Close other GPU-heavy apps.
  * MMS models can be big; you need at least 6–8 GB of GPU VRAM.
  * If you still run out of memory, use CPU mode (it’ll be slower).

* **Slow Performance**:

  * Check if your GPU is actually being used. Run `nvidia-smi` (Linux/macOS) or check Task Manager on Windows.
  * Make sure you installed PyTorch with the right CUDA version.
  * Close other programs that are eating CPU/GPU.
  * The very first run is slower because it downloads and caches models. Later runs are faster.

* **Installation Errors**:

  * Double-check that your virtual environment is active.
  * Try reinstalling packages:

    ```bash
    uv pip install --reinstall torch transformers gradio psutil numpy scipy
    ```
  * If you have CUDA errors, make sure your CUDA Toolkit and PyTorch versions match.

* **Port Already in Use (e.g., 7860)**:

  * Find and stop the process using that port:

    * **Windows**: `netstat -ano | findstr "7860"`
    * **macOS/Linux**: `sudo lsof -i :7860`
  * Or change the port in `app.py` (look for `iface.launch(server_port=XXXX)`).

### Performance Tips

* **Hardware**:

  * For best results, use an NVIDIA RTX 3060 or newer (8 GB+ VRAM) or Apple M1 Pro/Max.
  * At least 16 GB of RAM.
  * Store everything on an SSD (NVMe preferred).
* **System**:

  * Close background apps.
  * Make sure your computer is cool and on a high-performance power plan.
* **Software**:

  * Keep Python packages up to date.
  * Always run inside a virtual environment.

---

## 📊 Performance Benchmarks (Example)

These numbers will vary by machine. Lower RTF means faster-than-real-time.

| Hardware Configuration        | Estimated RTF | VRAM/RAM Usage | Notes                        |
| ----------------------------- | ------------- | -------------- | ---------------------------- |
| NVIDIA RTX 4090 (24 GB VRAM)  | 0.01 – 0.03   | 2–4 GB VRAM    | Top tier (Exceptional speed) |
| NVIDIA RTX 3080 (10 GB VRAM)  | 0.03 – 0.06   | 3–5 GB VRAM    | Very fast                    |
| Apple M3 Max (Unified Memory) | 0.06 – 0.12   | 4–7 GB memory  | Great on Apple MPS           |
| Intel Core i9 (CPU only)      | 0.5 – 1.0     | 6–10 GB RAM    | Slow, but okay for testing   |

---

## 🏗️ Architecture Overview

This is what’s inside `app.py`:

* **`UltimateOptimizedTTS`**
  Loads the TTS models, applies optimizations, handles inference, and picks the device (CUDA/MPS/CPU).

* **`UltimatePerformanceTracker`**
  Gathers real-time stats on RTF, memory, GPU, cache, and so on.

* **`AdvancedCache`**
  Stores audio in memory and on disk for faster re-use.

* **`ComprehensiveSystemInfo`**
  Finds out details about your OS, CPU, GPU, RAM, Python, and PyTorch.

* **`UltimateConfig`**
  Holds configuration values and decides which optimizations to apply.

The pipeline is roughly:

1. Load the model (with quantization or compilation if needed).
2. Tokenize the input.
3. Run inference.
4. Post-process the audio.
5. Cache results.

---

## 🧪 Development

### Project Structure

```
DeepScaleR4/
├── app.py
├── README.md
├── requirements.txt
└── main.py
```

### Setting Up for Development

1. Follow the **Installation** steps above (clone, create venv, install deps).
2. Install tools to check code style (optional):

   ```bash
   uv pip install black flake8 mypy pytest
   ```
3. Run the app:

   ```bash
   python app.py
   ```

### How to Contribute

1. Fork the repo.
2. Create a new branch.
3. Make your changes.
4. Test everything.
5. Update documentation if needed.
6. Submit a pull request.

### Adding a New Language

1. Find the HuggingFace model ID (e.g., `facebook/mms-tts-ben` for Bengali).
2. In `app.py`, go to `UltimateOptimizedTTS._load_ultimate_optimized_models` and add your model to `models_config`.
3. In `create_ultimate_interface`, update the language choices in `language_input = gr.Radio(...)`.
4. Add warmup text for that language in `UltimateOptimizedTTS._ultimate_warmup`.
5. Test the new language thoroughly.

---

## 📋 System Requirements

### Minimum

* **OS**: Windows 10/11 (64-bit), macOS 10.15+, Ubuntu 18.04+ (or similar).
* **Python**: 3.9 or newer (3.11+ recommended).
* **RAM**: 8 GB (16 GB+ recommended).
* **Storage**: About 5 GB free (SSD recommended).
* **Internet**: Required for initial model downloads.

### Recommended

* **OS**: Windows 11, macOS 12+, Ubuntu 20.04+.
* **Python**: 3.11+.
* **RAM**: 16 GB or more.
* **GPU**:

  * NVIDIA RTX 3060+ (8 GB+ VRAM) with CUDA 12.6+.
  * Apple Silicon (M1 Pro/Max, M2/M3).
* **Storage**: 10 GB+ NVMe SSD.

---

## 🔒 Security and Privacy

* **All TTS processing happens locally.** No text or audio is sent to external servers.
* **Models come from HuggingFace Hub.**
* **Audio caching is local.**
* **Use a virtual environment** and install packages from trusted sources (pip or UV).
* **Be careful if you use Gradio’s public link** (`share=True`). Anyone with that link can access your app while it’s running.

---

## 📄 License

This project is licensed under the **MIT License**.

---

## 🤝 Acknowledgments

* Meta AI Research (MMS-TTS, VITS)
* HuggingFace (Transformers, Model Hub)
* PyTorch Team
* Gradio Team
* The Python Open Source Community

---

## 📞 Support

1. Read this README and the **Troubleshooting** section above.
2. Check that your system meets the requirements and that drivers are up to date.
3. If you still have issues, share your OS version, Python version, hardware specs, error messages, and steps to reproduce the problem.
4. For performance questions, include the analytics output and resource usage stats.

---

*Vani-TTS: Simple, Fast, and Efficient Text-to-Speech for Hindi and Gujarati, part of the DeepScaleR4 project.*
