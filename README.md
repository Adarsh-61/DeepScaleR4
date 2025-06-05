# Vani-TTS as Part of DeepScaleR4

**Simple. Fast. Efficient.**

> ⚠️ **Note**: The core logic is concise, found in `main.py`. This README focuses on the user-facing application derived from `app.py`.

Vani-TTS is a high-performance, GPU-accelerated Text-to-Speech (TTS) application supporting Hindi and Gujarati. It utilizes Facebook's MMS-TTS models and is optimized for speed and efficiency, achieving Real-Time Factors (RTF) as low as 0.02-0.05 on appropriate hardware.

## 🌟 Features

* **Multi-language Support**: Hindi and Gujarati text-to-speech.
* **Hardware Acceleration**: Supports NVIDIA CUDA, Apple Silicon (MPS), and CPU fallback.
* **Real-time Performance**: Capable of sub-100ms inference times with suitable hardware.
* **Intelligent Caching**: Multi-level caching (in-memory and disk) for rapid retrieval of previously synthesized audio.
* **Comprehensive Analytics**: Real-time performance monitoring (RTF, memory usage, GPU stats) and system information display.
* **Optimized Backend**: Incorporates PyTorch optimizations like mixed precision, cuDNN benchmarking, TensorFloat-32, dynamic quantization, and efficient memory management.
* **Asynchronous Processing**: Features a non-blocking inference pipeline.
* **Cross-platform**: Compatible with Windows, macOS, and Linux.

## 🛠️ Installation

Follow these steps to install Vani-TTS:

### 1. Prerequisites

* **Python**: 3.9 or higher (3.11+ recommended for optimal performance).
* **System Memory**: 8GB RAM minimum; 16GB+ strongly recommended.
* **NVIDIA GPU Users**:
    * CUDA Toolkit: Version 12.6 or higher.
    * Compatible NVIDIA drivers.
* **Internet Connection**: Required for initial model downloads (The approximate total size for both models is 300MB.) and dependencies.

### 2. Install UV Package Manager (Recommended)

UV is a fast Python package installer and resolver.

* **Windows (PowerShell)**:
    ```powershell
    powershell -c "irm [https://astral.sh/uv/install.ps1](https://astral.sh/uv/install.ps1) | iex"
    ```
* **macOS/Linux (Shell)**:
    ```bash
    curl -LsSf [https://astral.sh/uv/install.sh](https://astral.sh/uv/install.sh) | sh
    ```
* **Alternative (using pip)**:
    ```bash
    pip install uv
    ```
    Ensure `uv` is in your system's PATH after installation.

### 3. Clone and Set Up Virtual Environment

1.  **Clone the Repository**:
    ```bash
    git clone [https://github.com/Adarsh-61/DeepScaleR4.git](https://github.com/Adarsh-61/DeepScaleR4.git)
    cd DeepScaleR4
    ```

2.  **Create a Virtual Environment (using UV recommended)**:
    ```bash
    uv venv
    ```

3.  **Activate Virtual Environment**:
    * **Windows (PowerShell/CMD)**:
        ```powershell
        .venv\Scripts\activate  # PowerShell
        ```
        ```cmd
        .venv\Scripts\activate.bat  # Command Prompt
        ```
    * **macOS/Linux (Bash/Zsh)**:
        ```bash
        source .venv/bin/activate
        ```

### 4. Install Dependencies

1.  **Install Core Application Dependencies (using UV recommended)**:
    ```bash
    uv pip install -r requirements.txt
    ```

2.  **Install PyTorch (Choose ONE based on your hardware)**:

    * **For NVIDIA GPU Users (Recommended for Best Performance)**:
        Ensure your NVIDIA drivers and CUDA Toolkit (12.6+) are correctly installed.
        ```bash
        # CUDA 12.6 recommended for latest GPUs. Adjust if using a different CUDA version.
        uv pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu126](https://download.pytorch.org/whl/cu126)
        ```

    * **For Apple Silicon (M1/M2/M3/M4 Series with Metal Performance Shaders - MPS)**:
        ```bash
        uv pip install torch torchvision torchaudio
        ```

    * **For CPU-Only Systems**:
        Performance will be significantly slower.
        ```bash
        uv pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cpu](https://download.pytorch.org/whl/cpu)
        ```
3.  **Optional: GPU Monitoring Tools (for NVIDIA)**:
    If you want detailed GPU monitoring in the analytics panel and use an NVIDIA GPU, uncomment the relevant lines for `pynvml` and `nvidia-ml-py` in `requirements.txt` *before* running `uv pip install -r requirements.txt`, or install them separately:
    ```bash
    uv pip install pynvml nvidia-ml-py3 # nvidia-ml-py3 is a common fork for pynvml
    ```

## 🚀 Quick Start

1.  **Ensure your virtual environment is activated.**
2.  **Start the Application**:
    ```bash
    python app.py
    ```
    The application will initialize, download models (on the first run for each language), and start the Gradio web server.
3.  **Open Your Browser**:
    Navigate to the local URL displayed in your terminal (typically `http://127.0.0.1:7860`).
4.  **Generate Speech**:
    * Enter your Hindi or Gujarati text.
    * Select the correct language.
    * Click "Generate Speech".
    * Playback or download the generated audio.

> 🌐 **Note**: To share the app interface publicly, the `app.py` launches Gradio with `share=True` by default. A link like `https://<unique_hash>.gradio.live` will be generated. This link works as long as your computer is on and the script is running.

## 📖 Detailed Usage Guide

### Web Interface

* **Text Input**:
    * Supports Devanagari script for Hindi (e.g., `नमस्ते दुनिया`) and Gujarati script for Gujarati (e.g., `નમસ્તે દુનિયા`).
    * Models typically support sequences up to ~4096 tokens. Very long texts might be truncated or cause issues.
* **Language Selection**:
    * **Hindi**: Uses `facebook/mms-tts-hin`.
    * **Gujarati**: Uses `facebook/mms-tts-guj`.
    Ensure the selected language matches the input text.
* **Audio Output**:
    * **Format**: WAV.
    * **Sample Rate**: 16kHz (model default).
    * **Controls**: Playback and download options are provided.

### Performance Analytics & System Information

The interface includes expandable sections for:
* **Performance Analytics**: Displays real-time metrics like RTF, memory usage, GPU utilization (if applicable), cache performance, and an overall performance grade. Click "Refresh Analytics" to update.
* **System Information**: Shows details about your OS, CPU, Memory, GPU (if detected), and Python/PyTorch framework versions.

## ⚙️ Configuration

The application is designed for auto-optimization. Key parameters are managed within the `UltimateConfig` class in `app.py`. While modification is possible, it requires understanding their impact. Examples:
```python
# Illustrative parameters in app.py:
CACHE_SIZE = 4096              # In-memory audio cache entries
USE_MIXED_PRECISION = True     # FP16 on compatible GPUs
USE_DYNAMIC_QUANTIZATION = True # INT8 quantization
```
The application automatically applies hardware-specific optimizations (CUDA for NVIDIA, MPS for Apple Silicon, CPU vectorization) based on detected hardware.

## 🔧 Troubleshooting

### Common Issues

* **CUDA Out of Memory (OOM)**:
    * Close other GPU-intensive applications.
    * MMS models are relatively large; a GPU with at least 6-8GB VRAM is recommended.
    * Consider CPU mode (slower) if VRAM is insufficient.
* **Slow Performance**:
    * **Verify GPU Utilization**: Use the app's analytics or system tools (`nvidia-smi`, Activity Monitor). If GPU isn't used, check PyTorch installation and drivers.
    * **Update GPU Drivers**: Ensure latest stable drivers.
    * **System Load**: Close unnecessary background apps.
    * **Initial Run Latency**: First inference is slower due to model loading/caching. Subsequent runs are faster.
* **Installation Issues**:
    * Ensure the virtual environment is active.
    * Try reinstalling core dependencies: `uv pip install --reinstall torch transformers gradio psutil numpy scipy`
    * For CUDA issues, verify CUDA toolkit and PyTorch compatibility. Reinstall PyTorch for your specific CUDA version (see installation section).
* **Port Already in Use (e.g., 7860)**:
    * Identify and terminate the conflicting process (e.g., using `netstat -ano | findstr "7860"` on Windows or `sudo lsof -i :7860` on macOS/Linux).
    * Alternatively, modify `iface.launch(server_port=XXXX)` in `app.py` to use a different port.

### Performance Optimization Tips

* **Hardware**:
    * **GPU**: NVIDIA RTX 3060 / Apple M1 Pro (or equivalents) or better with 8GB+ VRAM for optimal RTF.
    * **RAM**: 16GB+ system RAM.
    * **Storage**: SSD (NVMe preferred).
* **System**: Minimize background processes, ensure good cooling, use a high-performance power plan.
* **Software**: Keep dependencies updated, always use a virtual environment.

## 📊 Performance Benchmarks (Illustrative)

Performance varies greatly with hardware. RTF (Real-Time Factor) is key; lower is better. RTF < 1.0 means faster-than-real-time synthesis.

| Hardware Configuration        | Estimated RTF | VRAM/RAM Usage | Notes                                    |
|-------------------------------|---------------|----------------|------------------------------------------|
| NVIDIA RTX 4090 (24GB VRAM)   | 0.01 - 0.03   | 2-4GB VRAM     | S++ tier (Exceptional)                   |
| NVIDIA RTX 3080 (10GB VRAM)   | 0.03 - 0.06   | 3-5GB VRAM     | S+ tier (Outstanding)                    |
| Apple M3 Max (Unified Memory) | 0.06 - 0.12   | 4-7GB (shared) | S to A+ tier (Excellent MPS)           |
| Intel Core i9 (e.g., 12900K)  | 0.5 - 1.0     | 6-10GB RAM     | CPU-only (Acceptable for non-real-time)  |

## 🏗️ Architecture Overview

The application (`app.py`) consists of several core classes:

* **`UltimateOptimizedTTS`**: Manages TTS model loading, optimization, inference pipeline, and device selection.
* **`UltimatePerformanceTracker`**: Collects and analyzes real-time performance metrics.
* **`AdvancedCache`**: Implements multi-level (memory/disk) caching for audio.
* **`ComprehensiveSystemInfo`**: Gathers and presents detailed system hardware/software information.
* **`UltimateConfig`**: Defines configuration parameters and environment settings for optimization.

The optimization pipeline involves model loading/preparation (quantization, compilation), tokenization, efficient model inference, post-processing, and caching.

## 🧪 Development

### Project Structure
```
DeepScaleR4/
├── app.py                    # Main application script
├── README.md
├── requirements.txt
└── main.py
```

### Development Setup
1.  Follow the main **Installation** steps (cloning, virtual environment, dependencies).
2.  Install development tools (optional):
    ```bash
    uv pip install black flake8 mypy pytest
    ```
3.  Run the application: `python app.py`

### Contributing
Fork the repo, create a branch, make changes, test, document, and submit a pull request.

### Adding New Languages
To add support for other Facebook MMS-TTS models:
1.  Identify the HuggingFace model ID (e.g., `facebook/mms-tts-ben` for Bengali).
2.  In `app.py`, update `models_config` in `UltimateOptimizedTTS._load_ultimate_optimized_models`.
3.  Update the `choices` in `language_input = gr.Radio(...)` in `create_ultimate_interface`.
4.  Add warmup texts for the new language in `UltimateOptimizedTTS._ultimate_warmup`.
5.  Test thoroughly.

## 📋 System Requirements

### Minimum
* **OS**: Windows 10/11 (64-bit), macOS 10.15+, modern Linux (Ubuntu 18.04+).
* **Python**: 3.9+ (3.11+ recommended).
* **RAM**: 8GB (16GB strongly recommended).
* **Storage**: ~5GB free disk space (SSD highly recommended).
* **Internet**: For initial downloads.

### Recommended for Optimal Performance
* **OS**: Windows 11, macOS 12+, Ubuntu 20.04+.
* **Python**: 3.11+.
* **RAM**: 16GB+ (DDR4/DDR5).
* **GPU**:
    * NVIDIA: RTX 3060+ (8GB+ VRAM), CUDA 12.6+.
    * Apple Silicon: M1 Pro/Max, M2/M3 series.
* **Storage**: 10GB+ NVMe SSD.

## 🔒 Security and Privacy

* **Local Processing**: All TTS synthesis occurs locally. No text or audio is sent to external servers.
* **Model Provenance**: Models are downloaded from HuggingFace Hub.
* **Local Caching**: Audio can be cached on your disk for speed.
* **Dependency Security**: Use `uv` or `pip` with reputable indices. Run in a virtual environment.
* **Network Exposure**: Gradio's `share=True` (active by default in `app.py`) makes the app accessible via a public URL. Be mindful of this if running on an untrusted network or with sensitive capabilities.

## 📄 License

This project is licensed under the MIT License.

## 🤝 Acknowledgments
* Meta AI Research (MMS-TTS, VITS)
* HuggingFace (Transformers, Model Hub)
* PyTorch Team
* Gradio Team
* Python Open Source Community

## 📞 Support
1.  Review this README and the Troubleshooting section.
2.  Verify system requirements and driver versions.
3.  For issues, provide OS details, Python version, hardware specs, error messages, and steps to reproduce.
4.  For performance inquiries, include analytics output and system resource usage.

---
*Vani-TTS: Simple, Fast, and Efficient Text-to-Speech for Hindi and Gujarati, as part of the DeepScaleR4 project.*
