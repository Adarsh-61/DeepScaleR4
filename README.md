# Vani-TTS as Part of DeepScaleR4

**Simple. Fast. Efficient.**

> ⚠️ **Note**: I know what you're thinking… 2000 lines of code? 😳 Don’t worry! The real code is actually less than 50 lines, all inside a small file called main.py. It just looks big, thanks to Claude Sonnet 4 for making it look so clean and fancy. So no need to panic — it’s still the same simple code in a new outfit!

A high-performance, GPU-accelerated Text-to-Speech (TTS) application supporting Hindi and Gujarati languages. Built with cutting-edge optimizations for maximum speed and efficiency using Facebook's MMS-TTS models. This state-of-the-art implementation achieves unprecedented inference speeds with Real-Time Factors (RTF) as low as 0.02-0.05, making it one of the fastest neural TTS systems available for Indian languages.

## 🌟 Features

### Core Capabilities
- **Multi-language Support**: Hindi and Gujarati text-to-speech
- **GPU Acceleration**: CUDA, MPS (Apple Silicon), and CPU fallback support
- **Real-time Performance**: Sub-100ms inference times with proper hardware
- **Advanced Caching**: Multi-level intelligent caching system
- **Comprehensive Analytics**: Real-time performance monitoring and metrics

### Performance Optimizations
- **PyTorch Optimizations**: Mixed precision, cuDNN benchmarking, and TensorFloat-32
- **Memory Management**: Advanced memory pooling and efficient tensor operations
- **Hardware Acceleration**: CUDA, Metal Performance Shaders, and CPU vectorization
- **Intelligent Caching**: Multi-level caching system with memory and disk storage
- **Batch Processing**: Efficient batching for improved throughput (when applicable)

### Advanced Features
- **Dynamic Quantization**: INT8 quantization for compatible operations
- **Memory Optimization**: Advanced memory pooling and tensor reuse
- **Asynchronous Processing**: Non-blocking inference pipeline
- **Performance Profiling**: Detailed timing breakdowns and system monitoring
- **Persistent Caching**: Disk-based cache with intelligent eviction
- **Cross-platform**: Windows, macOS, and Linux support

## 🛠️ Installation

### Prerequisites

- **Python**: 3.8 or higher (3.11+ recommended for optimal performance and compatibility)
- **System Memory**: 8GB RAM minimum; 16GB+ strongly recommended for smoother operation, especially with larger models or batch processing
- **CUDA Toolkit**: Version 12.6 or higher (specifically if using NVIDIA GPUs for acceleration). Ensure driver compatibility
- **Internet Connection**: Required for initial model downloads (~1-2GB per language) and dependency fetching

### Install Package Manager (Recommended)

UV is an extremely fast Python package installer and resolver, written in Rust, and is designed as a drop-in replacement for traditional package managers.

```bash
# Windows (PowerShell)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# macOS/Linux (Shell)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Alternative: Using pip to install UV
pip install uv
```
After installation, ensure `uv` is in your system's PATH.

### Quick Setup

1. **Clone the Repository**
   ```bash
   git clone https://github.com/Adarsh-61/DeepScaleR4.git
   cd DeepScaleR4
   ```

2. **Create and Activate Virtual Environment**
   It is highly recommended to use a virtual environment to manage project dependencies and avoid conflicts with system-wide packages.

   ```bash
   # Create a virtual environment using UV (recommended)
   uv venv
   ```

3. **Activate Virtual Environment**
   ```bash
   # Windows (PowerShell)
   .venv\Scripts\activate
   
   # macOS/Linux (Bash/Zsh)
   source .venv/bin/activate
   ```

4. **Install Dependencies**
   ```bash
   # Using UV (recommended for speed and reliability)
   uv pip install -r requirements.txt
   ```

### Installation Options

Tailor your installation to your specific hardware for optimal performance.

#### For CUDA Users (NVIDIA GPUs - Recommended for Best Performance)
Ensure your NVIDIA drivers and CUDA Toolkit (12.6+) are correctly installed.
```bash
# Install CUDA-optimized PyTorch using UV
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126 # CUDA 12.6 recommended for latest GPUs

# Install core application dependencies
uv pip install -r requirements.txt

# Optional: GPU monitoring tools (uncomment relevant lines in requirements.txt before running)
uv pip install pynvml nvidia-ml-py 
```

#### For Apple Silicon (M1/M2/M3/M4 Series with Metal Performance Shaders - MPS)
PyTorch provides native support for Apple Silicon, leveraging MPS for hardware acceleration.
```bash
# Install PyTorch with MPS support (usually the default for ARM64 macOS)
uv pip install torch torchvision torchaudio

# Install core application dependencies
uv pip install -r requirements.txt
```

#### For CPU-Only Systems
If no compatible GPU is available, the application will run on the CPU. Performance will be significantly slower.
```bash
# Install CPU-only PyTorch variant
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install core application dependencies
uv pip install -r requirements.txt
```

## 🚀 Quick Start

### Basic Usage

1. **Start the Application**
   Ensure your virtual environment is activated.
   ```bash
   python app.py
   ```
   The application will initialize, download models (on first run), and start the Gradio web server.

2. **Open Your Browser**
   Navigate to the local URL displayed in your terminal, typically `http://127.0.0.1:7860`.

3. **Generate Speech**
   - Enter your desired text in Hindi or Gujarati into the text input field
   - Select the appropriate language from the radio buttons
   - Click the "Generate Speech" button
   - The generated audio will appear, allowing for playback or download

### Command Line Usage
The application is primarily designed for web interface interaction.
```bash
# Default launch command
python app.py
```

> 🌐 **Note**: We can share this app with anyone by setting the `share` parameter to `True`. You just need internet access. A link like `https://e452f233dbd22f9a99.gradio.live` will be generated, which you can share with anyone around the world. However, this link will only work as long as your computer is turned on and the model is running in active mode.

## 📖 Detailed Usage Guide

### Web Interface

#### Text Input
- **Supported Scripts**:
  - Hindi: Devanagari script (e.g., `नमस्ते दुनिया`)
  - Gujarati: Gujarati script (e.g., `નમસ્તે દુનિયા`)
- **Character Limit**: The underlying models typically support sequences up to a certain length (e.g., 4096 tokens). Very long texts might be truncated or lead to performance issues.
- **Preprocessing**: Input text undergoes automatic tokenization and necessary preprocessing steps tailored for the VITS models.

#### Language Selection
- **Hindi**: Utilizes the `facebook/mms-tts-hin` model
- **Gujarati**: Utilizes the `facebook/mms-tts-guj` model

Ensure the correct language is selected to match the input text for coherent speech synthesis.

#### Audio Output
- **Format**: Audio is generated and presented in WAV format
- **Sample Rate**: Defaults to 16kHz, as per the VITS model's training
- **Quality**: High-fidelity neural speech synthesis, characteristic of VITS architecture
- **Controls**: The Gradio interface provides playback controls and a download button for the generated audio file

### Performance Analytics

The application features a sophisticated real-time performance monitoring dashboard:

#### Real-time Metrics Displayed
- **RTF (Real-Time Factor)**: Ratio of inference time to the duration of the synthesized audio. An RTF < 1.0 means synthesis is faster than real-time. Lower values indicate better performance.
- **Memory Usage**: Tracks current and peak RAM and VRAM (if GPU is used) consumption.
- **GPU Utilization**: For NVIDIA GPUs, displays real-time utilization percentage, memory usage, and temperature.
- **Cache Performance**: Statistics on cache hits and misses, indicating the effectiveness of the caching system.
- **System Resources**: Monitors CPU load, context switches, and I/O operations to provide a holistic view of system load.

#### Performance Grades
The system assigns a dynamic performance grade based on a composite score of RTF, cache efficiency, and resource utilization:
- **S++ (Perfect)**: RTF typically < 0.02, indicating exceptional, near-instantaneous synthesis
- **S+ (Exceptional)**: RTF typically < 0.05
- **S (Outstanding)**: RTF typically < 0.1
- **A+ (Excellent)**: RTF typically < 0.2
- **A through D**: Progressively lower grades indicating areas for potential optimization or system bottlenecks

### System Information

A dedicated section in the UI provides detailed insights into the host system's configuration:
- **CPU**: Model, architecture, core count, and current/max frequency
- **Memory**: Total/available/used RAM, swap space details
- **GPU**: Model, VRAM (total/available/used), driver version, CUDA version, and compute capability
- **Frameworks**: Versions of Python, PyTorch, NumPy, Gradio, and detected compute backends (CUDA, MPS, CPU)

## ⚙️ Configuration

### Performance Settings (`UltimateConfig` class in `app.py`)

The application is designed for auto-optimization, but key parameters can be inspected or (cautiously) modified within the `UltimateConfig` class in `app.py`:

```python
# Key configuration parameters (illustrative):
CACHE_SIZE = 4096              # Number of entries in the in-memory audio cache
MAX_BATCH_SIZE = 16            # Maximum number of requests processed in a single batch (if batching is enabled)
USE_MIXED_PRECISION = True     # Enables FP16 (half-precision) inference on compatible GPUs for significant speedup
USE_DYNAMIC_QUANTIZATION = True # Applies INT8 quantization to eligible model layers, reducing model size and potentially speeding up inference
TORCH_DTYPE = torch.float16    # Default data type for tensors on GPU (torch.float32 for CPU)
```
Modifying these requires understanding their impact on performance and stability.

### Hardware-Specific Optimizations

The application automatically applies several optimizations based on detected hardware:

#### NVIDIA GPU Users
- **CUDA Optimizations**: Leverages cuDNN benchmarking for selecting optimal convolution algorithms and enables TF32 (TensorFloat-32) on Ampere and newer GPUs for faster matrix math
- **Memory Management**: Attempts to reserve a significant portion of GPU memory (e.g., 98%) for PyTorch, minimizing overhead from other applications
- **Mixed Precision (FP16/AMP)**: Automatic Mixed Precision (AMP) is used to accelerate inference by performing operations in half-precision where possible, while maintaining model accuracy
- **Dynamic Quantization**: Converts weights and/or activations to INT8 dynamically, reducing memory footprint and computational cost for certain operations

#### Apple Silicon (M1/M2/M3/M4)
- **Metal Performance Shaders (MPS)**: Utilizes the MPS backend for PyTorch, enabling hardware acceleration on Apple's unified memory architecture
- **Unified Memory Advantage**: Efficient data transfer between CPU and GPU due to the shared memory architecture
- **Optimized Kernels**: Benefits from Apple's optimized implementations of neural network operations for their silicon

#### CPU-Only Systems
- **Multi-threading**: PyTorch operations are parallelized across available CPU cores. Thread count is typically managed by PyTorch or can be influenced by environment variables (e.g., `OMP_NUM_THREADS`)
- **SIMD Vectorization**: Modern CPUs with AVX/AVX2/AVX512 instruction sets can accelerate computations. PyTorch and underlying libraries like MKL or OpenBLAS leverage these
- **Memory Layout Optimizations**: Employs cache-friendly data structures and memory access patterns where possible

## 🔧 Troubleshooting

### Common Issues

#### CUDA Out of Memory (OOM)
This is a frequent issue when GPU VRAM is insufficient.
```bash
# Error message typically includes "CUDA out of memory"
# Solutions:
# 1. Close other applications using GPU VRAM
# 2. Reduce batch size if applicable (though this app is primarily single-request)
# 3. The MMS models are relatively large; a GPU with at least 6-8GB VRAM is recommended for comfortable use
# 4. If VRAM is consistently an issue, consider running in CPU mode (slower) or upgrading GPU
```

#### Slow Performance
1. **Verify GPU Utilization**:
   - NVIDIA: Use `nvidia-smi` command or the app's analytics
   - Apple Silicon: Check Activity Monitor for GPU usage
   - If GPU is not utilized, there might be an issue with PyTorch installation or driver compatibility
2. **Update GPU Drivers**: Ensure you have the latest stable drivers for your GPU
3. **System Load**: Close unnecessary background applications consuming CPU, RAM, or GPU resources
4. **Initial Run Latency**: The very first inference after starting the app might be slower due to model loading, JIT compilation, and cache warming. Subsequent inferences should be faster

#### Installation Issues
If you encounter problems during dependency installation:
```bash
# 1. Ensure your virtual environment is active
# 2. Try reinstalling core dependencies forcefully using UV:
uv pip install --reinstall torch transformers gradio psutil numpy scipy

# 3. For CUDA-specific issues, ensure CUDA toolkit and PyTorch version compatibility. Reinstall PyTorch for your CUDA version:
# Example for CUDA 12.6:
uv pip uninstall torch torchaudio torchvision
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

#### Port Already in Use (e.g., 7860)
If the default port for Gradio is occupied:
```bash
# Error message: "OSError: [Errno 98] Address already in use" or similar
# Identify process using the port:
# Windows: netstat -ano | findstr "7860"
# macOS/Linux: sudo lsof -i :7860
# Terminate the conflicting process or modify app.py to launch Gradio on a different port:
# iface.launch(server_name="127.0.0.1", server_port=7861) # Example for port 7861
```

### Performance Optimization Tips

1. **Hardware Recommendations**:
   - **GPU**: NVIDIA RTX 3060 (or equivalent AMD/Apple Silicon) or better for optimal RTF. More VRAM (8GB+) is beneficial
   - **RAM**: 16GB+ system RAM, especially if multitasking or using CPU mode
   - **Storage**: SSD (NVMe preferred) for faster application startup and model loading from disk cache

2. **System Configuration**:
   - Minimize background processes
   - Ensure adequate system cooling, especially for GPUs, to prevent thermal throttling during sustained use
   - Use a high-performance power plan on Windows/Linux

3. **Software Environment**:
   - Keep PyTorch, CUDA drivers (if applicable), and other dependencies updated to their latest stable versions
   - Always use a dedicated virtual environment to prevent dependency conflicts
   - Monitor the application's performance analytics to identify bottlenecks

## 📊 Performance Benchmarks

### Typical Performance Metrics (Illustrative)

Performance heavily depends on the specific hardware configuration. These are general estimates:

| Hardware Configuration        | Estimated RTF | Typical VRAM/RAM Usage | Notes                                       |
|-------------------------------|---------------|------------------------|---------------------------------------------|
| NVIDIA RTX 4090 (24GB VRAM)   | 0.01 - 0.03   | 2-4GB VRAM             | Blazing fast, S++ tier performance         |
| NVIDIA RTX 3080 (10GB VRAM)   | 0.03 - 0.06   | 3-5GB VRAM             | Exceptional, S+ tier performance           |
| NVIDIA RTX 2070 (8GB VRAM)    | 0.08 - 0.15   | 4-6GB VRAM             | Excellent, S to A+ tier                    |
| Apple M3 Max (Unified Memory) | 0.06 - 0.12   | 4-7GB (shared)         | Excellent MPS performance, S to A+ tier    |
| Apple M1 (Unified Memory)     | 0.12 - 0.25   | 5-8GB (shared)         | Good MPS performance, A+ to A tier         |
| Intel Core i9 (e.g., 12900K)  | 0.5 - 1.0     | 6-10GB RAM             | CPU-only, acceptable for non-real-time     |
| Intel Core i5 (e.g., 10400)   | 1.2 - 2.5     | 8-12GB RAM             | CPU-only, slow, primarily for testing      |

**RTF (Real-Time Factor) Interpretation:**
- **RTF < 0.1**: Significantly faster than real-time (e.g., 10x faster for RTF=0.1). Ideal for interactive applications
- **RTF ≈ 1.0**: Synthesizes speech at approximately real-time speed
- **RTF > 1.0**: Slower than real-time. May introduce noticeable delays

### Factors Influencing Performance
- **First Run vs. Subsequent Runs**: The initial inference includes overheads like model loading from disk, JIT compilation (if applicable), and cache warming. Subsequent inferences on cached models or warmed-up systems are much faster.
- **Text Length**: While RTF aims to normalize for audio duration, very short or very long texts can have slightly different characteristics due to fixed overheads or model sequence length limits.
- **Language Model**: Performance is generally similar between the Hindi and Gujarati models as they share the same architecture.
- **Caching Efficacy**: Repeatedly synthesizing the same text-language pair will result in near-instantaneous retrieval from cache.

## 🏗️ Architecture Overview

### Core Components

#### TTS Engine (`UltimateOptimizedTTS` class)
- **Model Management**: Handles loading, optimization (quantization, compilation), and device placement (CPU/GPU/MPS) of the VITS models
- **Inference Pipeline**: Orchestrates the end-to-end process from text input to audio waveform output, including tokenization, model forward pass, and post-processing
- **Device Abstraction**: Intelligently selects the optimal compute device based on availability and capabilities

#### Performance Tracker (`UltimatePerformanceTracker` class)
- **Metrics Collection**: Gathers a wide array of real-time performance data (timing, memory, GPU stats, etc.)
- **Analytics Engine**: Processes raw metrics to compute RTF, efficiency scores, and generate performance grades
- **Background Monitoring**: Continuously monitors system resources to provide context for inference performance

#### Caching System (`AdvancedCache` class)
- **Multi-Level Caching**: Implements an in-memory (LRU) cache for frequently accessed audio and a persistent disk cache for longer-term storage
- **Efficient Storage**: Uses techniques like memory-mapped files (experimentally) and optimized serialization (pickle) for disk cache
- **Asynchronous Writes**: Disk cache writes can be performed asynchronously to avoid blocking the main inference thread

#### System Information (`ComprehensiveSystemInfo` class)
- **Hardware Probing**: Detects and reports detailed information about CPU, GPU, memory, OS, and relevant software frameworks
- **Cross-Platform Compatibility**: Uses platform-agnostic libraries (e.g., `psutil`, `platform`) to gather system data
- **Resource Monitoring Utilities**: Provides helper functions to query current resource utilization

### Optimization Pipeline Stages

1. **Model Loading & Preparation**:
   - Efficient loading from HuggingFace Hub or local cache
   - Application of optimizations: mixed precision (FP16), dynamic quantization (INT8), JIT compilation (TorchScript, Inductor), TensorRT (experimental)
   - Transfer to the target compute device
2. **Text Processing (Tokenization)**:
   - Fast tokenization using HuggingFace `AutoTokenizer`
   - Caching of tokenized inputs for repeated texts
   - Batch tokenization support for potential future batch inference
3. **Model Inference**:
   - Execution of the VITS model's forward pass within a `torch.inference_mode()` context to disable gradient calculations
   - Leverages GPU acceleration (CUDA/MPS) with optimized kernels and memory management
   - AMP (Automatic Mixed Precision) context for CUDA
4. **Post-processing**:
   - Conversion of model output tensors to NumPy arrays
   - Reshaping and data type conversion as needed
   - Transfer of audio data from GPU to CPU memory if necessary
5. **Caching**:
   - Generated audio (sample rate + NumPy array) is stored in the `AdvancedCache`
   - Cache lookup precedes the inference pipeline for subsequent identical requests

## 🧪 Development

### Project Structure
```
DeepScaleR4/
├── app.py                    
├── main.py               
├── README.md        
├── requirements.txt                
```

### Development Setup

1. **Clone Repository**:
   ```bash
   git clone https://github.com/Adarsh-61/DeepScaleR4.git
   cd DeepScaleR4
   ```
2. **Create & Activate Virtual Environment**:
   ```bash
   uv venv
   source .venv/bin/activate  # macOS/Linux
   # .venv\Scripts\Activate.ps1 # Windows PowerShell
   ```
3. **Install Dependencies**:
   ```bash
   uv pip install -r requirements.txt
   ```
4. **Install Development Tools (Optional but Recommended)**:
   For code formatting, linting, and type checking.
   ```bash
   uv pip install black flake8 mypy pytest
   ```
5. **Run Application**:
   ```bash
   python app.py
   ```

### Contributing
Contributions are welcome! Please follow these general guidelines:
1. Fork the repository on GitHub
2. Create a new branch for your feature or bug fix (e.g., `feature/new-language` or `fix/oom-error`)
3. Make your changes, adhering to the existing code style
4. Write or update relevant tests if applicable
5. Ensure your changes are well-documented (code comments, README updates if necessary)
6. Test thoroughly on your local machine
7. Commit your changes with clear, descriptive commit messages
8. Push your branch to your fork
9. Submit a pull request to the main repository, detailing your changes

### Adding New Languages

To extend support for additional languages using compatible Facebook MMS-TTS models:

1. **Identify a Compatible VITS Model**: Search the HuggingFace Model Hub for `facebook/mms-tts-*` models for your target language (e.g., `facebook/mms-tts-ben` for Bengali).

2. **Update Model Configuration** (in `UltimateOptimizedTTS._load_ultimate_optimized_models` within `app.py`):
   Add the new language and its corresponding HuggingFace model identifier to the `models_config` dictionary.
   ```python
   # Inside _load_ultimate_optimized_models method
   models_config = {
       "Hindi": "facebook/mms-tts-hin",
       "Gujarati": "facebook/mms-tts-guj",
       "Bengali": "facebook/mms-tts-ben",  # Example: Adding Bengali
       # Add other languages here
   }
   ```

3. **Update Gradio Interface** (in `create_ultimate_interface` within `app.py`):
   Add the new language to the `choices` list of the `language_input` Gradio Radio component.
   ```python
   # Inside create_ultimate_interface function
   language_input = gr.Radio(
       choices=["Hindi", "Gujarati", "Bengali"],  # Add the new language display name
       label="Language",
       value="Hindi"  # Or your preferred default
   )
   ```
   Also, update example texts if desired.

4. **Update Warmup Texts** (in `UltimateOptimizedTTS._ultimate_warmup` within `app.py`):
   Add appropriate warmup texts for the new language to ensure proper model initialization.
   ```python
   # Inside _ultimate_warmup method, adapt the logic for new languages
   warmup_texts_map = {
       "Hindi": ["नमस्ते", ...],
       "Gujarati": ["નમસ્તે", ...],
       "Bengali": ["নমস্কার", ...], # Example for Bengali
   }
   warmup_texts = warmup_texts_map.get(language, ["Default warmup text"])
   ```

5. **Test Thoroughly**: Launch the application and test speech generation for the newly added language, checking for quality, performance, and any errors.

## 📋 System Requirements

### Minimum Requirements
- **Operating System**: Windows 10/11 (64-bit), macOS 10.15+ (Catalina or newer, ARM64 for Apple Silicon), or a modern Linux distribution (e.g., Ubuntu 18.04+, Fedora 32+)
- **Python Version**: 3.8 to 3.11 (3.11+ recommended)
- **RAM**: 8GB system RAM (16GB strongly recommended, especially for CPU mode or multitasking)
- **Storage**: ~5GB free disk space for Python environment, dependencies, models, and cache. SSD is highly recommended
- **Internet Connection**: Required for initial download of dependencies and TTS models

### Recommended Requirements for Optimal Performance
- **Operating System**: Windows 11 (64-bit), macOS 12+ (Monterey or newer), or a recent Linux distribution (e.g., Ubuntu 20.04+)
- **Python Version**: 3.11+
- **RAM**: 16GB+ system RAM (DDR4/DDR5)
- **GPU**:
  - **NVIDIA**: RTX 3060 or newer with at least 8GB VRAM. CUDA 12.6+ compatible drivers
  - **Apple Silicon**: M1 Pro/Max, M2 series, M3 series for good MPS performance
- **Storage**: 10GB+ free space on an NVMe SSD
- **Internet**: Stable broadband connection for initial setup

### Supported Hardware Architectures
- **NVIDIA GPUs**: Maxwell architecture (GTX 900 series) or newer. Ampere (RTX 30 series) or Ada Lovelace (RTX 40 series) for best results. Tesla and Quadro series supporting compatible CUDA versions are also viable
- **Apple Silicon SoCs**: M1, M2, M3, M4 series chips with integrated GPUs supporting Metal Performance Shaders
- **Intel/AMD CPUs**: x86-64 architecture with AVX2 instruction set support is beneficial for CPU-bound operations. Modern multi-core CPUs (e.g., Intel Core i7/i9, AMD Ryzen 7/9) are preferred for CPU fallback
- **Memory Types**: DDR4 or DDR5 system RAM. GDDR6/GDDR6X/HBM2e VRAM for discrete GPUs

## 🔒 Security and Privacy

### Data Handling and Processing
- **Local-First Operation**: All TTS synthesis, including text processing and audio generation, occurs entirely on the user's local machine
- **No External Data Transmission**: User-provided text and generated audio are not transmitted to any external servers or third parties by this application
- **Model Provenance**: TTS models are downloaded directly from HuggingFace Hub, a trusted repository for machine learning models. These downloads occur once and are then cached locally
- **Local Caching**: Generated audio files may be cached locally (in-memory and/or on disk) to speed up subsequent identical requests. This cache resides within the application's designated cache directory on the user's system
- **User Privacy**: The design prioritizes user privacy; input text is processed ephemerally for synthesis and is not stored beyond the optional local cache

### Security Considerations
- **Input Sanitization**: While Gradio provides some basic input handling, users should be aware that inputs are processed by complex neural network models. Avoid inputting sensitive personal information
- **Resource Management**: The application includes mechanisms to manage memory and processing resources (e.g., PyTorch's memory allocators, thread limits). However, like any computationally intensive software, it can place a significant load on system resources
- **Dependency Security**: Dependencies are managed via `requirements.txt`. It's good practice to source dependencies from reputable indices and keep them updated. `uv` can help in resolving dependencies securely
- **Execution Environment**: Running the application within a virtual environment is a key security best practice, isolating its dependencies and runtime from the system-wide Python installation
- **Network Exposure**: The Gradio web interface is, by default, served locally (`127.0.0.1`). If explicitly configured to be accessible over a network (e.g., `share=True` or binding to `0.0.0.0`), appropriate network security measures (firewall, access controls) should be considered

## 📄 License

This project is licensed under the MIT License. Please see the `LICENSE` file (if included in the repository, standard MIT terms apply) for full details.

## 🤝 Acknowledgments

This work heavily relies on the contributions of the broader AI and open-source communities:
- **Meta AI Research**: For developing and open-sourcing the MMS-TTS (Massively Multilingual Speech) models and the underlying VITS (Variational Inference with adversarial learning for End-to-end Text-to-Speech) architecture
- **HuggingFace**: For the `transformers` library, which provides seamless access to pre-trained models, and for hosting the MMS models on their Hub
- **PyTorch Team**: For the PyTorch deep learning framework, which powers the model inference and optimization capabilities
- **Gradio Team**: For the Gradio library, enabling the rapid creation of intuitive web interfaces for machine learning models
- **The Python Open Source Community**: For numerous libraries and tools (e.g., NumPy, SciPy, psutil) that are integral to this application

## 📞 Support

### Getting Assistance
1. **Consult Documentation**: This README file is the primary source of information. Please review it carefully
2. **Verify System Requirements**: Ensure your system configuration meets at least the minimum requirements
3. **Review Troubleshooting Section**: Common issues and their resolutions are documented above
4. **Check Hardware and Drivers**: Confirm that your GPU (if used) is compatible and that drivers are up-to-date

### Reporting Issues
If you encounter a bug or an issue not covered by the troubleshooting guide, please provide the following when reporting:
- **Operating System**: Version and architecture (e.g., Windows 11 Pro 22H2, macOS Sonoma 14.1, Ubuntu 22.04 LTS)
- **Python Version**: Output of `python --version`
- **Hardware Details**: CPU model, GPU model, total system RAM, VRAM
- **Error Messages**: Full, verbatim error messages and stack traces from the console
- **Steps to Reproduce**: A clear, step-by-step description of how to trigger the issue
- **Context**: Indicate if this is a first-time run, an issue after an update, or a recurring problem

### Performance-Related Inquiries
For questions or concerns about performance:
- **Detailed Hardware Specifications**: As above
- **Performance Analytics Output**: A screenshot or copy-paste of the analytics panel from the application
- **Observed vs. Expected RTF**: If you have a specific performance expectation, please state it
- **System Resource Usage**: Note CPU, GPU, and RAM utilization during inference (e.g., from Task Manager, `nvidia-smi`, Activity Monitor)

---

*Vani-TTS: Simple, Fast, and Efficient Text-to-Speech for Hindi and Gujarati, as part of the DeepScaleR4 project.*
