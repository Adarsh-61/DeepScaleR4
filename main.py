import gradio as gr, torch, numpy as np
from transformers import VitsModel, AutoTokenizer
from functools import lru_cache

MODELS_CONFIG = {"Hindi": "facebook/mms-tts-hin", "Gujarati": "facebook/mms-tts-guj"}
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if hasattr(torch.backends,"mps") and torch.backends.mps.is_available() else "cpu"

@lru_cache(maxsize=2)
def load_tts_components(language):
    model_name = MODELS_CONFIG[language]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = VitsModel.from_pretrained(model_name).to(DEVICE)
    print(f"Loaded {language} model ('{model_name}') on {DEVICE}.")
    return model, tokenizer

def generate_tts_audio(text_input, selected_language):
    if not text_input or not text_input.strip():
        gr.Info("Input text is empty. Please provide some text to synthesize.")
        return (16000, np.zeros(100, dtype=np.float32))
    try:
        model, tokenizer = load_tts_components(selected_language)
        inputs = tokenizer(text_input, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            output_waveform = model(**inputs).waveform.cpu().squeeze().numpy().astype(np.float32)
        return (model.config.sampling_rate, output_waveform)
    except Exception as e:
        error_detail = f"TTS Error for {selected_language} ('{text_input[:20]}...'): {e}"
        print(error_detail)
        gr.Error(f"Failed to generate audio. Ensure Python dependencies (e.g., torch, transformers, gradio) are installed. Error: {e}")
        return (16000, np.zeros(100, dtype=np.float32)) 

if __name__ == "__main__":
    gr.Interface(
        fn=generate_tts_audio,
        inputs=[gr.Textbox(label="Enter Text", lines=3, placeholder="Type your text here..."),
                gr.Radio(choices=list(MODELS_CONFIG.keys()), label="Select Language", value="Hindi")],
        outputs=gr.Audio(label="Generated Speech Output", type="numpy"),
        title="Vani-TTS Lite",
        description="Simple. Fast. Efficient.",
        examples=[["नमस्ते, आज का मौसम बहुत अच्छा है।", "Hindi"],
                  ["નમસ્તે, આજે હવામાન ખૂબ સરસ છે.", "Gujarati"]],
        allow_flagging='never'
    ).launch()