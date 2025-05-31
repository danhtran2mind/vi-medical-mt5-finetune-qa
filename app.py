import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import gradio as gr
import subprocess
import os
from underthesea import word_tokenize

def run_shell_command(command):
    try:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
        output, error = process.communicate()
        if process.returncode != 0:
            raise Exception(f"Error running command: {command}\n{error.decode('utf-8')}")
        return output.decode('utf-8')
    except Exception as e:
        raise Exception(f"Failed to execute command: {command}\n{str(e)}")

def load_model_and_tokenizer(model_path):
    try:
        # Load the trained tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        # Load the trained model
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        # Move the model to the GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        return tokenizer, model, device
    except Exception as e:
        raise Exception(f"Failed to load model or tokenizer from {model_path}: {str(e)}")

def generate_text(tokenizer, model, device, prompt, max_length=100,
                  num_return_sequences=1, top_p=0.95, temperature=0.7, seed=123):
    # Set the random seed for reproducibility
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU setups

    # Tokenize the input prompt with word segmentation
    prompt = word_tokenize(prompt, format='text')
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)

    # Generate text
    output = model.generate(
        input_ids,
        max_length=int(max_length),
        num_return_sequences=int(num_return_sequences),
        no_repeat_ngram_size=2,
        top_k=50,
        top_p=top_p,
        temperature=temperature,
        do_sample=True
    )

    # Convert the generated text back to a string
    generated_text = [tokenizer.decode(ids, skip_special_tokens=True).replace("_", " ").replace(" ,", ",").replace(" .", ".") for ids in output]
    return "\n\n".join(generated_text)  # Join multiple sequences with newlines

def gradio_generate_text(prompt, max_length, top_p, temperature, seed, num_return_sequences):
    try:
        # Load model and tokenizer
        model_path = "models/vi-medical-mt5-finetune-qa"
        tokenizer, model, device = load_model_and_tokenizer(model_path)
        # Generate text
        result = generate_text(tokenizer, model, device, prompt, max_length, num_return_sequences, top_p, temperature, seed)
        return result
    except Exception as e:
        return f"Error: {str(e)}"

# Ensure the models directory exists and clone the model if needed
if not os.path.exists('models'):
    os.makedirs('models')
if not os.path.exists('models/vi-medical-mt5-finetune-qa'):
    try:
        run_shell_command('git lfs install')
        run_shell_command('cd models && git clone https://huggingface.co/danhtran2mind/vi-medical-mt5-finetune-qa && cd ..')
    except Exception as e:
        print(f"Failed to clone model: {str(e)}")

# Define the Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# Vietnamese Medical mT5 Fine-Tune Question and Answer")
    with gr.Row():
        with gr.Column():
            prompt = gr.Textbox(lines=3, label="Input Prompt", placeholder="Enter your prompt, e.g., 'vaccine covid-19 là gì?'")
            max_length = gr.Slider(minimum=10, maximum=768, value=32, label="Max Length", step=1)
            top_p = gr.Slider(minimum=0.1, maximum=1.0, value=0.95, label="Top-p Sampling", step=0.01)
            temperature = gr.Slider(minimum=0.1, maximum=1.0, value=0.7, label="Temperature", step=0.01)
            seed = gr.Slider(minimum=0, maximum=10000, value=123, label="Seed", step=1)
            num_return_sequences = gr.Slider(minimum=1, maximum=5, value=1, label="Number of Sequences", step=1)
            submit_button = gr.Button("Generate")
        with gr.Column():
            output = gr.Textbox(label="Generated Text", lines=10)
    
    submit_button.click(
        fn=gradio_generate_text,
        inputs=[prompt, max_length, top_p, temperature, seed, num_return_sequences],
        outputs=output
    )
demo.launch()
