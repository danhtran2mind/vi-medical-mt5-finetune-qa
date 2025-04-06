import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import gradio as gr
import subprocess
import os

def run_shell_command(command):
    process = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True)
    output, error = process.communicate()
    if error:
        raise Exception(f"Error running command: {command}\n{error.decode('utf-8')}")
    return output.decode('utf-8')

def load_model_and_tokenizer(model_path):
    # Load the trained tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Load the trained model
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

    # Move the model to the GPU if available
    device = torch.device("cuda" if torch.cuda.is available() else "cpu")
    model.to(device)

    return tokenizer, model, device

def generate_text(tokenizer, model, device, prompt, max_length=100,
                  num_return_sequences=1, top_p=0.95, temperature=0.7):
    # Tokenize the input prompt
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)

    # Generate text
    output = model.generate(
        input_ids,
        max_length=max_length,
        num_return_sequences=num_return_sequences,
        no_repeat_ngram_size=2,
        top_k=50,
        top_p=top_p,
        temperature=temperature,
        do_sample=True
    )

    # Convert the generated text back to a string
    generated_text = [tokenizer.decode(ids, skip_special_tokens=True) for ids in output]

    return generated_text

def gradio_generate_text(prompt, max_length=100, num_return_sequences=1, top_p=0.95, temperature=0.7):
    generated_text = generate_text(tokenizer, model, device, prompt, max_length, num_return_sequences, top_p, temperature)
    return generated_text

# Ensure the models directory exists
if not os.path.exists('models'):
    os.makedirs('models')
if not os.path.exists('models/vi-medical-t5-finetune-qa'):
    # Run the Git LFS commands to clone the model
    run_shell_command('git lfs install')
    run_shell_command('cd models && git lfs clone https://huggingface.co/danhtran2mind/vi-medical-t5-finetune-qa && cd ..')

# Load the trained model and tokenizer
model_path = "models/vi-medical-t5-finetune-qa"
tokenizer, model, device = load_model_and_tokenizer(model_path)

# Create Gradio interface
iface = gr.Interface(
    fn=gradio_generate_text,
    inputs=[
        gr.inputs.Textbox(lines=5, label="Input Prompt"),
        gr.inputs.Slider(minimum=10, maximum=500, default=100, label="Max Length"),
        gr.inputs.Slider(minimum=1, maximum=10, default=1, label="Number of Sequences"),
        gr.inputs.Slider(minimum=0.1, maximum=1.0, default=0.95, label="Top-p Sampling"),
        gr.inputs.Slider(minimum=0.1, maximum=1.0, default=0.7, label="Temperature")
    ],
    outputs=gr.outputs.Textbox(label="Generated Text"),
    title="Vietnamese Medical T5 Fine-Tuned Model",
    description="Generate text using a fine-tuned Vietnamese medical T5 model."
)

# Launch the Gradio interface
iface.launch()
