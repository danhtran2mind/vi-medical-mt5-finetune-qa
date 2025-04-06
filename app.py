import torch
import sentencepiece
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import gradio as gr
import subprocess
import os
import string
import re

def run_shell_command(command):
    process = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True)
    output, error = process.communicate()
    if error:
        raise Exception(f"Error running command: {command}\n{error.decode('utf-8')}")
    return output.decode('utf-8')

def load_model_and_tokenizer(model_path):
    # Load the trained tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, add_prefix_space=True)

    # Load the trained model
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

    # Move the model to the GPU if available
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        
    model.to(device)

    return tokenizer, model, device

def string_processing(text, exclude=None):
    if exclude is None:
        exclude = "-"
    
    # Define a regular expression pattern to match punctuation marks, excluding the specified ones
    punctuation_to_handle = ''.join([p for p in string.punctuation if p not in exclude])
    pattern = r'([{}])'.format(re.escape(punctuation_to_handle))
    
    # Use re.sub to replace each punctuation mark with a space before and after it
    text_with_spaces = re.sub(pattern, r' \1 ', text)
    
    # Remove any extra spaces that might have been added
    text_with_spaces = re.sub(r'\s{2,}', ' ', text_with_spaces).strip()
    
    return text_with_spaces

def generate_text(tokenizer, model, device, prompt, max_length=100,
                  num_return_sequences=1, top_p=0.95, temperature=0.7):

    prompt = string_processing(prompt)
    
    # Tokenize the input prompt
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)

    # Generate text
    output = model.generate(
        input_ids,
        max_length=max_length,
        num_return_sequences=num_return_sequences,
        top_k=50,
        top_p=top_p,
        temperature=temperature,
    )

    # Convert the generated text back to a string
    generated_text = [tokenizer.decode(ids, skip_special_tokens=True) for ids in output]

    return generated_text[0].replace("_", " ").replace(" ,", ",").replace(" .", ".")

def gradio_generate_text(prompt, max_length=100, num_return_sequences=1, top_p=0.95, temperature=0.7):
    generated_text = generate_text(tokenizer, model, device, prompt, max_length, num_return_sequences, top_p, temperature)
    return generated_text

# def gradio_generate_text(prompt, max_length, num_sequences, top_p, temperature):
#     # Placeholder for your text generation logic
#     return f"Generated text based on: {prompt}"

# Ensure the models directory exists
if not os.path.exists('models'):
    os.makedirs('models')
if not os.path.exists('models/vi-medical-t5-finetune-qa'):
    # Run the Git LFS commands to clone the model
    run_shell_command('git lfs install')
    run_shell_command('cd models && git clone https://huggingface.co/danhtran2mind/vi-medical-t5-finetune-qa && cd ..')

# Load the trained model and tokenizer
model_path = "models/vi-medical-t5-finetune-qa"
tokenizer, model, device = load_model_and_tokenizer(model_path)
# Create Gradio interface


# Create the Gradio interface
iface = gr.Interface(
    fn=gradio_generate_text,
    inputs=[
        gr.Textbox(lines=3, label="Input Prompt"),
        gr.Slider(minimum=10, maximum=768, value=32, label="Max Length"),
        # gr.Slider(minimum=1, maximum=5, value=1, label="Number of Sequences"),
        gr.Slider(minimum=0.1, maximum=1.0, value=0.95, label="Top-p Sampling"),
        gr.Slider(minimum=0.1, maximum=1.0, value=0.7, label="Temperature")
    ],
    outputs=gr.Textbox(lines=5, label="Generated Text"),
    title="Vietnamese Medical T5 Fine-Tuned Model",
    description="Generate text using a fine-tuned Vietnamese medical T5 model."
)

# Launch the Gradio interface
iface.launch()
