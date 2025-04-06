# Vietnames Medical T5 Finetune Question and Answer
## Dataset
You can download dataset at this url: https://github.com/triet2397/UIT-ViCoV19QA
## Metrics
- Loss:
  - Trainging Set: 0.306100.
  - Validation Set: 0.322764.

## Demo
You can try this project demo at: https://huggingface.co/spaces/danhtran2mind/vi-medical-t5-finetune-qa

## Usage
- Install Denpendencies:
```bash
pip install -r requirements.txt
```
- Download download the 'danhtran2mind/vi-medical-t5-finetune-qa' model from Hugging Face using the following commands:
```bash
cd models
git clone https://huggingface.co/danhtran2mind/vi-medical-t5-finetune-qa
cd ..
```
- Run Gradio app:
```bash
python app.py
```
- Your app will run at `localhost:7860`
