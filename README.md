<div align="center">
  <h1>
    Vietnamese Medical mT5 Finetune Question and Answer 👨🏻‍⚕️
  </h1>
</div>

> **⚠️** Use this model only for research. Seek qualified medical advice for health concerns.

## Introduction

This repository provides a **Vietnamese medical question‑answering system** built by fine‑tuning the **mT5** model on the **UIT‑ViCoV19QA** dataset. This model achieves low loss on both training (≈ 0.306) and validation (≈ 0.323) sets, indicating strong performance for Vietnamese‑language medical queries.

## Key Features

- **Dataset**: Downloadable from the linked GitHub repository.  
- **Demo**: An interactive Gradio interface is available both as a screenshot and a live Hugging Face Space.  
- **Usage**: Simple installation steps, model download commands, and a one‑line command to launch the app locally (`python app.py`).  
- **Environment**: Tested with Python 3.10.12 and a specific set of library versions (e.g., `transformers==4.47.0`, `torch==2.5.1+cu121`).

The project is ready for researchers and developers who need a Vietnamese‑language medical QA system that can be deployed locally or integrated into larger applications.
## Dataset
You can download dataset at this url: https://github.com/triet2397/UIT-ViCoV19QA
## Metrics
- Loss:
  - Trainging Set: 0.306100.
  - Validation Set: 0.322764.

## Demo
### Demo Image
![Demo Image](https://github.com/danhtran2mind/vi-medical-mt5-finetune-qa/blob/main/demo_images/demo.png)
### Demo Space
You can try this project demo at: https://huggingface.co/spaces/danhtran2mind/vi-medical-mt5-finetune-qa

## Usage
- Install Denpendencies:
```bash
pip install -r requirements.txt
```
- Download download the 'danhtran2mind/vi-medical-mt5-finetune-qa' model from Hugging Face using the following commands:
```bash
cd models
git clone https://huggingface.co/danhtran2mind/vi-medical-mt5-finetune-qa
cd ..
```
- Run Gradio app:
```bash
python app.py
```
- Your app will run at `localhost:7860`

## Denpendencies Enviroment
- Python version: 3.10.12
- Libraries verions:
  ```bash
  pandas==2.2.3
  numpy==1.26.4
  matplotlib==3.7.5
  scikit-learn==1.2.2
  gensim==4.3.3
  underthesea==6.8.4
  tensorflow==2.17.1
  datasets==3.3.1
  torch==2.5.1+cu121
  transformers==4.47.0
  ```
