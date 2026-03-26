# Image Classifier

A Streamlit app that lets users upload an image and classify what is in it using a pretrained PyTorch `MobileNetV3-Large` model.

## Features

- Upload `jpg`, `jpeg`, `png`, or `webp` images
- Preview the uploaded image before inference
- Show the top predicted ImageNet classes with confidence scores
- Use a cached pretrained model for faster repeat predictions
- Deploy easily to Streamlit Community Cloud or with Docker

## Project structure

```text
.
|-- app.py
|-- Dockerfile
|-- README.md
|-- requirements.txt
`-- src
    `-- image_classifier
        |-- classifier.py
        `-- ui.py
```

## Local setup

1. Create and activate a virtual environment:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Start the app:

```powershell
streamlit run app.py
```

4. Open the local URL shown in the terminal, usually `http://localhost:8501`.

## Deployment option 1: Streamlit Community Cloud

1. Push this project to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io/).
3. Create a new app and point it to your repository.
4. Set the main file path to `app.py`.
5. Deploy. The platform will install `requirements.txt` automatically.

## Deployment option 2: Docker

Build and run locally:

```powershell
docker build -t image-classifier .
docker run -p 8501:8501 image-classifier
```

For Render, Railway, Fly.io, or any other container host, point the service to this `Dockerfile`.

## Notes

- The first run downloads pretrained model weights from PyTorch.
- The model predicts ImageNet labels, so outputs are limited to the ImageNet class list.
- For production use with your own categories, the next step would be fine-tuning on a custom dataset.
