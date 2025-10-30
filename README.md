# image-classifier
A simple web app that uses **MobileNetV2** to classify images with **TensorFlow**. Built with **Streamlit** for an easy-to-use interface.

## Features

- Upload an image (JPG or PNG) and get the top 3 predicted labels.
- Uses pre-trained MobileNetV2 model from Keras Applications.
- Provides a simple, interactive web interface with Streamlit.

## Installation

1. Clone the repository:

```bash
git clone https://github.com/e-mason03/image-classifier.git
cd image-classifier
```
2. Create a virtual environment and activate it (using uv or python -m venv)
```bash
uv venv
source .venv/bin/activate
```
3. Install dependencies
```bash
uv pip install -r requirements.txt
```
4. Run the app
```bash
streamlit run main.py
```
