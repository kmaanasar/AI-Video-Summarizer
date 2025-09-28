import streamlit as st
import whisper
import moviepy
import pytube
import transformers
import torch

print("✅ All packages imported successfully!")
print(f"Torch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")