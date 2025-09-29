🎥 AI Video Summarizer

An AI-powered Streamlit app that transcribes and summarizes videos using local machine learning models.
Supports YouTube links and uploaded video files, with options to export results as PDF or plain text.

✨ Features:
📺 YouTube & Local Uploads – Summarize videos from YouTube URLs or uploaded files.
🎙 Speech-to-Text with Whisper – Local transcription in multiple languages.
📝 Flexible Summaries – Choose between key points, paragraphs, or detailed outputs.
⏱ Timestamps & Chunks – Breaks transcripts into manageable chunks with summaries per section.
📄 Export Options – Download summaries as PDF reports or raw transcripts as text.
🚀 Runs Locally – No reliance on paid APIs; everything uses open-source models.

🛠 Tech Stack:
Streamlit
 – Web UI

OpenAI Whisper
 – Speech-to-text transcription

Hugging Face Transformers
 – Local summarization with pretrained models (BART, T5)

yt-dlp
 – YouTube video/audio downloader

MoviePy
 – Video/audio extraction

ReportLab
 – PDF export

📦 Installation
# Clone repo
git clone https://github.com/your-username/ai-video-summarizer.git
cd ai-video-summarizer

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt


Dependencies include:
streamlit torch transformers openai-whisper moviepy yt-dlp reportlab

🚀 Usage
streamlit run app.py
Load models via the sidebar (choose Whisper size + summarizer).
Select YouTube URL or Upload Video.
Click Start Processing to transcribe & summarize.
View results in the Results tab:
Executive summary
Timestamped breakdowns
Full transcript
Export to PDF or TXT

📂 Project Structure
├── app.py                 # Main Streamlit app
├── requirements.txt       # Dependencies
├── README.md              # Documentation

⚖️ License
MIT License. Free to use and modify.