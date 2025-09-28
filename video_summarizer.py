import streamlit as st
import os
import tempfile
import json
from datetime import datetime, timedelta
import re
from typing import List, Dict, Optional
import requests
from pathlib import Path

# Core libraries
import whisper
import moviepy.editor as mp
import yt_dlp
import openai
from transformers import pipeline, T5ForConditionalGeneration, T5Tokenizer
import torch

# PDF generation
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch

class VideoSummarizer:
    def __init__(self):
        self.whisper_model = None
        self.summarizer_model = None
        self.setup_models()
    
    def setup_models(self):
        """Initialize AI models"""
        try:
            # Load Whisper model (using base for balance of speed/accuracy)
            st.info("Loading Whisper model...")
            self.whisper_model = whisper.load_model("tiny")
            
            # Setup summarization model
            st.info("Loading summarization model...")
            if st.session_state.get('use_openai', False) and st.session_state.get('openai_api_key'):
                openai.api_key = st.session_state.openai_api_key
            else:
                # Use local T5 model for summarization
                device = "cuda" if torch.cuda.is_available() else "cpu"
                model_name = "t5-small"
                self.summarizer_model = pipeline(
                    "summarization",
                    model=model_name,
                    tokenizer=model_name,
                    device=0 if device == "cuda" else -1,
                    max_length=512,
                    min_length=50
                )
            
            st.success("Models loaded successfully!")
        except Exception as e:
            st.error(f"Error loading models: {str(e)}")
    
    def download_youtube_audio(self, url: str) -> str:
        """Download audio from YouTube video using yt-dlp"""
        try:
            with st.spinner("Downloading YouTube video..."):
                # Create temp directory
                temp_dir = tempfile.mkdtemp()
                audio_file = os.path.join(temp_dir, "audio")
                
                # yt-dlp options
                ydl_opts = {
                    'format': 'bestaudio/best',
                    'outtmpl': audio_file + '.%(ext)s',
                    'postprocessors': [{
                        'key': 'FFmpegExtractAudio',
                        'preferredcodec': 'wav',
                    }],
                    'quiet': True,
                    'no_warnings': True,
                }
                
                # Download with yt-dlp
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    # Get video info first
                    try:
                        info = ydl.extract_info(url, download=False)
                        st.write(f"**Title:** {info.get('title', 'Unknown')}")
                        st.write(f"**Duration:** {str(timedelta(seconds=info.get('duration', 0)))}")
                        st.write(f"**Uploader:** {info.get('uploader', 'Unknown')}")
                    except:
                        st.write("**Processing video...** (info extraction failed but continuing)")
                    
                    # Download the audio
                    ydl.download([url])
                
                # Find the downloaded file
                for file in os.listdir(temp_dir):
                    if file.startswith("audio") and (file.endswith('.wav') or file.endswith('.mp3') or file.endswith('.m4a')):
                        return os.path.join(temp_dir, file)
                
                # If no specific audio file found, return any file that was downloaded
                files = [f for f in os.listdir(temp_dir) if not f.startswith('.')]
                if files:
                    return os.path.join(temp_dir, files[0])
                
                return None
                
        except Exception as e:
            st.error(f"Error downloading YouTube video: {str(e)}")
            st.error("Try using a different YouTube URL or upload a video file instead.")
            return None
    
    def extract_audio_from_video(self, video_path: str) -> str:
        """Extract audio from uploaded video file"""
        try:
            with st.spinner("Extracting audio from video..."):
                video = mp.VideoFileClip(video_path)
                
                # Create temp audio file
                temp_dir = tempfile.mkdtemp()
                audio_file = os.path.join(temp_dir, "extracted_audio.wav")
                
                # Extract audio
                video.audio.write_audiofile(audio_file, verbose=False, logger=None)
                video.close()
                
                return audio_file
        except Exception as e:
            st.error(f"Error extracting audio: {str(e)}")
            return None
    
    def transcribe_audio(self, audio_file: str) -> Dict:
        """Transcribe audio using Whisper"""
        try:
            with st.spinner("Transcribing audio... This may take a few minutes."):
                result = self.whisper_model.transcribe(
                    audio_file,
                    task="transcribe",
                    language=st.session_state.get('target_language', 'en') if st.session_state.get('target_language') != 'auto' else None
                )
                
                return result
        except Exception as e:
            st.error(f"Error transcribing audio: {str(e)}")
            return None
    
    def chunk_transcript(self, transcript: Dict, chunk_size: int = 500) -> List[Dict]:
        """Split transcript into chunks with timestamps"""
        segments = transcript.get('segments', [])
        chunks = []
        current_chunk = ""
        current_start = 0
        current_end = 0
        word_count = 0
        
        for segment in segments:
            segment_text = segment['text'].strip()
            segment_words = len(segment_text.split())
            
            if word_count == 0:
                current_start = segment['start']
            
            if word_count + segment_words > chunk_size and current_chunk:
                # Save current chunk
                chunks.append({
                    'text': current_chunk.strip(),
                    'start_time': current_start,
                    'end_time': current_end,
                    'timestamp': self.format_timestamp(current_start)
                })
                
                # Start new chunk
                current_chunk = segment_text
                current_start = segment['start']
                word_count = segment_words
            else:
                current_chunk += " " + segment_text
                word_count += segment_words
            
            current_end = segment['end']
        
        # Add final chunk
        if current_chunk:
            chunks.append({
                'text': current_chunk.strip(),
                'start_time': current_start,
                'end_time': current_end,
                'timestamp': self.format_timestamp(current_start)
            })
        
        return chunks
    
    def format_timestamp(self, seconds: float) -> str:
        """Convert seconds to MM:SS format"""
        minutes = int(seconds // 60)
        seconds = int(seconds % 60)
        return f"{minutes:02d}:{seconds:02d}"
    
    def summarize_with_openai(self, text: str, style: str) -> str:
        """Summarize text using OpenAI GPT"""
        try:
            style_prompts = {
                "bullet_points": "Create a bullet-point summary of the following text. Focus on key points and main ideas:",
                "paragraph": "Create a concise paragraph summary of the following text:",
                "study_notes": "Create detailed study notes from the following text, organized with headings and key concepts:",
                "meeting_recap": "Create a meeting-style recap of the following content with action items and key decisions:"
            }
            
            prompt = style_prompts.get(style, style_prompts["bullet_points"])
            
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that creates clear, concise summaries."},
                    {"role": "user", "content": f"{prompt}\n\n{text}"}
                ],
                max_tokens=500,
                temperature=0.3
            )
            
            return response.choices[0].message.content.strip()
        except Exception as e:
            st.error(f"Error with OpenAI summarization: {str(e)}")
            return None
    
    def summarize_with_local_model(self, text: str, style: str) -> str:
        """Summarize text using local T5 model"""
        try:
            # Truncate text if too long
            max_input_length = 512
            if len(text.split()) > max_input_length:
                text = " ".join(text.split()[:max_input_length])
            
            # Add style-specific prefix
            style_prefixes = {
                "bullet_points": "summarize with bullet points: ",
                "paragraph": "summarize: ",
                "study_notes": "create study notes: ",
                "meeting_recap": "create meeting recap: "
            }
            
            prefix = style_prefixes.get(style, "summarize: ")
            input_text = prefix + text
            
            summary = self.summarizer_model(input_text, max_length=150, min_length=30, do_sample=False)
            return summary[0]['summary_text']
        except Exception as e:
            st.error(f"Error with local summarization: {str(e)}")
            return text[:200] + "..."  # Fallback to truncated text
    
    def summarize_chunks(self, chunks: List[Dict], style: str) -> List[Dict]:
        """Summarize each chunk of text"""
        summarized_chunks = []
        
        progress_bar = st.progress(0)
        total_chunks = len(chunks)
        
        for i, chunk in enumerate(chunks):
            with st.spinner(f"Summarizing chunk {i+1}/{total_chunks}..."):
                if st.session_state.get('use_openai', False) and st.session_state.get('openai_api_key'):
                    summary = self.summarize_with_openai(chunk['text'], style)
                else:
                    summary = self.summarize_with_local_model(chunk['text'], style)
                
                summarized_chunks.append({
                    **chunk,
                    'summary': summary or "Summary not available"
                })
            
            progress_bar.progress((i + 1) / total_chunks)
        
        return summarized_chunks
    
    def generate_final_summary(self, summarized_chunks: List[Dict], style: str) -> str:
        """Generate final consolidated summary"""
        all_summaries = " ".join([chunk['summary'] for chunk in summarized_chunks])
        
        if st.session_state.get('use_openai', False) and st.session_state.get('openai_api_key'):
            return self.summarize_with_openai(all_summaries, style)
        else:
            return self.summarize_with_local_model(all_summaries, style)
    
    def export_to_pdf(self, content: Dict, filename: str):
        """Export summary to PDF"""
        try:
            doc = SimpleDocTemplate(filename, pagesize=letter)
            styles = getSampleStyleSheet()
            story = []
            
            # Title
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=16,
                spaceAfter=30,
            )
            story.append(Paragraph("Video Summary Report", title_style))
            story.append(Spacer(1, 12))
            
            # Video info
            if 'video_info' in content:
                story.append(Paragraph(f"<b>Title:</b> {content['video_info'].get('title', 'N/A')}", styles['Normal']))
                story.append(Paragraph(f"<b>Duration:</b> {content['video_info'].get('duration', 'N/A')}", styles['Normal']))
                story.append(Spacer(1, 12))
            
            # Overall summary
            story.append(Paragraph("Overall Summary", styles['Heading2']))
            story.append(Paragraph(content.get('final_summary', ''), styles['Normal']))
            story.append(Spacer(1, 12))
            
            # Timestamped summaries
            story.append(Paragraph("Timestamped Key Points", styles['Heading2']))
            for chunk in content.get('chunks', []):
                story.append(Paragraph(f"<b>{chunk['timestamp']}</b>", styles['Heading3']))
                story.append(Paragraph(chunk['summary'], styles['Normal']))
                story.append(Spacer(1, 8))
            
            doc.build(story)
            return True
        except Exception as e:
            st.error(f"Error creating PDF: {str(e)}")
            return False

def main():
    st.set_page_config(
        page_title="AI Video Summarizer",
        page_icon="🎥",
        layout="wide"
    )
    
    st.title("🎥 AI Video Summarizer")
    st.markdown("Transform any video into structured notes and summaries")
    
    # Initialize session state
    if 'summarizer' not in st.session_state:
        st.session_state.summarizer = None
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # API Configuration
        use_openai = st.checkbox("Use OpenAI GPT (Better Quality)", value=False)
        st.session_state.use_openai = use_openai
        
        if use_openai:
            api_key = st.text_input("OpenAI API Key", type="password")
            st.session_state.openai_api_key = api_key
            if not api_key:
                st.warning("OpenAI API key required for GPT summaries")
        
        # Summary style
        summary_style = st.selectbox(
            "Summary Style",
            ["bullet_points", "paragraph", "study_notes", "meeting_recap"],
            format_func=lambda x: {
                "bullet_points": "📝 Bullet Points",
                "paragraph": "📄 Paragraph",
                "study_notes": "📚 Study Notes",
                "meeting_recap": "🤝 Meeting Recap"
            }[x]
        )
        
        # Language options
        target_language = st.selectbox(
            "Transcription Language",
            ["auto", "en", "es", "fr", "de", "it", "pt", "ru", "ja", "ko", "zh"],
            format_func=lambda x: {
                "auto": "🌐 Auto-detect",
                "en": "🇺🇸 English",
                "es": "🇪🇸 Spanish",
                "fr": "🇫🇷 French",
                "de": "🇩🇪 German",
                "it": "🇮🇹 Italian",
                "pt": "🇵🇹 Portuguese",
                "ru": "🇷🇺 Russian",
                "ja": "🇯🇵 Japanese",
                "ko": "🇰🇷 Korean",
                "zh": "🇨🇳 Chinese"
            }[x]
        )
        st.session_state.target_language = target_language
        
        # Chunk size
        chunk_size = st.slider("Words per chunk", 300, 1000, 500)
        
        st.markdown("---")
        st.markdown("**💡 Tips:**")
        st.markdown("- Longer videos may take several minutes")
        st.markdown("- YouTube URLs work best with public videos")
        st.markdown("- Supported formats: MP4, AVI, MOV, MKV")
        st.markdown("- First run downloads AI models (be patient!)")
    
    # Main interface
    tab1, tab2 = st.tabs(["📥 Process Video", "📊 Results"])
    
    with tab1:
        st.header("Input Video")
        
        input_method = st.radio(
            "Choose input method:",
            ["YouTube URL", "Upload Video File"]
        )
        
        video_file = None
        youtube_url = None
        
        if input_method == "YouTube URL":
            youtube_url = st.text_input(
                "Enter YouTube URL:",
                placeholder="https://www.youtube.com/watch?v=..."
            )
            st.info("💡 **Test these URLs if you need examples:**")
            st.code("https://www.youtube.com/watch?v=jNQXAC9IVRw")
            st.code("https://www.youtube.com/watch?v=dQw4w9WgXcQ") 
        else:
            video_file = st.file_uploader(
                "Upload video file:",
                type=['mp4', 'avi', 'mov', 'mkv', 'webm']
            )
        
        if st.button("🚀 Start Processing", type="primary"):
            if not youtube_url and not video_file:
                st.error("Please provide a YouTube URL or upload a video file")
                return
            
            # Initialize summarizer
            if st.session_state.summarizer is None:
                st.session_state.summarizer = VideoSummarizer()
            
            summarizer = st.session_state.summarizer
            
            try:
                # Step 1: Get audio file
                audio_file = None
                
                if youtube_url:
                    audio_file = summarizer.download_youtube_audio(youtube_url)
                    video_info = {
                        'source': 'YouTube',
                        'url': youtube_url
                    }
                else:
                    # Save uploaded file temporarily
                    temp_dir = tempfile.mkdtemp()
                    video_path = os.path.join(temp_dir, video_file.name)
                    with open(video_path, 'wb') as f:
                        f.write(video_file.read())
                    
                    audio_file = summarizer.extract_audio_from_video(video_path)
                    video_info = {
                        'source': 'Upload',
                        'filename': video_file.name
                    }
                
                if not audio_file:
                    st.error("Failed to process audio. Try a different video or check the URL.")
                    return
                
                # Step 2: Transcribe audio
                transcript = summarizer.transcribe_audio(audio_file)
                if not transcript:
                    st.error("Failed to transcribe audio")
                    return
                
                st.success(f"Transcription complete! Detected language: {transcript.get('language', 'Unknown')}")
                
                # Step 3: Chunk transcript
                chunks = summarizer.chunk_transcript(transcript, chunk_size)
                st.success(f"Text split into {len(chunks)} chunks")
                
                # Step 4: Summarize chunks
                st.header("🔄 Generating Summaries...")
                summarized_chunks = summarizer.summarize_chunks(chunks, summary_style)
                
                # Step 5: Generate final summary
                final_summary = summarizer.generate_final_summary(summarized_chunks, summary_style)
                
                # Store results in session state
                st.session_state.results = {
                    'video_info': video_info,
                    'transcript': transcript,
                    'chunks': summarized_chunks,
                    'final_summary': final_summary,
                    'style': summary_style
                }
                
                st.success("✅ Processing complete! Check the Results tab.")
                
                # Clean up temp files
                try:
                    if audio_file and os.path.exists(audio_file):
                        os.remove(audio_file)
                except:
                    pass
                
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                st.error("Try using a different video or check your internet connection.")
    
    with tab2:
        if 'results' not in st.session_state:
            st.info("👈 Process a video first to see results here")
            return
        
        results = st.session_state.results
        
        st.header("📋 Summary Results")
        
        # Video info
        col1, col2 = st.columns([2, 1])
        
        with col1:
            if results['video_info']['source'] == 'YouTube':
                st.write(f"**Source:** YouTube - {results['video_info']['url']}")
            else:
                st.write(f"**Source:** Uploaded file - {results['video_info']['filename']}")
        
        with col2:
            # Export options
            if st.button("📄 Export to PDF"):
                pdf_filename = f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
                if st.session_state.summarizer.export_to_pdf(results, pdf_filename):
                    with open(pdf_filename, 'rb') as f:
                        st.download_button(
                            "⬇️ Download PDF",
                            f.read(),
                            pdf_filename,
                            "application/pdf"
                        )
        
        # Overall summary
        st.subheader("🎯 Overall Summary")
        st.write(results['final_summary'])
        
        # Timestamped summaries
        st.subheader("⏰ Timestamped Key Points")
        
        for i, chunk in enumerate(results['chunks']):
            with st.expander(f"🕐 {chunk['timestamp']} - Segment {i+1}"):
                st.write("**Summary:**")
                st.write(chunk['summary'])
                
                st.write("**Original Text:**")
                st.text_area(
                    f"Full transcript segment {i+1}",
                    chunk['text'],
                    height=100,
                    key=f"transcript_{i}"
                )
        
        # Full transcript
        with st.expander("📝 Complete Transcript"):
            full_text = results['transcript']['text']
            st.text_area("Full Transcript", full_text, height=300)
            
            # Download transcript
            st.download_button(
                "⬇️ Download Transcript",
                full_text,
                f"transcript_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                "text/plain"
            )

if __name__ == "__main__":
    main()