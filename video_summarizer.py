import streamlit as st
import os
import tempfile
import json
from datetime import datetime, timedelta
import re
from typing import List, Dict, Optional
import requests
from pathlib import Path
import time

# Core libraries
import moviepy.editor as mp
import yt_dlp
import torch

# Hugging Face
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from huggingface_hub import InferenceClient

# PDF generation
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch

class HuggingFaceVideoSummarizer:
    def __init__(self, hf_token: str):
        self.hf_token = hf_token
        self.inference_client = InferenceClient(token=hf_token)
        
        # Use the best models by default
        self.transcription_model = "openai/whisper-large-v3"
        self.summarization_model = "facebook/bart-large-cnn"
        
        # Load local summarization pipeline as backup
        try:
            st.info("Loading backup summarization model...")
            self.local_summarizer = pipeline(
                "summarization",
                model="facebook/bart-base",
                device=0 if torch.cuda.is_available() else -1
            )
            st.success("Backup model ready!")
        except Exception as e:
            st.warning(f"Backup model failed to load: {e}")
            self.local_summarizer = None
    
    def download_youtube_audio(self, url: str) -> str:
        """Download audio from YouTube video using yt-dlp"""
        try:
            with st.spinner("Downloading YouTube video..."):
                temp_dir = tempfile.mkdtemp()
                audio_file = os.path.join(temp_dir, "audio")
                
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
                
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    try:
                        info = ydl.extract_info(url, download=False)
                        st.write(f"**Title:** {info.get('title', 'Unknown')}")
                        st.write(f"**Duration:** {str(timedelta(seconds=info.get('duration', 0)))}")
                        st.write(f"**Uploader:** {info.get('uploader', 'Unknown')}")
                    except:
                        st.write("**Processing video...**")
                    
                    ydl.download([url])
                
                # Find downloaded file
                for file in os.listdir(temp_dir):
                    if file.startswith("audio"):
                        return os.path.join(temp_dir, file)
                
                return None
                
        except Exception as e:
            st.error(f"Download error: {str(e)}")
            return None
    
    def extract_audio_from_video(self, video_path: str) -> str:
        """Extract audio from uploaded video file"""
        try:
            with st.spinner("Extracting audio from video..."):
                video = mp.VideoFileClip(video_path)
                temp_dir = tempfile.mkdtemp()
                audio_file = os.path.join(temp_dir, "extracted_audio.wav")
                video.audio.write_audiofile(audio_file, verbose=False, logger=None)
                video.close()
                return audio_file
        except Exception as e:
            st.error(f"Audio extraction error: {str(e)}")
            return None
    
    def transcribe_audio(self, audio_file: str) -> Dict:
        """Transcribe audio using Hugging Face Whisper"""
        try:
            # Check file size
            file_size_mb = os.path.getsize(audio_file) / (1024 * 1024)
            st.info(f"Audio file size: {file_size_mb:.1f}MB")
            
            if file_size_mb > 25:
                st.warning("Large file detected. This may take longer or fail.")
                st.info("Consider using a shorter video clip.")
            
            with st.spinner("Transcribing with Hugging Face Whisper..."):
                # Read audio file
                with open(audio_file, 'rb') as f:
                    audio_data = f.read()
                
                # Transcribe with retry logic
                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        result = self.inference_client.automatic_speech_recognition(
                            audio_data,
                            model=self.transcription_model
                        )
                        
                        if isinstance(result, dict) and 'text' in result:
                            text = result['text']
                        elif isinstance(result, str):
                            text = result
                        else:
                            raise Exception(f"Unexpected result format: {type(result)}")
                        
                        # Create simple segments (HF API doesn't provide detailed timestamps)
                        words = text.split()
                        segments = []
                        
                        # Estimate timestamps (very rough approximation)
                        words_per_second = 2.5  # Average speaking rate
                        current_time = 0
                        
                        for i in range(0, len(words), 10):  # 10 words per segment
                            segment_words = words[i:i+10]
                            segment_text = " ".join(segment_words)
                            segment_duration = len(segment_words) / words_per_second
                            
                            segments.append({
                                'start': current_time,
                                'end': current_time + segment_duration,
                                'text': segment_text
                            })
                            
                            current_time += segment_duration
                        
                        return {
                            'text': text,
                            'language': 'unknown',  # HF API doesn't return language
                            'segments': segments
                        }
                        
                    except Exception as e:
                        if attempt < max_retries - 1:
                            wait_time = 2 ** attempt
                            st.warning(f"Attempt {attempt + 1} failed, retrying in {wait_time}s...")
                            time.sleep(wait_time)
                        else:
                            raise e
                
        except Exception as e:
            st.error(f"Transcription failed: {str(e)}")
            st.error("Try using a smaller audio file or check your internet connection.")
            return None
    
    def chunk_transcript(self, transcript: Dict, chunk_size: int = 500) -> List[Dict]:
        """Split transcript into chunks"""
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
                chunks.append({
                    'text': current_chunk.strip(),
                    'start_time': current_start,
                    'end_time': current_end,
                    'timestamp': self.format_timestamp(current_start)
                })
                
                current_chunk = segment_text
                current_start = segment['start']
                word_count = segment_words
            else:
                current_chunk += " " + segment_text
                word_count += segment_words
            
            current_end = segment['end']
        
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
    
    def summarize_text(self, text: str, style: str = "bullet_points") -> str:
        """Summarize text using Hugging Face BART"""
        try:
            # Adjust parameters based on style
            if style == "bullet_points":
                max_length = 200
                min_length = 50
            elif style == "paragraph":
                max_length = 150
                min_length = 30
            elif style == "detailed":
                max_length = 300
                min_length = 100
            else:
                max_length = 150
                min_length = 50
            
            # Try HF API first
            try:
                result = self.inference_client.summarization(
                    text,
                    model=self.summarization_model,
                    parameters={
                        "max_length": max_length,
                        "min_length": min_length,
                        "do_sample": False
                    }
                )
                
                if isinstance(result, list) and len(result) > 0:
                    return result[0].get('summary_text', 'Summary not available')
                elif isinstance(result, dict):
                    return result.get('summary_text', 'Summary not available')
                else:
                    raise Exception("Unexpected API response format")
                    
            except Exception as api_error:
                st.warning(f"HF API failed: {api_error}")
                
                # Fallback to local model
                if self.local_summarizer:
                    st.info("Using backup local model...")
                    
                    # Truncate text for local model
                    words = text.split()
                    if len(words) > 500:
                        text = " ".join(words[:500])
                    
                    result = self.local_summarizer(
                        text,
                        max_length=max_length,
                        min_length=min_length,
                        do_sample=False
                    )
                    
                    return result[0]['summary_text']
                else:
                    return f"Summary unavailable: {api_error}"
                    
        except Exception as e:
            st.error(f"Summarization error: {str(e)}")
            return text[:200] + "..."
    
    def summarize_chunks(self, chunks: List[Dict], style: str) -> List[Dict]:
        """Summarize each chunk"""
        summarized_chunks = []
        progress_bar = st.progress(0)
        
        for i, chunk in enumerate(chunks):
            with st.spinner(f"Summarizing chunk {i+1}/{len(chunks)}..."):
                summary = self.summarize_text(chunk['text'], style)
                
                summarized_chunks.append({
                    **chunk,
                    'summary': summary
                })
            
            progress_bar.progress((i + 1) / len(chunks))
        
        return summarized_chunks
    
    def generate_final_summary(self, summarized_chunks: List[Dict], style: str) -> str:
        """Generate final summary from all chunk summaries"""
        all_summaries = " ".join([chunk['summary'] for chunk in summarized_chunks])
        return self.summarize_text(all_summaries, style)
    
    def export_to_pdf(self, content: Dict, filename: str):
        """Export summary to PDF"""
        try:
            doc = SimpleDocTemplate(filename, pagesize=letter)
            styles = getSampleStyleSheet()
            story = []
            
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=16,
                spaceAfter=30,
            )
            story.append(Paragraph("Video Summary Report", title_style))
            story.append(Spacer(1, 12))
            
            story.append(Paragraph(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
            story.append(Paragraph(f"**Style:** {content.get('style', 'N/A')}", styles['Normal']))
            story.append(Spacer(1, 12))
            
            if 'video_info' in content:
                story.append(Paragraph(f"**Source:** {content['video_info'].get('source', 'N/A')}", styles['Normal']))
                story.append(Spacer(1, 12))
            
            story.append(Paragraph("Overall Summary", styles['Heading2']))
            story.append(Paragraph(content.get('final_summary', ''), styles['Normal']))
            story.append(Spacer(1, 12))
            
            story.append(Paragraph("Timestamped Analysis", styles['Heading2']))
            for chunk in content.get('chunks', []):
                story.append(Paragraph(f"**{chunk['timestamp']}**", styles['Heading3']))
                story.append(Paragraph(chunk['summary'], styles['Normal']))
                story.append(Spacer(1, 8))
            
            doc.build(story)
            return True
        except Exception as e:
            st.error(f"PDF creation error: {str(e)}")
            return False

def main():
    st.set_page_config(
        page_title="Simple HF Video Summarizer",
        page_icon="🎥",
        layout="wide"
    )
    
    st.title("🎥 Simple Video Summarizer")
    st.markdown("**Powered by Hugging Face AI Models**")
    
    # Initialize session state
    if 'summarizer' not in st.session_state:
        st.session_state.summarizer = None
    
    # Sidebar - Just the essentials
    with st.sidebar:
        st.header("⚙️ Setup")
        
        # HF Token
        hf_token = st.text_input(
            "Hugging Face Token",
            type="password",
            help="Get your free token at https://huggingface.co/settings/tokens"
        )
        
        if hf_token:
            st.success("✅ Token provided!")
        else:
            st.error("❌ Hugging Face token required")
            st.info("This app uses only HF models for best quality")
        
        st.markdown("---")
        
        # Simple options
        st.subheader("📝 Options")
        
        summary_style = st.selectbox(
            "Summary Style",
            ["bullet_points", "paragraph", "detailed"],
            format_func=lambda x: {
                "bullet_points": "📝 Key Points",
                "paragraph": "📄 Paragraph",
                "detailed": "📚 Detailed"
            }[x]
        )
        
        chunk_size = st.slider("Text chunk size", 300, 800, 500, help="Smaller = more detailed")
        
        st.markdown("---")
        
        # Models being used
        st.subheader("🤖 Models Used")
        st.info("🎙️ **Transcription:** Whisper Large v3")
        st.info("📝 **Summarization:** BART Large CNN")
        st.caption("These are the best open-source models available")
        
        if torch.cuda.is_available():
            st.success("🚀 GPU acceleration enabled")
        else:
            st.warning("💻 Using CPU (slower)")
    
    # Main interface
    if not hf_token:
        st.warning("👈 Please add your Hugging Face token in the sidebar to get started")
        st.info("**Why do you need a token?**")
        st.markdown("""
        - Access to the latest AI models
        - Free with rate limits (generous for personal use)  
        - No local model downloads needed
        - Always up-to-date models
        """)
        st.markdown("**Get your token:** https://huggingface.co/settings/tokens")
        st.markdown("**Select permissions:** Just choose 'Read' access")
        return
    
    # Initialize summarizer
    if st.session_state.summarizer is None:
        with st.spinner("Initializing AI models..."):
            st.session_state.summarizer = HuggingFaceVideoSummarizer(hf_token)
    
    summarizer = st.session_state.summarizer
    
    tab1, tab2 = st.tabs(["📥 Process Video", "📊 Results"])
    
    with tab1:
        st.header("Input Video")
        
        input_method = st.radio(
            "Choose input:",
            ["YouTube URL", "Upload Video File"],
            horizontal=True
        )
        
        video_file = None
        youtube_url = None
        
        if input_method == "YouTube URL":
            youtube_url = st.text_input(
                "YouTube URL:",
                placeholder="https://www.youtube.com/watch?v=..."
            )
        else:
            video_file = st.file_uploader(
                "Upload video:",
                type=['mp4', 'avi', 'mov', 'mkv', 'webm']
            )
        
        if st.button("🚀 Start Processing", type="primary"):
            if not youtube_url and not video_file:
                st.error("Please provide a video")
                return
            
            try:
                # Step 1: Get audio
                audio_file = None
                
                if youtube_url:
                    audio_file = summarizer.download_youtube_audio(youtube_url)
                    video_info = {'source': 'YouTube', 'url': youtube_url}
                else:
                    temp_dir = tempfile.mkdtemp()
                    video_path = os.path.join(temp_dir, video_file.name)
                    with open(video_path, 'wb') as f:
                        f.write(video_file.read())
                    
                    audio_file = summarizer.extract_audio_from_video(video_path)
                    video_info = {'source': 'Upload', 'filename': video_file.name}
                
                if not audio_file:
                    st.error("Failed to process audio")
                    return
                
                # Step 2: Transcribe
                st.header("🎙️ Transcribing...")
                transcript = summarizer.transcribe_audio(audio_file)
                if not transcript:
                    st.error("Transcription failed")
                    return
                
                st.success("Transcription complete!")
                
                # Step 3: Chunk and summarize
                st.header("📝 Summarizing...")
                chunks = summarizer.chunk_transcript(transcript, chunk_size)
                st.info(f"Split into {len(chunks)} chunks")
                
                summarized_chunks = summarizer.summarize_chunks(chunks, summary_style)
                final_summary = summarizer.generate_final_summary(summarized_chunks, summary_style)
                
                # Store results
                st.session_state.results = {
                    'video_info': video_info,
                    'transcript': transcript,
                    'chunks': summarized_chunks,
                    'final_summary': final_summary,
                    'style': summary_style
                }
                
                st.success("✅ Complete! Check the Results tab.")
                
                # Quick preview
                st.subheader("🎯 Quick Preview")
                st.write(final_summary)
                
                # Cleanup
                try:
                    if audio_file and os.path.exists(audio_file):
                        os.remove(audio_file)
                except:
                    pass
                    
            except Exception as e:
                st.error(f"Processing failed: {str(e)}")
                st.info("Try a different video or check your internet connection")
    
    with tab2:
        if 'results' not in st.session_state:
            st.info("👈 Process a video first")
            return
        
        results = st.session_state.results
        
        st.header("📋 Results")
        
        # Info and export
        col1, col2 = st.columns([2, 1])
        
        with col1:
            if results['video_info']['source'] == 'YouTube':
                st.write(f"**Source:** {results['video_info']['url']}")
            else:
                st.write(f"**File:** {results['video_info']['filename']}")
            st.write(f"**Style:** {results['style']}")
        
        with col2:
            if st.button("📄 Export PDF"):
                pdf_filename = f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
                if summarizer.export_to_pdf(results, pdf_filename):
                    with open(pdf_filename, 'rb') as f:
                        st.download_button(
                            "⬇️ Download PDF",
                            f.read(),
                            pdf_filename,
                            "application/pdf"
                        )
        
        # Final summary
        st.subheader("🎯 Overall Summary")
        st.write(results['final_summary'])
        
        # Timestamped summaries
        st.subheader("⏰ Timestamped Analysis")
        
        for i, chunk in enumerate(results['chunks']):
            with st.expander(f"🕐 {chunk['timestamp']} - Part {i+1}"):
                st.write("**Summary:**")
                st.write(chunk['summary'])
                
                st.write("**Original:**")
                st.text_area(
                    f"Transcript part {i+1}",
                    chunk['text'],
                    height=100,
                    key=f"transcript_{i}"
                )
        
        # Full transcript
        with st.expander("📝 Full Transcript"):
            full_text = results['transcript']['text']
            st.text_area("Complete transcript", full_text, height=300)
            
            st.download_button(
                "⬇️ Download Transcript",
                full_text,
                f"transcript_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                "text/plain"
            )

if __name__ == "__main__":
    main()