import streamlit as st
import os
import tempfile
from datetime import datetime, timedelta
import re
from typing import List, Dict
import time

# Core libraries
import moviepy.editor as mp
import yt_dlp
import torch

# Local AI models
import whisper
from transformers import pipeline

# PDF generation
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch

class LocalVideoSummarizer:
    def __init__(self):
        self.whisper_model = None
        self.summarizer = None
    
    def setup_models(self, whisper_size: str = "base"):
        """Setup local models"""
        try:
            # Load Whisper model
            st.info(f"Loading Whisper {whisper_size} model...")
            self.whisper_model = whisper.load_model(whisper_size)
            st.success(f"✅ Whisper {whisper_size} loaded!")
            
            # Load summarization model
            st.info("Loading summarization model...")
            device = 0 if torch.cuda.is_available() else -1
            
            # Try models in order of preference
            models_to_try = [
                ("facebook/bart-large-cnn", "BART Large CNN"),
                ("facebook/bart-base", "BART Base"),
                ("t5-small", "T5 Small")
            ]
            
            for model_name, display_name in models_to_try:
                try:
                    self.summarizer = pipeline(
                        "summarization",
                        model=model_name,
                        device=device,
                        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
                    )
                    st.success(f"✅ {display_name} loaded!")
                    break
                except Exception as e:
                    st.warning(f"Couldn't load {display_name}, trying next...")
                    continue
            
            if not self.summarizer:
                st.error("Failed to load any summarization model")
                return False
            
            return True
            
        except Exception as e:
            st.error(f"Model setup failed: {str(e)}")
            st.error("Install required packages: pip install torch transformers openai-whisper")
            return False
    
    def download_youtube_audio(self, url: str) -> str:
        """Download audio from YouTube"""
        try:
            with st.spinner("Downloading YouTube audio..."):
                temp_dir = tempfile.mkdtemp()
                
                ydl_opts = {
                    'format': 'bestaudio[ext=m4a]/bestaudio/best',
                    'outtmpl': os.path.join(temp_dir, 'audio.%(ext)s'),
                    'postprocessors': [{
                        'key': 'FFmpegExtractAudio',
                        'preferredcodec': 'wav',
                        'preferredquality': '192',
                    }],
                    'quiet': True,
                    'no_warnings': True,
                }
                
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    try:
                        info = ydl.extract_info(url, download=False)
                        duration = info.get('duration', 0)
                        
                        if duration > 3600:
                            st.warning("⚠️ Video is over 1 hour. Processing will take a long time.")
                        
                        st.write(f"**Title:** {info.get('title', 'Unknown')}")
                        st.write(f"**Duration:** {str(timedelta(seconds=duration))}")
                        st.write(f"**Channel:** {info.get('uploader', 'Unknown')}")
                    except:
                        st.info("Downloading video...")
                    
                    ydl.download([url])
                
                # Find downloaded audio file
                for file in os.listdir(temp_dir):
                    if file.endswith('.wav'):
                        audio_path = os.path.join(temp_dir, file)
                        file_size = os.path.getsize(audio_path) / (1024 * 1024)
                        st.success(f"Downloaded: {file_size:.1f}MB")
                        return audio_path
                
                # Look for any audio file
                audio_files = [f for f in os.listdir(temp_dir) 
                              if f.endswith(('.mp3', '.m4a', '.wav', '.flac'))]
                if audio_files:
                    return os.path.join(temp_dir, audio_files[0])
                
                return None
                
        except Exception as e:
            st.error(f"YouTube download failed: {str(e)}")
            return None
    
    def extract_audio_from_video(self, video_path: str) -> str:
        """Extract audio from video file"""
        try:
            with st.spinner("Extracting audio from video..."):
                video = mp.VideoFileClip(video_path)
                
                temp_dir = tempfile.mkdtemp()
                audio_file = os.path.join(temp_dir, "extracted_audio.wav")
                
                video.audio.write_audiofile(
                    audio_file,
                    verbose=False,
                    logger=None,
                    codec='pcm_s16le'
                )
                video.close()
                
                file_size = os.path.getsize(audio_file) / (1024 * 1024)
                st.success(f"Extracted: {file_size:.1f}MB")
                
                return audio_file
                
        except Exception as e:
            st.error(f"Audio extraction failed: {str(e)}")
            return None
    
    def transcribe_audio(self, audio_file: str, language: str = None) -> Dict:
        """Transcribe audio using local Whisper"""
        try:
            if not self.whisper_model:
                st.error("Whisper model not loaded!")
                return None
            
            file_size = os.path.getsize(audio_file) / (1024 * 1024)
            if file_size > 100:
                st.warning(f"Large audio file ({file_size:.1f}MB). This will take several minutes.")
            
            with st.spinner("Transcribing audio with Whisper... Please wait."):
                result = self.whisper_model.transcribe(
                    audio_file,
                    language=language if language != 'auto' else None,
                    task="transcribe",
                    verbose=False,
                    fp16=torch.cuda.is_available()
                )
            
            if not result or not result.get('text'):
                st.error("Transcription produced no text")
                return None
            
            detected_lang = result.get('language', 'unknown')
            st.success(f"✅ Transcription complete! Language: {detected_lang}")
            st.info(f"Transcribed {len(result['text'].split())} words")
            
            return result
            
        except Exception as e:
            st.error(f"Transcription failed: {str(e)}")
            
            if "CUDA" in str(e) or "GPU" in str(e):
                st.info("💡 GPU error detected. Try restarting the app.")
            elif "memory" in str(e).lower():
                st.info("💡 Memory error. Try using 'tiny' or 'base' Whisper model.")
            
            return None
    
    def chunk_transcript(self, transcript: Dict, chunk_size: int = 500) -> List[Dict]:
        """Split transcript into chunks"""
        segments = transcript.get('segments', [])
        
        if not segments:
            text = transcript.get('text', '')
            words = text.split()
            chunks = []
            
            for i in range(0, len(words), chunk_size):
                chunk_words = words[i:i + chunk_size]
                chunk_text = ' '.join(chunk_words)
                start_time = i * 0.5
                end_time = (i + len(chunk_words)) * 0.5
                
                chunks.append({
                    'text': chunk_text,
                    'start_time': start_time,
                    'end_time': end_time,
                    'timestamp': self.format_timestamp(start_time),
                    'word_count': len(chunk_words)
                })
            
            return chunks
        
        chunks = []
        current_chunk = ""
        current_start = 0
        current_end = 0
        word_count = 0
        
        for segment in segments:
            segment_text = segment.get('text', '').strip()
            if not segment_text:
                continue
                
            segment_words = len(segment_text.split())
            
            if word_count == 0:
                current_start = segment.get('start', 0)
            
            if word_count + segment_words > chunk_size and current_chunk:
                chunks.append({
                    'text': current_chunk.strip(),
                    'start_time': current_start,
                    'end_time': current_end,
                    'timestamp': self.format_timestamp(current_start),
                    'word_count': word_count
                })
                
                current_chunk = segment_text
                current_start = segment.get('start', current_end)
                word_count = segment_words
            else:
                if current_chunk:
                    current_chunk += " " + segment_text
                else:
                    current_chunk = segment_text
                word_count += segment_words
            
            current_end = segment.get('end', current_start + 10)
        
        if current_chunk:
            chunks.append({
                'text': current_chunk.strip(),
                'start_time': current_start,
                'end_time': current_end,
                'timestamp': self.format_timestamp(current_start),
                'word_count': word_count
            })
        
        return chunks
    
    def format_timestamp(self, seconds: float) -> str:
        """Format seconds as timestamp"""
        total_seconds = int(seconds)
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        secs = total_seconds % 60
        
        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"
        else:
            return f"{minutes:02d}:{secs:02d}"
    
    def summarize_text(self, text: str, style: str = "bullet_points") -> str:
        """Summarize text using local model"""
        try:
            if not self.summarizer:
                return "Summarization model not available"
            
            # Truncate if too long
            words = text.split()
            max_words = 800
            
            if len(words) > max_words:
                text = ' '.join(words[:max_words])
            
            # Set parameters based on style
            if style == "bullet_points":
                max_length = 200
                min_length = 50
            elif style == "paragraph":
                max_length = 150
                min_length = 40
            elif style == "detailed":
                max_length = 300
                min_length = 100
            else:
                max_length = 150
                min_length = 50
            
            result = self.summarizer(
                text,
                max_length=max_length,
                min_length=min_length,
                do_sample=False,
                early_stopping=True
            )
            
            if result and len(result) > 0:
                summary = result[0].get('summary_text', '')
                
                # Format as bullet points if requested
                if style == "bullet_points" and summary:
                    sentences = re.split(r'[.!?]+', summary)
                    bullets = []
                    for sentence in sentences:
                        sentence = sentence.strip()
                        if sentence and len(sentence) > 10:
                            bullets.append(f"• {sentence}")
                    if bullets:
                        summary = '\n'.join(bullets)
                
                return summary if summary else "Summary could not be generated"
            
            return "No summary generated"
            
        except Exception as e:
            st.warning(f"Summarization error: {str(e)}")
            sentences = re.split(r'[.!?]+', text)
            return '. '.join(sentences[:3]) + "..."
    
    def export_to_pdf(self, results: Dict, video_info: Dict, filename: str) -> bool:
        """Export results to PDF"""
        try:
            doc = SimpleDocTemplate(filename, pagesize=letter)
            styles = getSampleStyleSheet()
            story = []
            
            title_style = ParagraphStyle(
                'Title',
                parent=styles['Title'],
                fontSize=18,
                spaceAfter=30
            )
            story.append(Paragraph("Video Summary Report", title_style))
            story.append(Spacer(1, 20))
            
            story.append(Paragraph(f"<b>Generated:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
            story.append(Paragraph(f"<b>Processing Time:</b> {results['processing_time']:.1f} seconds", styles['Normal']))
            
            if video_info.get('source') == 'YouTube':
                story.append(Paragraph(f"<b>Source:</b> YouTube", styles['Normal']))
            else:
                story.append(Paragraph(f"<b>Source:</b> Uploaded file", styles['Normal']))
            
            story.append(Spacer(1, 20))
            
            story.append(Paragraph("Executive Summary", styles['Heading2']))
            story.append(Paragraph(results['final_summary'], styles['Normal']))
            story.append(Spacer(1, 20))
            
            story.append(Paragraph("Detailed Analysis", styles['Heading2']))
            for chunk in results['chunks']:
                story.append(Paragraph(f"<b>{chunk['timestamp']}</b>", styles['Heading3']))
                story.append(Paragraph(chunk['summary'], styles['Normal']))
                story.append(Spacer(1, 10))
            
            doc.build(story)
            return True
            
        except Exception as e:
            st.error(f"PDF export failed: {str(e)}")
            return False

def main():
    st.set_page_config(
        page_title="Local Video Summarizer",
        page_icon="🎥",
        layout="wide"
    )
    
    st.title("🎥 Local Video Summarizer")
    st.markdown("**Powered by Local AI Models - No API Keys Required**")
    
    # Initialize session state
    if 'summarizer' not in st.session_state:
        st.session_state.summarizer = None
        st.session_state.models_loaded = False
    if 'results' not in st.session_state:
        st.session_state.results = None
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Model selection
        whisper_size = st.selectbox(
            "Whisper Model",
            ["tiny", "base", "small", "medium", "large"],
            index=1,
            help="tiny=fastest, large=most accurate"
        )
        
        # Language
        target_language = st.selectbox(
            "Language",
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
        
        # Summary options
        summary_style = st.selectbox(
            "Summary Style",
            ["bullet_points", "paragraph", "detailed"],
            format_func=lambda x: {
                "bullet_points": "📝 Key Points",
                "paragraph": "📄 Paragraph",
                "detailed": "📚 Detailed"
            }[x]
        )
        
        chunk_size = st.slider("Words per chunk", 300, 800, 500)
        
        st.markdown("---")
        
        # System info
        st.subheader("💻 System")
        if torch.cuda.is_available():
            st.success("🚀 GPU Available")
        else:
            st.warning("💻 CPU Mode")
        
        if st.session_state.models_loaded:
            st.success("✅ Models Ready")
        else:
            st.info("⏳ Models not loaded")
    
    # Main interface
    tab1, tab2 = st.tabs(["📥 Process Video", "📊 Results"])
    
    with tab1:
        st.header("Video Input")
        
        # Model loading
        if not st.session_state.models_loaded:
            st.info("👇 First, load the AI models (this downloads models on first run)")
            
            if st.button("🔄 Load AI Models", type="primary"):
                summarizer = LocalVideoSummarizer()
                if summarizer.setup_models(whisper_size):
                    st.session_state.summarizer = summarizer
                    st.session_state.models_loaded = True
                    st.rerun()
            
            st.stop()
        
        summarizer = st.session_state.summarizer
        
        # Input selection
        input_method = st.radio(
            "Input Method:",
            ["YouTube URL", "Upload Video"],
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
                "Video file:",
                type=['mp4', 'avi', 'mov', 'mkv', 'webm', 'm4v']
            )
        
        # Process button
        if st.button("🚀 Start Processing", type="primary", disabled=not (youtube_url or video_file)):
            if not youtube_url and not video_file:
                st.error("Please provide a video")
                return
            
            start_time = time.time()
            
            try:
                # Get audio
                audio_file = None
                video_info = {}
                
                if youtube_url:
                    st.header("📺 Downloading")
                    audio_file = summarizer.download_youtube_audio(youtube_url)
                    video_info = {'source': 'YouTube', 'url': youtube_url}
                else:
                    st.header("📁 Processing File")
                    temp_dir = tempfile.mkdtemp()
                    video_path = os.path.join(temp_dir, video_file.name)
                    
                    with open(video_path, 'wb') as f:
                        f.write(video_file.read())
                    
                    audio_file = summarizer.extract_audio_from_video(video_path)
                    video_info = {'source': 'Upload', 'filename': video_file.name}
                
                if not audio_file:
                    st.error("Failed to get audio")
                    return
                
                # Transcribe
                st.header("🎙️ Transcribing")
                transcript = summarizer.transcribe_audio(
                    audio_file,
                    target_language if target_language != 'auto' else None
                )
                
                if not transcript:
                    st.error("Transcription failed")
                    return
                
                # Chunk
                st.header("📝 Processing Text")
                chunks = summarizer.chunk_transcript(transcript, chunk_size)
                st.success(f"Split into {len(chunks)} chunks")
                
                # Summarize
                st.header("🔄 Generating Summaries")
                progress_bar = st.progress(0)
                
                summarized_chunks = []
                for i, chunk in enumerate(chunks):
                    summary = summarizer.summarize_text(chunk['text'], summary_style)
                    
                    chunk_result = {
                        **chunk,
                        'summary': summary
                    }
                    summarized_chunks.append(chunk_result)
                    
                    progress_bar.progress((i + 1) / len(chunks))
                
                # Final summary
                st.header("🎯 Final Summary")
                all_summaries = ' '.join([c['summary'] for c in summarized_chunks])
                final_summary = summarizer.summarize_text(all_summaries, summary_style)
                
                processing_time = time.time() - start_time
                
                # Store results
                st.session_state.results = {
                    'processing_results': {
                        'success': True,
                        'transcript': transcript,
                        'chunks': summarized_chunks,
                        'final_summary': final_summary,
                        'processing_time': processing_time
                    },
                    'video_info': video_info,
                    'settings': {
                        'style': summary_style,
                        'chunk_size': chunk_size,
                        'whisper_model': whisper_size
                    }
                }
                
                st.success(f"✅ Complete in {processing_time:.1f} seconds!")
                
                # Preview
                with st.expander("👀 Preview", expanded=True):
                    st.write("**Final Summary:**")
                    st.write(final_summary)
                
                # Cleanup
                try:
                    if audio_file and os.path.exists(audio_file):
                        os.remove(audio_file)
                except:
                    pass
                    
            except Exception as e:
                st.error(f"Error: {str(e)}")
    
    with tab2:
        if not st.session_state.results:
            st.info("👈 Process a video first")
            return
        
        data = st.session_state.results
        results = data['processing_results']
        video_info = data['video_info']
        
        st.header("📋 Results")
        
        # Metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Processing Time", f"{results['processing_time']:.1f}s")
        
        with col2:
            st.metric("Chunks", len(results['chunks']))
        
        with col3:
            total_words = sum(c['word_count'] for c in results['chunks'])
            st.metric("Words", f"{total_words:,}")
        
        # Export
        if st.button("📄 Generate PDF"):
            pdf_filename = f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
            
            if st.session_state.summarizer.export_to_pdf(results, video_info, pdf_filename):
                with open(pdf_filename, 'rb') as f:
                    st.download_button(
                        "⬇️ Download PDF",
                        f.read(),
                        pdf_filename,
                        "application/pdf"
                    )
        
        # Summary
        st.subheader("🎯 Executive Summary")
        st.write(results['final_summary'])
        
        # Timestamped
        st.subheader("⏰ Timestamped Analysis")
        
        for i, chunk in enumerate(results['chunks']):
            with st.expander(f"🕐 {chunk['timestamp']} - Part {i+1} ({chunk['word_count']} words)"):
                st.write("**Summary:**")
                st.write(chunk['summary'])
                
                with st.expander("📝 Full Text"):
                    st.text_area(
                        "Transcript",
                        chunk['text'],
                        height=150,
                        key=f"transcript_{i}"
                    )
        
        # Download transcript
        st.subheader("📄 Full Transcript")
        st.download_button(
            "⬇️ Download Transcript",
            results['transcript']['text'],
            f"transcript_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            "text/plain"
        )

if __name__ == "__main__":
    main()