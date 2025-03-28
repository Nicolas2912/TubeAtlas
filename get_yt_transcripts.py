# Standard library imports
import glob
import os
import re
from functools import partial
from multiprocessing import Pool

# Third-party imports
import google.generativeai as genai
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import scrapetube
import spacy
import torch
from google.generativeai import GenerativeModel
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from youtube_transcript_api import YouTubeTranscriptApi

# Load SpaCy model (Initialize after imports)
nlp = spacy.load("en_core_web_sm")


class YouTubeTranscriptManager:
    """Manages downloading and cleaning YouTube transcripts."""
    def __init__(self, output_dir="transcripts"):
        """
        Initialize the manager.

        Args:
            output_dir (str): Directory to save downloaded transcripts.
        """
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def _get_channel_video_ids(self, channel_id):
        """Get all video IDs from a YouTube channel."""
        print(f"Getting video IDs from channel {channel_id}...")
        videos = scrapetube.get_channel(channel_id=channel_id)
        video_ids = [video['videoId'] for video in videos]
        print(f"Found {len(video_ids)} videos")
        return video_ids

    def _download_single_transcript(self, video_id):
        """Download transcript for a single video."""
        try:
            transcript = YouTubeTranscriptApi.get_transcript(video_id)
            
            # Format the transcript as plain text
            text = ""
            for entry in transcript:
                text += f"{entry['start']:.2f} - {entry['start'] + entry['duration']:.2f}: {entry['text']}\n"
            
            # Save to file
            filepath = os.path.join(self.output_dir, f"{video_id}.txt")
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(text)
            
            return True
        except Exception as e:
            print(f"Error downloading transcript for {video_id}: {e}")
            return False

    def download_channel_transcripts(self, channel_id):
        """Download all transcripts from a YouTube channel."""
        video_ids = self._get_channel_video_ids(channel_id)
        
        # Download transcripts
        successful = 0
        for i, video_id in enumerate(video_ids):
            print(f"Downloading transcript {i+1}/{len(video_ids)}: {video_id}")
            if self._download_single_transcript(video_id):
                successful += 1
        
        print(f"Downloaded {successful}/{len(video_ids)} transcripts to {self.output_dir}")
        return successful

    def clean_transcript_files(self, directory_path=None):
        """
        Parse all transcript files in the directory, removing timestamps
        and keeping only the text content. Overwrites the original files.

        Args:
            directory_path (str, optional): Path to the directory containing transcript files. 
                                            Defaults to the manager's output directory.

        Returns:
            int: Number of files processed
        """
        if directory_path is None:
            directory_path = self.output_dir
            
        # Get all txt files in the directory
        file_pattern = os.path.join(directory_path, "*.txt")
        transcript_files = glob.glob(file_pattern)
        
        files_processed = 0
        
        print(f"Cleaning {len(transcript_files)} files in {directory_path}...")
        for file_path in transcript_files:
            try:
                # Read the file content
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Use a simpler approach - just remove the timestamp patterns
                # This keeps the text flow intact
                cleaned_content = re.sub(r'\d+\.\d+ - \d+\.\d+: ', '', content)
                
                # Write the cleaned content back to the same file
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(cleaned_content)
                
                files_processed += 1
                # print(f"Processed: {os.path.basename(file_path)}") # Optional: uncomment for detailed progress
                
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
        
        print(f"Finished cleaning {files_processed} files.")
        return files_processed


def count_tokens_in_directory(api_key, directory_path):
    """
    Count tokens in all text files in a directory using Gemini 2.0 Flash model.  
    Args:
        api_key (str): Google AI API key
        directory_path (str): Path to the directory containing text files      
    Returns:
        dict: Dictionary with filenames as keys and token counts as values
    """
    # Configure the API
    genai.configure(api_key=api_key)   
    # Initialize the model
    model = GenerativeModel('gemini-2.0-flash')   
    # Get all txt files in the directory
    file_pattern = os.path.join(directory_path, "*.txt")
    text_files = glob.glob(file_pattern)  

    # Dictionary to store results
    token_counts = {}
    total_tokens = 0   

    # Process each file
    for file_path in text_files:
        try:
            # Read file content
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()          
            # Count tokens
            token_count = model.count_tokens(content).total_tokens           
            # Store result
            filename = os.path.basename(file_path)

            token_counts[filename] = token_count

            total_tokens += token_count           

            print(f"{filename}: {token_count} tokens")
          
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
   
    print(f"\nTotal tokens across all files: {total_tokens}")
    return token_counts

 

if __name__ == "__main__":
    # Configuration
    channel_id = "UC2D2CMWXMOVWx7giW1n3LIg"  # Example channel ID
    transcript_dir = "transcripts"
    # api_key = "YOUR_API_KEY_HERE"  # Replace with your actual Google AI API key if using token counting

    