import glob
import os
import re
from functools import partial
from multiprocessing import Pool

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import scrapetube
import spacy
import torch
from google import genai
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from youtube_transcript_api import YouTubeTranscriptApi


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
        video_ids = [video["videoId"] for video in videos]
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

        print(
            f"Downloaded {successful}/{len(video_ids)} transcripts to {self.output_dir}"
        )
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
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()

                # Use a simpler approach - just remove the timestamp patterns
                # This keeps the text flow intact
                cleaned_content = re.sub(r"\d+\.\d+ - \d+\.\d+: ", "", content)

                # Write the cleaned content back to the same file
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(cleaned_content)

                files_processed += 1
                # print(f"Processed: {os.path.basename(file_path)}") # Optional: uncomment for detailed progress

            except Exception as e:
                print(f"Error processing {file_path}: {e}")

        print(f"Finished cleaning {files_processed} files.")
        return files_processed


class KnowledgeGraphBuilder:
    def __init__(self, model_name="Babelscape/rebel-large", use_gpu=True):
        """
        Initialize the Knowledge Graph Builder with NLP models.

        Args:
            model_name (str): Name of the pre-trained model to use
            use_gpu (bool): Whether to use GPU if available
        """
        # Load SpaCy model
        self.nlp = spacy.load("en_core_web_sm")

        # Load the relation extraction model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.to("cpu")
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

        # Set device
        self.device = (
            "cuda"
            if use_gpu and hasattr(torch, "cuda") and torch.cuda.is_available()
            else "cpu"
        )
        self.model.to(self.device)

        # Create sentence splitter regex
        self.sentence_splitter = re.compile(
            r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s"
        )

    def preprocess_and_chunk_text(self, text_file, chunk_size=512, overlap=50):
        """Split large text into manageable chunks with some overlap."""
        with open(text_file, "r", encoding="utf-8") as f:
            text = f.read()

        # Process with SpaCy to get sentence boundaries
        doc = self.nlp(text)
        sentences = [sent.text.strip() for sent in doc.sents]

        chunks = []
        current_chunk = []
        current_length = 0

        for sentence in sentences:
            # Tokenize to count tokens accurately
            sentence_tokens = self.tokenizer.tokenize(sentence)
            sentence_length = len(sentence_tokens)

            if current_length + sentence_length > chunk_size:
                # Save current chunk
                chunks.append(" ".join(current_chunk))

                # Keep some sentences for overlap
                overlap_sentences = (
                    current_chunk[-overlap:]
                    if overlap < len(current_chunk)
                    else current_chunk
                )
                current_chunk = overlap_sentences
                current_length = sum(
                    len(self.tokenizer.tokenize(s)) for s in current_chunk
                )

            current_chunk.append(sentence)
            current_length += sentence_length

        # Add the last chunk if it's not empty
        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks

    def process_file(self, file_path, chunk_size=512, overlap=50):
        """Process a single file into chunks."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                file_content = f.read()

            # Split text into sentences using regex (much faster than spaCy)
            sentences = [
                s.strip()
                for s in self.sentence_splitter.split(file_content)
                if s.strip()
            ]

            # Batch tokenize all sentences at once
            sentence_tokens = [self.tokenizer.tokenize(s) for s in sentences]
            sentence_lengths = [len(tokens) for tokens in sentence_tokens]

            # Create chunks based on token count
            file_chunks = []
            current_chunk = []
            current_length = 0

            for i, sentence in enumerate(sentences):
                if current_length + sentence_lengths[i] > chunk_size:
                    # Save current chunk
                    if current_chunk:
                        file_chunks.append(" ".join(current_chunk))

                    # Keep some sentences for overlap
                    overlap_idx = max(0, len(current_chunk) - overlap)
                    current_chunk = current_chunk[overlap_idx:]
                    current_length = sum(sentence_lengths[i - len(current_chunk) : i])

                current_chunk.append(sentence)
                current_length += sentence_lengths[i]

            # Add the last chunk if it's not empty
            if current_chunk:
                file_chunks.append(" ".join(current_chunk))

            return file_chunks

        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return []

    def collect_and_process_all_files(
        self,
        directory_path,
        file_extensions=["txt", "md", "csv"],
        chunk_size=512,
        overlap=50,
        n_processes=4,
    ):
        """Process files efficiently with multiprocessing."""
        # Find all files with specified extensions
        file_paths = []
        for ext in file_extensions:
            file_paths.extend(glob.glob(os.path.join(directory_path, f"*.{ext}")))

        print(f"Found {len(file_paths)} files to process")

        # Use partial to pass the extra arguments to process_file
        process_func = partial(
            self.process_file, chunk_size=chunk_size, overlap=overlap
        )

        # Process files in parallel
        with Pool(processes=n_processes) as pool:
            all_chunks_nested = list(
                tqdm(
                    pool.imap(process_func, file_paths),
                    total=len(file_paths),
                    desc="Processing files",
                )
            )

        # Flatten the list of chunks
        all_chunks = [chunk for chunks in all_chunks_nested for chunk in chunks]

        print(f"Total chunks created: {len(all_chunks)}")
        return all_chunks

    def extract_triplets(self, text):
        """Extract (subject, relation, object) triplets from text using RebelLarge."""
        # Tokenize and generate output
        inputs = self.tokenizer(
            text, return_tensors="pt", max_length=512, truncation=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        outputs = self.model.generate(
            inputs["input_ids"],
            max_length=256,
            num_beams=5,
            num_return_sequences=1,
            early_stopping=True,
        )

        # Decode the output
        extracted_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Process the output to extract triplets
        triplets = []
        relations = extracted_text.strip().split("), (")

        for relation in relations:
            relation = relation.replace("(", "").replace(")", "")
            if relation:
                try:
                    subj, rel, obj = relation.split(", ")
                    triplets.append((subj.strip('"'), rel.strip('"'), obj.strip('"')))
                except ValueError:
                    # Skip if the triplet isn't properly formatted
                    continue

        return triplets

    def extract_triplets_from_output(self, output):
        """Extract triplets from REBEL model output."""
        # Decode the output if it's a tensor
        if hasattr(output, "cpu"):
            extracted_text = self.tokenizer.decode(output, skip_special_tokens=True)
        else:
            extracted_text = output

        # Process the output to extract triplets
        triplets = []
        relations = extracted_text.strip().split("), (")

        for relation in relations:
            relation = relation.replace("(", "").replace(")", "")
            if relation:
                try:
                    subj, rel, obj = relation.split(", ")
                    triplets.append((subj.strip('"'), rel.strip('"'), obj.strip('"')))
                except ValueError:
                    # Skip if the triplet isn't properly formatted
                    continue

        return triplets

    def process_batch(self, chunk_list):
        """Process a batch of chunks to extract triplets."""
        batch_triplets = []
        for chunk in chunk_list:
            try:
                triplets = self.extract_triplets(chunk)
                batch_triplets.extend(triplets)
            except Exception as e:
                print(f"Error processing chunk: {e}")
        return batch_triplets

    def prepare_batch_inputs(self, batch_idx, batches, tokenizer):
        """
        Prepare inputs for a batch by tokenizing text.

        Args:
            batch_idx: Index of the batch to process
            batches: List of all batches
            tokenizer: The tokenizer to use

        Returns:
            Tuple of (batch_idx, batch, encoded_inputs) or None if batch_idx is invalid
        """
        if batch_idx >= len(batches):
            return None

        batch = batches[batch_idx]

        # Batch tokenize all texts at once instead of one-by-one
        try:
            # This is much faster than tokenizing one by one
            encoded_inputs = tokenizer(
                batch,
                return_tensors="pt",
                max_length=512,
                truncation=True,
                padding=True,
            )
            return batch_idx, batch, encoded_inputs
        except Exception as e:
            print(f"Error batch encoding texts: {e}")
            return batch_idx, None, None

    def process_chunks_for_triplets(self, chunks, batch_size=64, max_workers=4):
        """
        Process all t ext chunks and extract triplets with highly optimized GPU acceleration.

        Args:
            chunks: List of text chunks to process
            batch_size: Size of batches for GPU processing (increased from default)
            max_workers: Maximum number of worker threads

        Returns:
            List of unique (subject, relation, object) triplets
        """
        import time
        from queue import Queue
        from threading import Thread

        import torch
        from tqdm.auto import tqdm

        print(f"Starting optimized triplet extraction with {len(chunks)} chunks")
        print(
            f"Using device: {self.device}, batch size: {batch_size}, workers: {max_workers}"
        )

        # Use a set for O(1) duplicate checking
        seen_triplets = set()
        all_triplets = []

        # Warm up the GPU
        if self.device == "cuda":
            torch.cuda.empty_cache()
            # Run a small batch to initialize CUDA context
            dummy_input = self.tokenizer("Warming up the GPU", return_tensors="pt").to(
                self.device
            )
            self.model.eval()
            with torch.no_grad():
                _ = self.model.generate(dummy_input["input_ids"], max_length=10)

        # Create larger batches for better GPU utilization
        # Increase batch size to better utilize the GPU
        batches = [
            chunks[i : i + batch_size] for i in range(0, len(chunks), batch_size)
        ]
        total_batches = len(batches)
        print(f"Created {total_batches} batches")

        # Use three queues to manage the pipeline effectively
        to_tokenize_queue = Queue(maxsize=max_workers * 2)  # Indices to tokenize
        to_process_queue = Queue(maxsize=max_workers)  # Tokenized batches to process
        processed_queue = Queue(maxsize=max_workers)  # Processed results

        # Track progress and state
        next_batch_to_submit = max_workers * 2
        next_batch_to_process = 0
        batches_submitted = 0
        batches_processed = 0

        # Tokenization worker thread
        def tokenization_worker():
            """Thread worker for tokenization"""
            while True:
                try:
                    batch_idx = to_tokenize_queue.get()
                    if batch_idx is None:  # Sentinel to stop
                        to_tokenize_queue.task_done()
                        break

                    # Tokenize this batch
                    result = self.prepare_batch_inputs(
                        batch_idx, batches, self.tokenizer
                    )
                    to_process_queue.put(result)
                    to_tokenize_queue.task_done()
                except Exception as e:
                    print(f"Error in tokenization worker: {e}")
                    to_tokenize_queue.task_done()

        # GPU processing worker thread
        def gpu_worker():
            """Thread worker for GPU processing"""
            while True:
                try:
                    item = to_process_queue.get()
                    if item is None:  # Sentinel to stop
                        to_process_queue.task_done()
                        break

                    batch_idx, batch, encoded_inputs = item

                    # Skip if there was an error during tokenization
                    if batch is None:
                        processed_queue.put((batch_idx, []))
                        to_process_queue.task_done()
                        continue

                    # Process with GPU
                    with torch.no_grad():
                        triplets = self._process_batch_optimized(batch, encoded_inputs)

                    processed_queue.put((batch_idx, triplets))
                    to_process_queue.task_done()
                except Exception as e:
                    print(f"Error in GPU worker: {e}")
                    processed_queue.put((None, []))
                    to_process_queue.task_done()

        # Start the tokenization workers
        tokenizers = []
        for i in range(max_workers):
            worker = Thread(target=tokenization_worker)
            worker.daemon = True
            worker.start()
            tokenizers.append(worker)

        # Start the GPU worker (just one since we want to maximize GPU usage)
        gpu_thread = Thread(target=gpu_worker)
        gpu_thread.daemon = True
        gpu_thread.start()

        # Submit initial batches to tokenization queue
        for i in range(min(max_workers * 2, total_batches)):
            to_tokenize_queue.put(i)
            batches_submitted += 1

        # Create progress bar
        progress_bar = tqdm(total=total_batches, desc="Processing chunks")
        start_time = time.time()
        last_update = start_time

        # Main processing loop
        try:
            while batches_processed < total_batches:
                # Get processed results
                try:
                    batch_idx, triplets = processed_queue.get(timeout=1.0)

                    # Update progress
                    batches_processed += 1
                    progress_bar.update(1)

                    # Add unique triplets to our collection
                    for triplet in triplets:
                        if triplet not in seen_triplets:
                            seen_triplets.add(triplet)
                            all_triplets.append(triplet)

                    # Log progress periodically
                    current_time = time.time()
                    if current_time - last_update > 5.0:  # Every 5 seconds
                        elapsed = current_time - start_time
                        rate = batches_processed / elapsed if elapsed > 0 else 0
                        eta = (
                            (total_batches - batches_processed) / rate
                            if rate > 0
                            else float("inf")
                        )

                        print(
                            f"Processed: {batches_processed}/{total_batches} batches, "
                            f"Rate: {rate:.2f} batches/sec, "
                            f"Triplets: {len(all_triplets)} ({len(seen_triplets)} unique), "
                            f"ETA: {eta/60:.1f} minutes"
                        )

                        last_update = current_time

                    processed_queue.task_done()

                    # Submit next batch if available
                    if next_batch_to_submit < total_batches:
                        to_tokenize_queue.put(next_batch_to_submit)
                        next_batch_to_submit += 1
                        batches_submitted += 1

                except Exception as e:
                    # Queue.get timeout or other error, continue
                    continue

        finally:
            # Clean up resources
            progress_bar.close()

            # Send sentinel values to stop workers
            for _ in range(max_workers):
                to_tokenize_queue.put(None)

            # Wait for tokenizers to finish
            for worker in tokenizers:
                worker.join(timeout=1.0)

            # Stop the GPU worker
            to_process_queue.put(None)
            gpu_thread.join(timeout=1.0)

            # Final report
            elapsed = time.time() - start_time
            print(f"\nExtraction completed in {elapsed:.1f} seconds")
            print(f"Processed {len(chunks)} chunks in {batches_processed} batches")
            print(
                f"Extracted {len(all_triplets)} unique triplets from {sum(len(seen_triplets) for _ in [1])} total"
            )

        return all_triplets

    def _process_batch_optimized(self, batch, encoded_inputs):
        """
        Process a batch of text chunks with optimized GPU acceleration.

        Args:
            batch: List of text chunks
            encoded_inputs: Pre-tokenized inputs

        Returns:
            List of (subject, relation, object) triplets
        """
        import torch

        # Fix deprecation warning by using the new import
        from torch.amp import autocast

        if encoded_inputs is None:
            return []

        batch_triplets = []

        try:
            # Process entire batch at once on GPU for maximum efficiency
            # Move everything to device once
            inputs = {k: v.to(self.device) for k, v in encoded_inputs.items()}

            # Process with mixed precision for faster computation
            with autocast(device_type=self.device, enabled=True):
                outputs = self.model.generate(
                    inputs["input_ids"],
                    max_length=256,
                    num_beams=3,  # Reduced from 5 to speed up with minimal quality loss
                    num_return_sequences=1,
                    early_stopping=True,
                )

                # Process all outputs at once
                for i, output in enumerate(outputs):
                    extracted_text = self.tokenizer.decode(
                        output, skip_special_tokens=True
                    )
                    text_triplets = self._extract_triplets_fast(extracted_text)
                    batch_triplets.extend(text_triplets)

        except Exception as e:
            print(f"Error processing batch: {e}")

        return batch_triplets

    def _extract_triplets_fast(self, extracted_text):
        """
        Extract triplets with optimized string processing.

        Args:
            extracted_text: Text containing relation triplets

        Returns:
            List of (subject, relation, object) triplets
        """
        # Use faster string splitting approach
        if not extracted_text.strip():
            return []

        triplets = []

        # More efficient string processing
        relations = extracted_text.strip().split("), (")
        for relation in relations:
            relation = relation.replace("(", "").replace(")", "")
            if relation:
                try:
                    # Use a more direct approach
                    parts = relation.split(", ", 2)  # Split by comma, limit to 3 parts
                    if len(parts) == 3:
                        subj = parts[0].strip('"')
                        rel = parts[1].strip('"')
                        obj = parts[2].strip('"')
                        triplets.append((subj, rel, obj))
                except Exception:
                    continue

        return triplets

    def build_knowledge_graph(self, triplets):
        """Build a NetworkX graph from triplets."""
        G = nx.DiGraph()

        # Add edges with relation as attribute
        for subj, rel, obj in triplets:
            G.add_edge(subj, obj, relation=rel)

        return G

    def analyze_knowledge_graph(self, G):
        """Analyze the knowledge graph structure."""
        # Basic statistics
        stats = {
            "nodes": len(G.nodes()),
            "edges": len(G.edges()),
            "density": nx.density(G),
            "connected_components": nx.number_weakly_connected_components(G),
        }

        # Find key entities (highest degree nodes)
        degree_centrality = nx.degree_centrality(G)
        top_entities = sorted(
            degree_centrality.items(), key=lambda x: x[1], reverse=True
        )[:20]

        # Find key relationships (most frequent)
        relation_counts = {}
        for _, _, data in G.edges(data=True):
            rel = data["relation"]
            relation_counts[rel] = relation_counts.get(rel, 0) + 1

        top_relations = sorted(
            relation_counts.items(), key=lambda x: x[1], reverse=True
        )[:20]

        return {
            "statistics": stats,
            "top_entities": top_entities,
            "top_relations": top_relations,
        }

    def save_knowledge_graph(self, G, output_dir="knowledge_graph_output"):
        """Save the graph in multiple formats with analysis."""
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Save in GraphML format for use with other tools
        nx.write_graphml(G, os.path.join(output_dir, "knowledge_graph.graphml"))

        # Save as CSV files for easier analysis
        edges = []
        for s, o, data in G.edges(data=True):
            edges.append({"subject": s, "relation": data["relation"], "object": o})

        edge_df = pd.DataFrame(edges)
        edge_df.to_csv(
            os.path.join(output_dir, "knowledge_graph_edges.csv"), index=False
        )

        nodes_df = pd.DataFrame({"node": list(G.nodes())})
        nodes_df.to_csv(
            os.path.join(output_dir, "knowledge_graph_nodes.csv"), index=False
        )

        # Save analysis
        analysis = self.analyze_knowledge_graph(G)

        with open(os.path.join(output_dir, "graph_analysis.txt"), "w") as f:
            f.write("Knowledge Graph Analysis\n")
            f.write("======================\n\n")

            f.write("Basic Statistics:\n")
            for stat, value in analysis["statistics"].items():
                f.write(f"- {stat}: {value}\n")

            f.write("\nTop Entities (by centrality):\n")
            for entity, score in analysis["top_entities"]:
                f.write(f"- {entity}: {score:.4f}\n")

            f.write("\nTop Relations (by frequency):\n")
            for relation, count in analysis["top_relations"]:
                f.write(f"- {relation}: {count}\n")

        return edge_df, nodes_df, analysis

    def visualize_knowledge_graph(
        self, G, output_dir="knowledge_graph_output", max_nodes=100
    ):
        """Visualize a subset of the knowledge graph."""
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Create community-based visualization
        if len(G) > max_nodes:
            # Use PageRank to identify important nodes
            pagerank = nx.pagerank(G)
            top_nodes = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)[
                :max_nodes
            ]
            nodes_to_keep = [node for node, _ in top_nodes]
            G_sub = G.subgraph(nodes_to_keep)
        else:
            G_sub = G

        plt.figure(figsize=(20, 20))

        # Use community detection for coloring (requires python-louvain package)
        try:
            import community as community_louvain

            partition = community_louvain.best_partition(nx.Graph(G_sub))
            node_colors = [partition[node] for node in G_sub.nodes()]
        except ImportError:
            node_colors = "lightblue"

        pos = nx.spring_layout(G_sub, seed=42)

        nx.draw(
            G_sub,
            pos,
            with_labels=True,
            node_color=node_colors,
            node_size=1500,
            arrowsize=20,
            font_size=12,
            cmap=plt.cm.tab20,
        )

        # Draw edge labels (limit to keep visualization clean)
        edge_labels = {}
        for i, (s, o, d) in enumerate(G_sub.edges(data=True)):
            if i < 50:  # Only show first 50 labels to reduce clutter
                edge_labels[(s, o)] = d["relation"]

        nx.draw_networkx_edge_labels(G_sub, pos, edge_labels=edge_labels, font_size=10)

        plt.title("Knowledge Graph Visualization (Key Entities)", fontsize=20)
        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, "knowledge_graph_visualization.png"), dpi=300
        )

        return os.path.join(output_dir, "knowledge_graph_visualization.png")

    def create_knowledge_graph_from_directory(
        self,
        directory_path,
        output_dir="knowledge_graph_output",
        file_extensions=["txt", "md", "csv"],
        n_processes=4,
    ):
        """Run the complete knowledge graph extraction pipeline on a directory of text files."""
        print(f"Starting knowledge graph creation from files in {directory_path}")

        # Step 1: Collect and chunk all text files
        print("Collecting and chunking text from files...")
        chunks = self.collect_and_process_all_files(
            directory_path, file_extensions=file_extensions, n_processes=n_processes
        )

        # Step 2: Extract triplets from all chunks
        print("Extracting relationship triplets...")
        triplets = self.process_chunks_for_triplets(chunks)

        # Step 3: Build the knowledge graph
        print("Building knowledge graph...")
        G = self.build_knowledge_graph(triplets)
        print(
            f"Knowledge graph created with {len(G.nodes())} nodes and {len(G.edges())} edges"
        )

        # Step 4: Save and analyze the graph
        print("Saving and analyzing knowledge graph...")
        edge_df, node_df, analysis = self.save_knowledge_graph(G, output_dir)

        # Step 5: Visualize a portion of the graph
        print("Creating visualization...")
        viz_path = self.visualize_knowledge_graph(G, output_dir)

        print(f"Knowledge graph processing complete! Output saved to {output_dir}")

        return G, edge_df, node_df, analysis, viz_path


if __name__ == "__main__":
    # Configuration
    channel_id = "UCgfe2ooZD3VJPB6aJAnuQng"  # Example channel ID
    transcript_dir = "transcripts"
    # api_key = "YOUR_API_KEY_HERE"  # Replace with your actual Google AI API key if using token counting

    # --- Workflow ---

    # 1. Initialize Managers
    transcript_manager = YouTubeTranscriptManager(output_dir=transcript_dir)
    # kg_builder = KnowledgeGraphBuilder(use_gpu=True) # Assuming GPU is desired

    # 2. Download Transcripts (Optional - Uncomment to run)
    print("--- Downloading Transcripts ---")
    transcript_manager.download_channel_transcripts(channel_id)
    print("\n")

    # 3. Clean Transcripts (Optional - Uncomment to run)
    print("--- Cleaning Transcripts ---")
    transcript_manager.clean_transcript_files()  # Cleans files in the default output_dir
    print("\n")

    # 4. Count Tokens (Optional - Uncomment and provide API Key to run)
    # print("--- Counting Tokens ---")
    # if 'api_key' in locals() and api_key != "YOUR_API_KEY_HERE":
    #     token_counts = count_tokens_in_directory(api_key=api_key, directory_path=transcript_dir)
    #     # print(token_counts) # Optional: print detailed counts
    # else:
    #     print("API Key not provided. Skipping token count.")
    # print("\n")

    # 5. Build Knowledge Graph
    # print("--- Building Knowledge Graph ---")
    # Build knowledge graph from the specified transcript directory
    # G, edges, nodes, analysis, viz_path = kg_builder.create_knowledge_graph_from_directory(
    #     transcript_dir,
    #     n_processes=4 # Adjust number of processes as needed
    # )

    # print(f"\n--- Workflow Complete ---")
    # print(f"Knowledge Graph output saved to: {os.path.abspath(kg_builder.output_dir)}") # Use output_dir from builder instance
    # print(f"Visualization saved to: {viz_path}")
