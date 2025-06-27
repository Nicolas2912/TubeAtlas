import argparse
import os
import re
import sqlite3
import time

import networkx as nx
import spacy
import torch

# load .env
from dotenv import load_dotenv
from google import genai

# Import GenerateContentConfig
from google.genai.types import GenerateContentConfig
from tqdm import tqdm
from transformers import pipeline

load_dotenv()


class KnowledgeGraphBuilderGoogle:
    def __init__(self, model_name: str, yt_channel: str) -> None:
        load_dotenv()

        self.api_key = os.getenv("GEMINI_API_KEY")
        self.model_name = model_name
        self.yt_channel = yt_channel

    def _system_prompt(self):
        # formulate a good system prompt for a LLM to build a correct and accurate knowledge graph
        system_prompt = f"""
        You are a specialized knowledge graph extraction system designed to convert unstructured text into a structured knowledge graph. Your task is to identify entities and their relationships from the provided text with high precision and accuracy.

        # OBJECTIVE
        Extract meaningful (subject, relation, object) triplets from the input text that accurately represent the knowledge contained within. Focus on factual information rather than opinions or hypothetical statements.

        # INPUT FORMAT
        You will receive text chunks that may contain YouTube video transcript content. These chunks may include domain-specific terminology, technical concepts, and natural language explanations.

        # OUTPUT FORMAT
        Respond ONLY with a list of knowledge triplets in the following format:
        (subject, relation, object), (subject, relation, object), ...

        For example:
        ("neural networks", "are used for", "deep learning"), ("GPT", "is a type of", "language model")

        # EXTRACTION GUIDELINES
        1. ENTITIES (subjects and objects):
           - Extract specific, well-defined entities
           - Normalize entity names (e.g., "LLMs" → "large language models")
           - Preserve technical terminology exactly as presented
           - Include relevant qualifiers when they change the meaning

        2. RELATIONS:
           - Use clear, concise relation phrases
           - Standardize similar relations (e.g., "is part of", "belongs to" → "is part of")
           - Ensure relations accurately represent the text's meaning
           - Use active voice when possible

        3. VALIDITY RULES:
           - Each triplet must be factually supported by the text
           - Avoid extracting opinions unless explicitly attributed
           - Do not infer relationships not directly stated or strongly implied
           - Ensure logical consistency between triplets

        # PRIORITY EXTRACTION TARGETS
        - Technical concepts and their definitions
        - Hierarchical relationships (is-a, part-of)
        - Causal relationships (causes, enables, prevents)
        - Functional relationships (used-for, capable-of)
        - Temporal relationships (precedes, follows)
        - Comparative relationships (similar-to, different-from)

        # QUALITY CONTROL
        - Avoid overly generic triplets that don't add meaningful information
        - Avoid redundant triplets that express the same relationship
        - DO NOT REPEAT THE SAME RELATIONSHIP BETWEEN THE SAME ENTITY. DO NOT WRITE DUPLICATES!
        - Prioritize precision over recall - it's better to extract fewer high-quality triplets than many low-quality ones
        - Ensure extracted triplets would be meaningful even without the original context

        # SPECIAL HANDLING
        - For YouTube content: Pay special attention to the main topics discussed, key concepts explained, and factual claims made by the speaker
        - For technical content: Accurately capture technical relationships, specifications, and methodologies
        - For definitions: Express them as (term, "is defined as", definition) or (term, "refers to", meaning)

        Remember, your goal is to create a knowledge graph that accurately represents the factual content of the input text in a structured, machine-readable format.

        """
        return system_prompt

    def _load_transcripts(self):
        """
        Load transcripts from the database.

        Returns:
            list: List of transcripts
        """
        # Load transcripts from the data directory
        db_name = os.path.join("data", f"{self.yt_channel}.db")
        conn = sqlite3.connect(db_name)
        cursor = conn.cursor()
        cursor.execute("SELECT title, transcript_text FROM transcripts")
        results = cursor.fetchall()
        conn.close()

        # Print titles and return only transcript texts
        for title, _ in results:
            print(f"Processing transcript: {title}")
        transcripts = [(text,) for _, text in results]
        if not transcripts:
            print(f"Warning: No transcripts found in {db_name}")
            return []
        return transcripts

    def build_kg(self, batch_size_chars=2048):
        """
        Use the gemini 2.0 Flash model for building a KG, processing the transcript in a sliding window and enforcing stricter triple extraction.
        """
        import time

        client = genai.Client(api_key=self.api_key)

        # get transcripts
        transcripts = self._load_transcripts()
        transcripts = transcripts[:1]
        start_response_time = time.time()
        for transcript in transcripts:
            text = (
                transcript[0] if isinstance(transcript, (list, tuple)) else transcript
            )
            triples_set = set()
            for i in tqdm(range(0, len(text), batch_size_chars)):
                window = text[i : i + batch_size_chars]
                response = client.models.generate_content(
                    model=self.model_name,
                    contents=window,
                    config=GenerateContentConfig(
                        system_instruction=self._system_prompt(),
                        temperature=0.2,
                        max_output_tokens=10000,
                    ),
                )
                triples_text = response.text.strip()
                # Basic post-processing: parse triples and filter
                if triples_text.startswith("[") and triples_text.endswith("]"):
                    triples_text = triples_text[1:-1]
                triples = re.findall(r"\((?:[^()]|\([^()]*\))*\)", triples_text)
                for triple in triples:
                    parts = [p.strip(' "') for p in triple.strip("()").split(",")]
                    if len(parts) == 3:
                        subj, rel, obj = parts
                        # Filter: skip if any entity is empty, generic, or too long
                        if (
                            not subj
                            or not obj
                            or len(subj.split()) > 3
                            or len(obj.split()) > 3
                        ):
                            continue
                        # Optionally split multi-word entities (if not proper noun)
                        if " " in subj and subj.lower() not in [
                            "openai",
                            "chatgpt",
                            "python",
                            "gemini",
                            "claude",
                        ]:
                            continue
                        if " " in obj and obj.lower() not in [
                            "openai",
                            "chatgpt",
                            "python",
                            "gemini",
                            "claude",
                        ]:
                            continue
                        triples_set.add((subj, rel, obj))
            print(
                f"Response time: {round(time.time() - start_response_time, 2)} seconds"
            )
            # Output and visualize
            triples_list = list(triples_set)
            print(triples_list)
            # Optionally, save to DB here
            # Visualize KG using the list of triples directly
            self._visualize_kg(triples_list)
            break

    def _fix_transcript(self, transcript):
        """
        Fix typos and grammar using a LLM. Process the entire transcript in batchsizes of 5000 tokens.
        Make sure the model does only output the corrected text with no additional text in the limit of 5000 tokens.
        """

        # use LLM to fix typos
        system_instruction = """
            You are a text editor that corrects typos and grammar in the given text.
            Only output the corrected text with no additional text.
        """
        fixed_text = ""
        client = genai.Client(api_key=self.api_key)
        # Process transcript in batches of 5000 tokens
        for i in tqdm(range(0, len(transcript), 5000)):
            batch = transcript[i : i + 5000]
            response = client.models.generate_content(
                model=self.model_name,
                contents=batch,
                config=GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=0.1,
                    max_output_tokens=10000,
                ),
            )
            fixed_text += response.text

        return fixed_text

    def _visualize_kg(self, kg_text, output_file="knowledge_graph.html"):
        """
        Enhanced interactive visualization of a knowledge graph using Plotly.
        - Hover on nodes/edges for details
        - Color nodes by type (subject/object/both)
        - Improved label readability
        - Legend for node types
        - Pan/zoom/drag enabled
        """
        import re
        from collections import defaultdict

        import networkx as nx
        import plotly.graph_objects as go

        # Accept triples list or parse from string representation
        if isinstance(kg_text, str):
            import ast

            try:
                triplets = ast.literal_eval(kg_text)
            except Exception as e:
                print(f"Warning: failed to parse triples string: {e}")
                triplets = []
        elif isinstance(kg_text, list):
            triplets = kg_text
        else:
            print("Warning: triples input is not a list or string")
            triplets = []
        G = nx.DiGraph()
        for subject, relation, obj in triplets:
            G.add_node(subject)
            G.add_node(obj)
            G.add_edge(subject, obj, label=relation)
        pos = nx.spring_layout(G, seed=42)
        # Node types and color coding
        node_types = defaultdict(set)
        for s, r, o in triplets:
            node_types[s].add("subject")
            node_types[o].add("object")
        node_type_color = {"subject": "#1f77b4", "object": "#ff7f0e", "both": "#2ca02c"}
        node_colors = []
        node_hovertexts = []
        for node in G.nodes():
            t = node_types[node]
            if "subject" in t and "object" in t:
                color = node_type_color["both"]
                label = "subject & object"
            elif "subject" in t:
                color = node_type_color["subject"]
                label = "subject"
            else:
                color = node_type_color["object"]
                label = "object"
            node_colors.append(color)
            node_hovertexts.append(
                f"<b>{node}</b><br>Type: {label}<br>Degree: {G.degree(node)}"
            )
        # Edge traces and label traces
        edge_traces = []
        edge_label_traces = []
        for u, v, data in G.edges(data=True):
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            edge_traces.append(
                go.Scatter(
                    x=[x0, x1],
                    y=[y0, y1],
                    line=dict(width=2, color="#888"),
                    hoverinfo="none",
                    mode="lines",
                )
            )
            mid_x, mid_y = (x0 + x1) / 2, (y0 + y1) / 2
            edge_label_traces.append(
                go.Scatter(
                    x=[mid_x],
                    y=[mid_y],
                    text=[data["label"]],
                    mode="text",
                    hovertext=[
                        f"<b>{u}</b> → <b>{v}</b><br><b>Relation:</b> {data['label']}"
                    ],
                    hoverinfo="text",
                    textfont=dict(size=12, color="#333"),
                    showlegend=False,
                )
            )
        node_trace = go.Scatter(
            x=[pos[node][0] for node in G.nodes()],
            y=[pos[node][1] for node in G.nodes()],
            mode="markers+text",
            text=[node for node in G.nodes()],
            textposition="top center",
            hoverinfo="text",
            hovertext=node_hovertexts,
            marker=dict(
                size=22,
                color=node_colors,
                line=dict(width=2, color="#222"),
                symbol="circle",
                opacity=0.85,
            ),
            showlegend=False,
        )
        # Legend for node types
        legend_traces = []
        for label, color in node_type_color.items():
            legend_traces.append(
                go.Scatter(
                    x=[None],
                    y=[None],
                    mode="markers",
                    marker=dict(size=18, color=color),
                    legendgroup=label,
                    name=label.capitalize(),
                )
            )
        fig = go.Figure(
            data=legend_traces + edge_traces + edge_label_traces + [node_trace],
            layout=go.Layout(
                title=f"Knowledge Graph Visualization ({len(G.nodes())} nodes, {len(G.edges())} relationships)",
                showlegend=True,
                legend=dict(
                    x=0.85,
                    y=0.99,
                    bgcolor="rgba(255,255,255,0.8)",
                    bordercolor="#444",
                    borderwidth=1,
                    font=dict(size=13),
                ),
                hovermode="closest",
                margin=dict(b=40, l=5, r=5, t=60),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                height=900,
                annotations=[
                    dict(
                        text=f"Total Nodes: {len(G.nodes())} | Total Relationships: {len(G.edges())}",
                        showarrow=False,
                        xref="paper",
                        yref="paper",
                        x=0.01,
                        y=0.01,
                        font=dict(size=14, color="#888"),
                    )
                ],
            ),
        )
        fig.add_annotation(
            text="Tip: Hover nodes/edges for info. Drag to explore. Zoom with mousewheel.",
            xref="paper",
            yref="paper",
            x=0.5,
            y=-0.08,
            showarrow=False,
            font=dict(size=13, color="#444"),
            align="center",
        )
        fig.write_html(output_file, include_plotlyjs="cdn", full_html=True)
        print(f"Knowledge graph visualization saved to {output_file}")
        return output_file


class KnowledgeGraphBuilderLocal:
    def __init__(self, yt_channel, use_gpu=True):
        """
        Initializes the KG builder with a local LLM using transformers AutoModel.
        Args:
            yt_channel (str): The YouTube channel identifier.
            use_gpu (bool): Whether to attempt using the GPU (via device_map='auto').
            Note: lmdeploy usually auto-detects GPU. This flag might become less relevant.
        """
        # self.model_name = "unsloth/Qwen2.5-7B-bnb-4bit" # Old model
        self.model_name = "unsloth/Qwen2.5-7B-bnb-4bit"  # New model from request
        print(f"Loading local model with lmdeploy: {self.model_name}...")

        # Determine device mapping - lmdeploy pipeline typically handles this
        # device_map = "auto" if use_gpu and torch.cuda.is_available() else "cpu"
        # self.device = device_map if device_map != "auto" else (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        # print(f"Attempting to use device map: {device_map} (Resolved to: {self.device})")
        print(f"Using GPU: {use_gpu and torch.cuda.is_available()}")  # Informational

        try:
            # --- LMDeploy Pipeline Initialization ---
            # Configure backend (TurboMind is default and usually fastest)
            # Adjust session_len as needed, e.g., based on max context + max new tokens
            backend_config = TurbomindEngineConfig(session_len=8192)  # Example value

            # Load the pipeline
            self.pipe = pipeline(self.model_name, backend_config=backend_config)
            print(f"LMDeploy pipeline for {self.model_name} loaded successfully.")
            # --- Remove transformers model/tokenizer loading ---
            # self.model = AutoModelForCausalLM.from_pretrained(
            #     self.model_name,
            #     torch_dtype="auto",
            #     device_map=device_map
            # )
            # self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            # print(f"Model {self.model_name} and tokenizer loaded successfully.")

            # --- Remove Chat Template setting (lmdeploy handles it) ---
            # self.tokenizer.chat_template = (
            #     "{% for message in messages %}"
            #     "{{'\\n' + message['role'] + '\\n' + message['content'] + '\\n'}}"
            #     "{% endfor %}"
            #     "{% if add_generation_prompt %}"
            #     "{{ '\\nassistant\\n' }}"
            #     "{% endif %}"
            # )

        except Exception as e:
            print(f"Error loading model {self.model_name} with LMDeploy: {e}")
            print(
                "Please ensure the model name is correct, dependencies (like 'unsloth') are installed, and you have sufficient hardware resources (VRAM)."
            )
            raise

        self.triplets = []
        self.graph = nx.DiGraph()  # Reset graph
        self.yt_channel = yt_channel
        # Use the data directory for transcript DB files
        self.db_path = os.path.join("data", f"{self.yt_channel}.db")

    def _system_prompt(self):
        # formulate a good system prompt for a LLM to build a correct and accurate knowledge graph
        system_prompt = f"""
        You are a specialized knowledge graph extraction system designed to convert unstructured text into a structured knowledge graph. Your task is to identify entities and their relationships from the provided text with high precision and accuracy.

        # OBJECTIVE
        Extract meaningful (subject, relation, object) triplets from the input text that accurately represent the knowledge contained within. Focus on factual information rather than opinions or hypothetical statements.

        # INPUT FORMAT
        You will receive text chunks that may contain YouTube video transcript content. These chunks may include domain-specific terminology, technical concepts, and natural language explanations.

        # OUTPUT FORMAT
        Respond ONLY with a list of knowledge triplets in the following format:
        (subject, relation, object), (subject, relation, object), ...

        For example:
        ("neural networks", "are used for", "deep learning"), ("GPT", "is a type of", "language model")

        # EXTRACTION GUIDELINES
        1. ENTITIES (subjects and objects):
           - Extract specific, well-defined entities
           - Normalize entity names (e.g., "LLMs" → "large language models")
           - Preserve technical terminology exactly as presented
           - Include relevant qualifiers when they change the meaning

        2. RELATIONS:
           - Use clear, concise relation phrases
           - Standardize similar relations (e.g., "is part of", "belongs to" → "is part of")
           - Ensure relations accurately represent the text's meaning
           - Use active voice when possible

        3. VALIDITY RULES:
           - Each triplet must be factually supported by the text
           - Avoid extracting opinions unless explicitly attributed
           - Do not infer relationships not directly stated or strongly implied
           - Ensure logical consistency between triplets

        # PRIORITY EXTRACTION TARGETS
        - Technical concepts and their definitions
        - Hierarchical relationships (is-a, part-of)
        - Causal relationships (causes, enables, prevents)
        - Functional relationships (used-for, capable-of)
        - Temporal relationships (precedes, follows)
        - Comparative relationships (similar-to, different-from)

        # QUALITY CONTROL
        - Avoid overly generic triplets that don't add meaningful information
        - Avoid redundant triplets that express the same relationship
        - DO NOT REPEAT THE SAME RELATIONSHIP BETWEEN THE SAME ENTITY. DO NOT WRITE DUPLICATES!
        - Prioritize precision over recall - it's better to extract fewer high-quality triplets than many low-quality ones
        - Ensure extracted triplets would be meaningful even without the original context

        # SPECIAL HANDLING
        - For YouTube content: Pay special attention to the main topics discussed, key concepts explained, and factual claims made by the speaker
        - For technical content: Accurately capture technical relationships, specifications, and methodologies
        - For definitions: Express them as (term, "is defined as", definition) or (term, "refers to", meaning)

        Remember, your goal is to create a knowledge graph that accurately represents the factual content of the input text in a structured, machine-readable format.

        """
        return system_prompt

    def _generate_text(self, prompt_text, max_new_tokens=1024, temperature=0.3):
        """Generates text using the loaded lmdeploy pipeline."""
        # if not hasattr(self, 'model') or not hasattr(self, 'tokenizer'): # Replaced with pipe check
        if not hasattr(self, "pipe"):
            # print("Model or tokenizer not initialized. Cannot generate text.")
            print("LMDeploy pipeline not initialized. Cannot generate text.")
            return ""

        system_prompt_content = self._system_prompt()

        # Prepare messages for chat template (lmdeploy pipeline takes this format)
        messages = [
            {"role": "system", "content": system_prompt_content},
            {"role": "user", "content": prompt_text},
        ]

        try:
            # --- LMDeploy Inference ---
            # Apply chat template - Handled internally by lmdeploy pipeline
            # text = self.tokenizer.apply_chat_template(...)

            # Prepare model inputs - Handled internally by lmdeploy pipeline
            # model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

            # Configure generation parameters
            gen_config = GenerationConfig(
                max_new_tokens=max_new_tokens,
                temperature=(
                    temperature if temperature > 0 else 0.01
                ),  # lmdeploy might require temp > 0 for sampling
                top_p=0.8,  # Example, adjust if needed
                top_k=50,  # Example, adjust if needed
                do_sample=True if temperature > 0 else False,
                # pad_token_id handling might be internal to lmdeploy
            )

            # Generate response using lmdeploy pipeline
            start_time = time.time()
            # lmdeploy pipeline typically takes messages or prompts
            # response_obj = self.pipe(prompt_text, gen_config=gen_config) # If taking raw prompt
            response_obj = self.pipe(
                messages, gen_config=gen_config
            )  # If taking message list
            print(f"Response time: {round(time.time() - start_time, 2)} seconds")

            # Extract the generated text (lmdeploy response structure might vary)
            # Check lmdeploy documentation for exact response format
            response = (
                response_obj.text
                if hasattr(response_obj, "text")
                else str(response_obj)
            )

            # --- Remove transformers decoding ---
            # Decode the generated tokens, excluding the input tokens
            # input_ids_len = model_inputs.input_ids.shape[1]
            # generated_ids_only = generated_ids[:, input_ids_len:]
            # response = self.tokenizer.batch_decode(generated_ids_only, skip_special_tokens=True)[0]

            # print(f"\n--- Generated Content ---\n{response}\n-------------------------")
            return response

        except Exception as e:
            print(f"Error during text generation with LMDeploy: {e}")
            import traceback

            traceback.print_exc()  # Print detailed traceback
            return ""

    def _parse_triplets(self, text):
        # Parse the triplets from the response text
        triplets = []
        pattern = r'\("([^"]+)", "([^"]+)", "([^"]+)"\)'
        matches = re.findall(pattern, text)

        for match in matches:
            subject, relation, obj = match
            triplets.append((subject, relation, obj))

        return triplets

    def build_kg(self, batch_size_chars=2048):
        """Builds the knowledge graph from transcript text using the local LLM."""
        if not hasattr(self, "pipe"):  # Replaced with pipe check
            print("LMDeploy pipeline not initialized. Cannot build KG.")
            return

        print(f"Processing transcript for KG extraction using {self.model_name}...")
        self.triplets = []  # Reset triplets for new build
        self.graph = nx.DiGraph()  # Reset graph

        transcripts = self._load_transcripts()
        if not transcripts:
            print("No transcripts found in the database. Cannot build KG.")
            return

        print(f"Processing {len(transcripts)} transcripts...")
        for idx, transcript in enumerate(transcripts):
            print(f"\nProcessing transcript {idx + 1}/{len(transcripts)}...")
            # Generate triplets for the transcript
            num_batches = (
                len(transcript) + batch_size_chars - 1
            ) // batch_size_chars  # Calculate number of batches
            for i in range(0, len(transcript), batch_size_chars):
                batch_num = i // batch_size_chars + 1
                print(f"  Processing batch {batch_num}/{num_batches}...")
                batch_text = transcript[i : min(i + batch_size_chars, len(transcript))]
                if not batch_text.strip():  # Skip empty batches
                    print(f"    Skipping empty batch {batch_num}.")
                    continue
                # Generate triplets for the batch
                # print(f"\n--- Sending Batch (Chars: {len(batch_text)}) ---\n{batch_text[:200]}...\n-------------------------")
                # Pass the user prompt directly to _generate_text
                response_text = self._generate_text(
                    batch_text, max_new_tokens=5000, temperature=0.1
                )
                print(f"Response: {response_text}")

                if response_text:
                    # print(f"\n--- Received Response ---\n{response_text[:200]}...\n----------------------")
                    batch_triplets = self._parse_triplets(response_text)
                    print(
                        f"    Extracted {len(batch_triplets)} triplets from batch {batch_num}."
                    )
                    self.triplets.extend(batch_triplets)
                else:
                    print(
                        f"    Warning: No response generated for batch {batch_num}."
                    )  # Keep warning for no response

        # Visualize only if triplets were generated
        if self.triplets:
            output_filename = f"knowledge_graph_{self.yt_channel}.html"
            self._visualize_kg(output_file=output_filename)
        else:
            print("No triplets were extracted, skipping visualization.")

        return self.triplets

    def _visualize_kg(self, output_file="knowledge_graph.html"):
        """
        Visualize a knowledge graph using Plotly.

        Args:
            output_file (str): The HTML file to save the visualization to

        Returns:
            str: Path to the saved HTML file
        """
        import random
        from collections import defaultdict

        import networkx as nx
        import plotly.graph_objects as go

        # Create a directed graph
        G = nx.DiGraph()

        # Add nodes and edges
        for subject, relation, obj in self.triplets:
            if subject not in G:
                G.add_node(subject)
            if obj not in G:
                G.add_node(obj)
            G.add_edge(subject, obj, label=relation)

        # Use a spring layout for node positioning
        pos = nx.spring_layout(G, seed=42)

        # Create edge traces
        edge_traces = []
        edge_info = []

        # Group edges by source-target pair to avoid overlapping edges
        edge_groups = defaultdict(list)
        for u, v, data in G.edges(data=True):
            edge_groups[(u, v)].append(data["label"])

        for (u, v), relations in edge_groups.items():
            x0, y0 = pos[u]
            x1, y1 = pos[v]

            # Join multiple relations with a newline for display
            relation_text = "<br>".join(relations)

            # Create a trace for the edge
            edge_trace = go.Scatter(
                x=[x0, None, x1, None],
                y=[y0, None, y1, None],
                line=dict(width=1.5, color="#888"),
                hoverinfo="none",
                mode="lines",
            )
            edge_traces.append(edge_trace)

            # Create a trace for the edge label
            # Position the label at the midpoint of the edge
            mid_x = (x0 + x1) / 2
            mid_y = (y0 + y1) / 2

            edge_info.append(
                {
                    "x": mid_x,
                    "y": mid_y,
                    "text": relation_text,
                    "source": u,
                    "target": v,
                }
            )

        # Create node trace
        node_trace = go.Scatter(
            x=[pos[node][0] for node in G.nodes()],
            y=[pos[node][1] for node in G.nodes()],
            mode="markers+text",
            text=list(G.nodes()),
            textposition="top center",
            hoverinfo="text",
            marker=dict(
                showscale=True,
                colorscale="YlGnBu",
                size=15,
                colorbar=dict(thickness=15, title="Node Connections", xanchor="left"),
                line_width=2,
            ),
        )

        # Color nodes by their degree
        node_adjacencies = []
        for node in G.nodes():
            node_adjacencies.append(len(list(G.neighbors(node))))

        node_trace.marker.color = node_adjacencies

        # Create edge label trace
        edge_label_trace = go.Scatter(
            x=[info["x"] for info in edge_info],
            y=[info["y"] for info in edge_info],
            mode="text",
            text=[info["text"] for info in edge_info],
            textposition="middle center",
            hovertext=[
                f"{info['source']} → {info['target']}<br>{info['text']}"
                for info in edge_info
            ],
            hoverinfo="text",
            textfont=dict(size=10, color="#555"),
        )

        # Create figure
        fig = go.Figure(
            data=edge_traces + [node_trace, edge_label_trace],
            layout=go.Layout(
                title=f"Knowledge Graph Visualization ({len(G.nodes())} concepts, {len(self.triplets)} relationships)",
                showlegend=False,
                hovermode="closest",
                margin=dict(b=20, l=5, r=5, t=40),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                height=800,
                annotations=[
                    dict(
                        text=f"Total Nodes: {len(G.nodes())}<br>Total Relationships: {len(self.triplets)}",
                        showarrow=False,
                        xref="paper",
                        yref="paper",
                        x=0.01,
                        y=0.01,
                        font=dict(size=12),
                    )
                ],
            ),
        )

        # Save to HTML file
        fig.write_html(output_file, include_plotlyjs="cdn")

        print(f"Knowledge graph visualization saved to {output_file}")
        return output_file

    def _load_transcripts(self):
        """Loads transcripts from the local storage."""
        # Load transcripts from SQLite DB
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT transcript_text FROM transcripts")
        transcripts = [row[0] for row in cursor.fetchall() if row[0] is not None]
        conn.close()
        if not transcripts:
            print(f"Warning: No valid (non-NULL) transcripts found in {self.db_path}")
        return transcripts


def main():
    parser = argparse.ArgumentParser(
        description="Build Knowledge Graphs from YouTube Transcripts."
    )
    parser.add_argument(
        "--model_type",
        type=str,
        choices=["local", "google"],
        default="google",  # Default to local model
        help='Specify the model type to use: "local" or "google".',
    )
    parser.add_argument(
        "--cpu_only",
        action="store_true",
        help="Force using CPU even if a GPU is available.",
    )
    parser.add_argument(
        "--yt_channel",
        type=str,
        required=False,
        help="YouTube channel name to build KG for.",
        default="AndrejKarpathy",
    )
    args = parser.parse_args()
    # ------------------------

    builder = None  # Initialize builder to None

    # --- Instantiate based on argument ---
    if args.model_type == "local":
        try:
            builder = KnowledgeGraphBuilderLocal(
                yt_channel=args.yt_channel, use_gpu=not args.cpu_only
            )
        except Exception as e:
            print(f"\n--- Error initializing local model: {e} ---")
            print(
                "Please ensure the model name is correct, dependencies (like 'unsloth') are installed, and you have sufficient hardware resources (VRAM)."
            )
            raise

    elif args.model_type == "google":
        API_KEY_GOOGLE = os.getenv("GEMINI_API_KEY")
        if not API_KEY_GOOGLE:
            print(
                "\n--- Error: GEMINI_API_KEY not found in .env file. Cannot use Google model. ---"
            )
        else:
            try:
                builder = KnowledgeGraphBuilderGoogle(
                    model_name="gemini-2.0-flash-001", yt_channel=args.yt_channel
                )
            except Exception as e:
                print(f"\n--- Error initializing Google model: {e} ---")
    # -------------------------------------

    # --- Run the selected builder ---
    if builder:
        try:
            builder.build_kg(batch_size_chars=2048)
        except Exception as e:
            print(
                f"\n--- An error occurred during KG building with {args.model_type} model: {e} ---"
            )
    else:
        print("\nNo model builder was successfully initialized. Exiting.")
    # -------------------------------

    print("\nKnowledge Graph Builder finished.")


if __name__ == "__main__":
    main()
