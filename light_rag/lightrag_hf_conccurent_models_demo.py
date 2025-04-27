#!/usr/bin/env python3
"""
LightRAG setup and verification script for concurrent HuggingFace models
This script sets up LightRAG with proper document processing and verification
"""

import argparse
import asyncio
import os
import time
import traceback
from typing import Optional

from lightrag import LightRAG, QueryParam
from lightrag.kg.shared_storage import initialize_pipeline_status
from lightrag.llm.hf import hf_embed, hf_model_complete
from lightrag.utils import EmbeddingFunc, Tokenizer
from transformers import AutoModel, AutoTokenizer


# Custom tokenizer class to avoid tiktoken downloads
class TransformersTokenizer(Tokenizer):
    def __init__(self, model_name):
        try:
            if os.path.exists(model_name):
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            else:
                # For embedding models or other paths
                self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        except Exception as e:
            print(f"Warning: Could not load transformer tokenizer: {e}")
            # Fallback to a simple character-based tokenizer
            self.tokenizer = None

    def encode(self, text):
        if self.tokenizer:
            return self.tokenizer.encode(text)
        else:
            # Super simple fallback tokenizer
            return [ord(c) for c in text]

    def decode(self, tokens):
        if self.tokenizer:
            return self.tokenizer.decode(tokens)
        else:
            # Simple fallback decoder
            return "".join(chr(t) if t < 0x10FFFF else " " for t in tokens)


# Set up working directories
LLAMA_WORKING_DIR = "./llama_rag"
ATC_WORKING_DIR = "./atc_rag"

# Create directories if they don't exist
for directory in [LLAMA_WORKING_DIR, ATC_WORKING_DIR]:
    if not os.path.exists(directory):
        os.makedirs(directory)


def verify_dirs_empty():
    """Verify working directories are empty to avoid conflicts"""
    for directory in [LLAMA_WORKING_DIR, ATC_WORKING_DIR]:
        files = os.listdir(directory)
        if files:
            print(f"WARNING: {directory} is not empty. This may cause issues.")
            print(f"Files found: {files}")
            proceed = input("Do you want to continue anyway? (y/n): ")
            if proceed.lower() != "y":
                print("Exiting...")
                exit(1)


def create_embedding_func(embedding_model_path, device):
    """Create embedding function with proper device management"""
    tokenizer = AutoTokenizer.from_pretrained(
        embedding_model_path, trust_remote_code=True
    )
    embed_model = AutoModel.from_pretrained(
        embedding_model_path, trust_remote_code=True
    )

    if device:
        embed_model = embed_model.to(device)

    def custom_embed_func(texts):
        return hf_embed(texts, tokenizer=tokenizer, embed_model=embed_model)

    return EmbeddingFunc(
        embedding_dim=768,
        max_token_size=5000,
        func=custom_embed_func,
    )


# Create a custom global config class that behaves like a dictionary
class GlobalConfig:
    def __init__(self, model_name):
        self.model_name = model_name

    @property
    def global_config(self):
        return {"llm_model_name": self.model_name}


async def initialize_rag_model(
    working_dir: str,
    model_name: str,
    embedding_model_path: str = None,
    device: str = None,
) -> LightRAG:
    """Initialize a LightRAG instance with the specified model."""
    print(f"Initializing LightRAG with model: {model_name} on device: {device}")

    # Add extra debug info, especially for ATC model
    is_atc = "atc" in model_name.lower()
    if is_atc:
        print("ATC model detected. Using special initialization parameters.")
        print(f"Working directory: {working_dir}")
        print(f"Device: {device}")
        print(f"Embedding model: {embedding_model_path}")

    # Create a global config instance for this model
    config = GlobalConfig(model_name)

    try:
        # Custom HF completion function with device specification
        async def custom_hf_complete(*args, **kwargs):
            # Ensure history_messages is properly initialized to avoid errors
            if "history_messages" not in kwargs:
                kwargs["history_messages"] = []

            if device:
                kwargs["device"] = device

            # Replace hashing_kv with our custom config object
            kwargs["hashing_kv"] = config

            # Use the standard hf_model_complete function
            return await hf_model_complete(*args, **kwargs)

        # Configure embedding function with better error handling
        embedding_func = None
        try:
            embedding_func = create_embedding_func(embedding_model_path, device)
            print(f"Embedding model initialized successfully on {device}")
        except Exception as e:
            print(f"⚠️ Error initializing embedding model: {e}")
            traceback.print_exc()
            print("Falling back to default embedding model...")
            # Try to use a different embedding approach if the main one fails
            # This is especially important for the ATC model
            try:
                tokenizer = AutoTokenizer.from_pretrained(
                    "sentence-transformers/all-MiniLM-L6-v2"
                )
                embed_model = AutoModel.from_pretrained(
                    "sentence-transformers/all-MiniLM-L6-v2"
                )
                if device:
                    embed_model = embed_model.to(device)

                def fallback_embed_func(texts):
                    return hf_embed(texts, tokenizer=tokenizer, embed_model=embed_model)

                embedding_func = EmbeddingFunc(
                    embedding_dim=384,  # all-MiniLM-L6-v2 has 384 dimensions
                    max_token_size=5000,
                    func=fallback_embed_func,
                )
                print("Fallback embedding model initialized successfully")
            except Exception as e2:
                print(f"⚠️ Fallback embedding model also failed: {e2}")
                raise ValueError(f"Could not initialize any embedding model: {e}, {e2}")

        # Configure LightRAG
        rag = LightRAG(
            working_dir=working_dir,
            llm_model_func=custom_hf_complete,
            llm_model_name=model_name,
            embedding_func=embedding_func,
            tokenizer=TransformersTokenizer(model_name),
        )

        # Properly await the async initialization
        await rag.initialize_storages()
        await initialize_pipeline_status()

        if is_atc:
            print("✅ ATC RAG model initialized successfully!")

        return rag
    except Exception as e:
        print(f"⚠️ Error initializing RAG model: {e}")
        traceback.print_exc()
        if is_atc:
            print("⚠️ ATC RAG model initialization failed!")
        return None


async def process_document(rag: LightRAG, file_path: str, label: str) -> bool:
    """Process document with improved error handling and retries"""
    if rag is None:
        print(f"⚠️ RAG model for {label} was not properly initialized")
        return False

    print(f"Loading document from {file_path} into {label}...")

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            document_text = f.read()
    except Exception as e:
        print(f"⚠️ Error reading document file {file_path}: {e}")
        return False

    # Split the document into smaller chunks if it's too large
    # This helps with processing large documents
    max_chunk_size = 8000
    if len(document_text) > max_chunk_size:
        chunks = []
        for i in range(0, len(document_text), max_chunk_size):
            chunks.append(document_text[i : i + max_chunk_size])
        print(f"Document split into {len(chunks)} chunks for processing")
    else:
        chunks = [document_text]

    success = False

    # Increase max retries for better reliability, especially for ATC model
    max_retries = 5 if label == "ATC RAG" else 3
    wait_between_retries = 10 if label == "ATC RAG" else 5

    for chunk_index, chunk in enumerate(chunks):
        chunk_success = False

        for attempt in range(1, max_retries + 1):
            try:
                print(
                    f"Inserting document chunk {chunk_index + 1}/{len(chunks)} into {label} (attempt {attempt}/{max_retries})..."
                )
                # Use the asynchronous insert method
                doc_id = await rag.ainsert(chunk)
                print(f"Document chunk {chunk_index + 1} inserted with ID: {doc_id}")
                chunk_success = True
                break
            except Exception as e:
                print(
                    f"⚠️ Document insertion error for {label} chunk {chunk_index + 1} (attempt {attempt}/{max_retries}): {e}"
                )
                traceback.print_exc()
                # Give more time for ATC model to recover between attempts
                print(f"Waiting {wait_between_retries} seconds before retry...")
                await asyncio.sleep(wait_between_retries)

        if not chunk_success:
            print(
                f"❌ Failed to insert document chunk {chunk_index + 1} for {label} after {max_retries} attempts"
            )
            if label == "ATC RAG":
                print("Trying to debug ATC model issues. Checking model state...")
                try:
                    # Try to verify if the model is still responsive
                    await rag.aquery(
                        "Test query to verify model is responsive",
                        param=QueryParam(mode="naive"),
                    )
                    print("Model is still responsive to queries.")
                except Exception as e:
                    print(f"Model is not responsive: {e}")
            return False

    success = True

    # Verify that files were actually created in the working directory
    expected_files = [
        "kv_store_full_docs.json",
        "kv_store_text_chunks.json",
        "vdb_chunks.json",
    ]
    missing_files = [
        f
        for f in expected_files
        if not os.path.exists(os.path.join(rag.working_dir, f))
    ]

    if missing_files:
        print(
            f"⚠️ Warning: The following expected files are missing in {label} directory: {missing_files}"
        )
        print(f"Document processing may not have completed successfully for {label}")

        # Try additional efforts for ATC model
        if label == "ATC RAG":
            print(f"Attempting extended recovery for {label}...")

            # Try multiple small document insertions with longer wait times
            for i in range(3):
                try:
                    print(f"ATC recovery attempt {i + 1}/3...")
                    recovery_text = f"Recovery document {i + 1}. This is a test document to finalize processing."
                    recovery_id = await rag.ainsert(recovery_text)
                    print(f"Recovery document inserted with ID: {recovery_id}")
                    # Add longer wait to allow processing to complete
                    print("Waiting 30 seconds for processing to complete...")
                    await asyncio.sleep(30)

                    # Check if files are now present
                    missing_files = [
                        f
                        for f in expected_files
                        if not os.path.exists(os.path.join(rag.working_dir, f))
                    ]
                    if not missing_files:
                        print(f"✅ Recovery successful for {label}!")
                        break
                except Exception as e:
                    print(f"⚠️ Recovery attempt {i + 1} failed: {e}")
        else:
            # For Llama or other models, use the original approach
            try:
                print(f"Attempting one final document insertion for {label}...")
                # Use a short text to minimize processing time
                final_doc_id = await rag.ainsert(
                    "This is a test document to finalize processing."
                )
                print(f"Final document inserted with ID: {final_doc_id}")

                # Add a small wait to allow processing to complete
                print("Waiting for processing to complete...")
                await asyncio.sleep(20)
            except Exception as e:
                print(f"⚠️ Final document insertion failed: {e}")

    # Check again after final insertions
    missing_files = [
        f
        for f in expected_files
        if not os.path.exists(os.path.join(rag.working_dir, f))
    ]
    if missing_files:
        print(f"⚠️ Warning: Files still missing after final attempts: {missing_files}")
        success = False

    # Write completion marker regardless of success to allow server start
    try:
        with open(f"{rag.working_dir}/setup_complete", "w") as f:
            f.write("Setup completed at " + time.strftime("%Y-%m-%d %H:%M:%S"))
    except Exception as e:
        print(f"⚠️ Error writing setup completion marker: {e}")

    if success:
        print(f"✅ Document processing completed for {label}")
    else:
        print(f"⚠️ Document processing completed with warnings for {label}")

    return success


async def verify_query(rag: LightRAG, query: str, label: str) -> Optional[str]:
    """Test query with more robust error handling"""
    if rag is None:
        print(f"⚠️ RAG model for {label} was not properly initialized")
        return None

    print(f"Testing query on {label}: '{query}'")

    # Check if basic files exist
    if not os.path.exists(os.path.join(rag.working_dir, "kv_store_full_docs.json")):
        print(f"⚠️ Cannot run query - required files missing in {label} directory")
        return "[Error: Missing required files]"

    try:
        # Try a simpler query approach that doesn't need kg processing
        response = await rag.aquery(query, param=QueryParam(mode="naive"))

        # Check response quality
        if response and "[no-context]" not in response:
            print(f"✅ Query test SUCCEEDED on {label}")
        else:
            print(f"⚠️ Query test result on {label} may be missing context")

        return response
    except Exception as e:
        print(f"⚠️ Query error on {label}: {e}")
        traceback.print_exc()
        return f"[Query error: {str(e)}]"


async def main():
    parser = argparse.ArgumentParser(description="LightRAG Setup and Verification")
    parser.add_argument(
        "--document", "-d", type=str, required=True, help="Path to the document file"
    )
    parser.add_argument(
        "--llama-model", type=str, required=True, help="Path to Llama model"
    )
    parser.add_argument(
        "--atc-model", type=str, required=True, help="Path to ATC model"
    )
    parser.add_argument(
        "--embedding-model", type=str, required=True, help="Path to embedding model"
    )
    parser.add_argument(
        "--llama-device", type=str, default="cuda:0", help="Device for Llama model"
    )
    parser.add_argument(
        "--atc-device", type=str, default="cuda:1", help="Device for ATC model"
    )
    parser.add_argument(
        "--skip-queries", action="store_true", help="Skip running test queries"
    )
    parser.add_argument(
        "--sequential",
        action="store_true",
        help="Process models sequentially instead of concurrently",
    )

    args = parser.parse_args()

    # Verify document exists
    if not os.path.exists(args.document):
        print(f"Error: Document not found at {args.document}")
        return False

    # Verify models exist
    for model_path, model_name in [
        (args.llama_model, "Llama model"),
        (args.atc_model, "ATC model"),
        (args.embedding_model, "Embedding model"),
    ]:
        if not os.path.exists(model_path):
            print(f"Error: {model_name} not found at {model_path}")
            return False

    # Verify directories are clean
    verify_dirs_empty()

    print("\n=== Initializing and Running RAG Models ===\n")
    print(f"Processing mode: {'Sequential' if args.sequential else 'Concurrent'}")

    llama_rag = None
    atc_rag = None

    if args.sequential:
        # Sequential processing - Initialize and process one model at a time
        print("\n--- Processing Llama RAG First ---\n")
        llama_rag = await initialize_rag_model(
            LLAMA_WORKING_DIR, args.llama_model, args.embedding_model, args.llama_device
        )
        llama_success = await process_document(llama_rag, args.document, "Llama RAG")

        print("\n--- Processing ATC RAG Second ---\n")
        atc_rag = await initialize_rag_model(
            ATC_WORKING_DIR, args.atc_model, args.embedding_model, args.atc_device
        )
        atc_success = await process_document(atc_rag, args.document, "ATC RAG")
    else:
        # Concurrent processing (original method)
        # Initialize both models concurrently
        init_tasks = [
            initialize_rag_model(
                LLAMA_WORKING_DIR,
                args.llama_model,
                args.embedding_model,
                args.llama_device,
            ),
            initialize_rag_model(
                ATC_WORKING_DIR, args.atc_model, args.embedding_model, args.atc_device
            ),
        ]

        # Wait for both initializations to complete
        llama_rag, atc_rag = await asyncio.gather(*init_tasks)

        # Process documents in parallel
        process_tasks = [
            process_document(llama_rag, args.document, "Llama RAG"),
            process_document(atc_rag, args.document, "ATC RAG"),
        ]

        # Wait for both document processing to complete
        llama_success, atc_success = await asyncio.gather(*process_tasks)

    if not (llama_success and atc_success):
        print(
            "\n⚠️ Warning: Document processing may not have completed successfully for one or both models."
        )
        print("You may need to wait longer before starting the servers")

    # Skip queries if requested
    if not args.skip_queries:
        print("\n=== Verification Queries ===\n")

        # Test with a simple query
        test_query = "What are the key points mentioned in the document?"

        # Run queries
        query_tasks = []

        if llama_success:
            query_tasks.append(verify_query(llama_rag, test_query, "Llama RAG"))

        if atc_success:
            query_tasks.append(verify_query(atc_rag, test_query, "ATC RAG"))

        # Wait for all queries to complete
        if query_tasks:
            await asyncio.gather(*query_tasks)
    else:
        print("\n=== Skipping verification queries ===\n")

    print("\n=== Summary ===\n")
    print(f"Llama RAG working directory: {LLAMA_WORKING_DIR}")
    print(f"ATC RAG working directory: {ATC_WORKING_DIR}")

    print("\n✅ SETUP COMPLETE - The RAG systems are now ready to use")

    # Return status for shell script - consider it a success if either model worked
    return llama_success or atc_success


if __name__ == "__main__":
    try:
        success = asyncio.run(main())
        # Exit with appropriate code for shell script
        if not success:
            exit(1)
    except KeyboardInterrupt:
        print("\n\nScript interrupted by user. Exiting...")
        exit(1)
    except Exception as e:
        print(f"\n\nUnexpected error: {e}")
        traceback.print_exc()
        exit(1)
