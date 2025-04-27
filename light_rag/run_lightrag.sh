#!/bin/bash

# Script to properly set up LightRAG instances
# This ensures documents are fully processed and knowledge graphs are built

# Colors for better output readability
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Clear previous RAG directories to start fresh
echo -e "${YELLOW}Cleaning up previous RAG directories...${NC}"
rm -rf ./llama_rag/
rm -rf ./atc_rag/

# Paths to models and document - adapt these to your local paths
LLAMA_MODEL="models/Llama-3.2-3B-Instruct"
ATC_MODEL="models/atc-llama"
EMBEDDING_MODEL="models/nomic-embed-text-v1.5"
DOCUMENT="atc.txt"
SKIP_QUERIES=false

# Parse command line options
while [[ $# -gt 0 ]]; do
  case $1 in
    --document)
      DOCUMENT="$2"
      shift 2
      ;;
    --llama-model)
      LLAMA_MODEL="$2"
      shift 2
      ;;
    --atc-model)
      ATC_MODEL="$2"
      shift 2
      ;;
    --embedding-model)
      EMBEDDING_MODEL="$2"
      shift 2
      ;;
    --skip-queries)
      SKIP_QUERIES=true
      shift
      ;;
    *)
      echo -e "${RED}Unknown option: $1${NC}"
      exit 1
      ;;
  esac
done

# Check for absolute paths, if not, make them relative to current directory
if [[ "$LLAMA_MODEL" != /* ]]; then
    LLAMA_MODEL="$(pwd)/$LLAMA_MODEL"
fi

if [[ "$ATC_MODEL" != /* ]]; then
    ATC_MODEL="$(pwd)/$ATC_MODEL"
fi

if [[ "$EMBEDDING_MODEL" != /* ]]; then
    EMBEDDING_MODEL="$(pwd)/$EMBEDDING_MODEL"
fi

# Check if document exists
if [ ! -f "$DOCUMENT" ]; then
    echo -e "${RED}Error: Document $DOCUMENT not found!${NC}"
    exit 1
fi

# Check if models exist
for MODEL_PATH in "$LLAMA_MODEL" "$ATC_MODEL" "$EMBEDDING_MODEL"; do
    if [ ! -d "$MODEL_PATH" ]; then
        echo -e "${RED}Error: Model directory $MODEL_PATH not found!${NC}"
        echo -e "${YELLOW}Current working directory: $(pwd)${NC}"
        echo -e "${YELLOW}Please make sure the model paths are correct.${NC}"
        exit 1
    fi
done

echo "=================================================================="
echo -e "${BLUE}Setting up LightRAG with proper document processing${NC}"
echo "=================================================================="
echo -e "Llama model: ${GREEN}$LLAMA_MODEL${NC}"
echo -e "ATC model: ${GREEN}$ATC_MODEL${NC}"
echo -e "Embedding model: ${GREEN}$EMBEDDING_MODEL${NC}"
echo -e "Document: ${GREEN}$DOCUMENT${NC}"
echo -e "Skip queries: ${GREEN}$SKIP_QUERIES${NC}"
echo "=================================================================="

# Run the initialization script
echo -e "${YELLOW}Starting initialization process. This may take several minutes...${NC}"
echo -e "${YELLOW}The script will initialize both models and process the document.${NC}"

# Build command with or without skip-queries
CMD="python examples/lightrag_hf_conccurent_models_demo.py \
    --document \"$DOCUMENT\" \
    --llama-model \"$LLAMA_MODEL\" \
    --atc-model \"$ATC_MODEL\" \
    --embedding-model \"$EMBEDDING_MODEL\" \
    --llama-device \"cuda:0\" \
    --atc-device \"cuda:1\" \
    --sequential"

if [ "$SKIP_QUERIES" = true ]; then
    CMD="$CMD --skip-queries"
fi

echo -e "${YELLOW}Running with trust_remote_code=True for embedding model...${NC}"

# Set environment variable to automatically trust remote code
export TRANSFORMERS_TRUST_REMOTE_CODE=1

# Run the command
eval $CMD

# Check if setup was successful
if [ $? -ne 0 ]; then
    echo -e "${RED}Setup failed. Please check the error messages above.${NC}"
    exit 1
fi

# Verify setup completion markers
if [ ! -f "./llama_rag/setup_complete" ] || [ ! -f "./atc_rag/setup_complete" ]; then
    echo -e "${YELLOW}Warning: Setup completion markers not found. Setup may not be complete.${NC}"
    echo -e "${YELLOW}You may need to wait longer before starting the servers.${NC}"
else
    echo -e "${GREEN}✅ Setup completion markers found.${NC}"
fi

echo ""
echo "=================================================================="
echo -e "${GREEN}Setup complete! 🎉${NC}"
echo -e "To start the LightRAG servers, run:"
echo ""
echo -e "${BLUE}./server_lightrag.sh${NC}"
echo -e "or with specific options:"
echo -e "${BLUE}lightrag-server --host 0.0.0.0 --port 9621 --working-dir ./llama_rag/${NC}"
echo -e "${BLUE}lightrag-server --host 0.0.0.0 --port 9622 --working-dir ./atc_rag/${NC}"
echo "==================================================================" 