#!/bin/bash

# Script to start LightRAG servers
# Run this after successfully running run_lightrag.sh

# Colors for better output readability
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default port and IP settings
LLAMA_PORT=9621
ATC_PORT=9622
HOST_IP="0.0.0.0"  # Use 0.0.0.0 to allow external connections
# HOST_IP="155.31.18.51"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --host)
      HOST_IP="$2"
      shift 2
      ;;
    --llama-port)
      LLAMA_PORT="$2"
      shift 2
      ;;
    --atc-port)
      ATC_PORT="$2"
      shift 2
      ;;
    *)
      echo -e "${RED}Unknown option: $1${NC}"
      exit 1
      ;;
  esac
done

# Check if the setup was completed
if [ ! -f "./llama_rag/setup_complete" ] && [ ! -f "./atc_rag/setup_complete" ]; then
    echo -e "${RED}Error: Setup appears incomplete. Please run run_lightrag.sh first!${NC}"
    echo -e "${BLUE}Command: ./run_lightrag.sh${NC}"
    exit 1
fi

# Working directories - check what actually exists
if [ -d "./llama_rag" ]; then
    LLAMA_DIR="./llama_rag"
else
    echo -e "${RED}Error: No Llama RAG directory found. Run run_lightrag.sh first!${NC}"
    exit 1
fi

if [ -d "./atc_rag" ]; then
    ATC_DIR="./atc_rag"
else
    echo -e "${YELLOW}Warning: No ATC RAG directory found. Only starting Llama server.${NC}"
    ATC_DIR=""
fi

# The embedding dimension for HuggingFace all-MiniLM-L6-v2 is 384
echo "=================================================================="
echo -e "${BLUE}Starting LightRAG servers${NC}"
echo "=================================================================="
echo -e "${GREEN}Using directories:${NC}"
echo -e "- Llama RAG: ${BLUE}$LLAMA_DIR${NC}"
if [ -n "$ATC_DIR" ]; then
    echo -e "- ATC RAG: ${BLUE}$ATC_DIR${NC}"
else
    echo -e "- ATC RAG: ${RED}Not available${NC}"
fi
echo "=================================================================="
echo -e "${GREEN}Server URLs:${NC}"
echo -e "Llama RAG: ${BLUE}http://$HOST_IP:$LLAMA_PORT/webui/${NC}"
if [ -n "$ATC_DIR" ]; then
    echo -e "ATC RAG: ${BLUE}http://$HOST_IP:$ATC_PORT/webui/${NC}"
fi
echo "=================================================================="

# Start Llama server directly
echo -e "${YELLOW}Starting Llama server on port $LLAMA_PORT...${NC}"
lightrag-server --working-dir "$LLAMA_DIR" --host "$HOST_IP" --port "$LLAMA_PORT"  --llm-binding ollama   --embedding-binding ollama &
LLAMA_PID=$!

# Start ATC server (if directory exists and has required files)
if [ -n "$ATC_DIR" ]; then
    # Check if ATC directory has necessary files
    REQUIRED_FILES=("kv_store_full_docs.json" "kv_store_text_chunks.json" "vdb_chunks.json")
    MISSING_FILES=()
    
    for file in "${REQUIRED_FILES[@]}"; do
        if [ ! -f "$ATC_DIR/$file" ]; then
            MISSING_FILES+=("$file")
        fi
    done
    
    # Warn about missing files but still try to start if setup_complete exists
    if [ ${#MISSING_FILES[@]} -gt 0 ]; then
        echo -e "${YELLOW}Warning: ATC RAG directory missing required files: ${MISSING_FILES[*]}${NC}"
        if [ -f "$ATC_DIR/setup_complete" ]; then
            echo -e "${YELLOW}Will try to start ATC server anyway since setup_complete marker exists${NC}"
            echo -e "${YELLOW}Starting ATC server on port $ATC_PORT...${NC}"
            lightrag-server --working-dir "$ATC_DIR" --host "$HOST_IP" --port "$ATC_PORT" --llm-binding ollama --embedding-binding ollama &
            ATC_PID=$!
        else
            echo -e "${RED}Cannot start ATC server - missing both files and setup marker${NC}"
            ATC_PID=""
        fi
    else
        echo -e "${YELLOW}Starting ATC server on port $ATC_PORT...${NC}"
        sleep 2
        lightrag-server --working-dir "$ATC_DIR" --host "$HOST_IP" --port "$ATC_PORT" --llm-binding ollama --embedding-binding ollama &
        ATC_PID=$!
    fi
else
    ATC_PID=""
fi

# Check if servers started successfully
sleep 3
if ! ps -p $LLAMA_PID > /dev/null; then
    echo -e "${RED}Error: Llama server failed to start!${NC}"
    if [ -n "$ATC_PID" ]; then
        kill $ATC_PID 2>/dev/null
    fi
    exit 1
fi

if [ -n "$ATC_PID" ] && ! ps -p $ATC_PID > /dev/null; then
    echo -e "${RED}Error: ATC server failed to start!${NC}"
    kill $LLAMA_PID 2>/dev/null
    exit 1
fi

if [ -n "$ATC_PID" ]; then
    echo -e "${GREEN}Both servers started successfully (PIDs: $LLAMA_PID, $ATC_PID)${NC}"
else
    echo -e "${GREEN}Llama server started successfully (PID: $LLAMA_PID)${NC}"
fi
echo -e "${YELLOW}Press Ctrl+C to stop all servers${NC}"

# Set trap to catch Ctrl+C and gracefully stop servers
function cleanup() {
    echo ""
    echo -e "${YELLOW}Stopping servers...${NC}"
    if [ -n "$ATC_PID" ]; then
        kill $LLAMA_PID $ATC_PID 2>/dev/null
        wait $LLAMA_PID $ATC_PID 2>/dev/null
    else
        kill $LLAMA_PID 2>/dev/null
        wait $LLAMA_PID 2>/dev/null
    fi
    echo -e "${GREEN}Servers stopped${NC}"
    exit 0
}
trap cleanup INT TERM

echo ""
echo "=================================================================="
echo -e "${BLUE}API Information:${NC}"
echo -e "Llama RAG API: ${GREEN}http://$HOST_IP:$LLAMA_PORT/api/v1/query${NC}"
if [ -n "$ATC_DIR" ]; then
    echo -e "ATC RAG API: ${GREEN}http://$HOST_IP:$ATC_PORT/api/v1/query${NC}"
fi
echo "=================================================================="

# Wait for servers to complete or be interrupted
if [ -n "$ATC_PID" ]; then
    wait $LLAMA_PID $ATC_PID
else
    wait $LLAMA_PID
fi 