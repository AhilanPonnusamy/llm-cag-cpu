
# vLLM Setup and CAG(Cache Augmented Generation) Chatbot on MacBook Pro (CPU-Only)

## 1. Setup vLLM on MacBook Pro (CPUs Only)

### Prerequisites
- Clone this repository and move in to the folder.
- Install Python 3.12 (latest version is **not** compatible with vLLM):

```bash
brew install python@3.12
```

### Create and Activate Virtual Environment

```bash
python3.12 -m venv vllm-openai-venv
source vllm-openai-venv/bin/activate
```

### Install Requirements and vLLM

```bash
git clone https://github.com/vllm-project/vllm.git
cd vllm
```
```bash
pip install -r ../requirements.txt

# Set environment variables for CPU build
export VLLM_TARGET_DEVICE=cpu
export VLLM_BUILD_WITH_CUDA=0

# Install vLLM in editable mode
pip install -e .
```

### Start the vLLM Server

```bash
export VLLM_CPU_KVCACHE_SPACE=16 # KV Cache in GBs
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxx  # Replace with your Hugging Face token

VLLM_USE_CUDA=0 python -m vllm.entrypoints.openai.api_server \
  --model microsoft/Phi-3-mini-128k-instruct \
  --dtype float32 \
  --max-model-len 16384 \
  --max-num-seqs 1 \
  --swap-space 8 \
  --hf-token $HF_TOKEN
```

#### Parameter Explanations:

- `--model`: Specifies the Hugging Face model to load. (`microsoft/Phi-3-mini-128k-instruct` supports long context).
- `--dtype`: Precision used for model weights (`float32` for CPU).
- `--max-model-len`: Max tokens (prompt + generation) per request (e.g., `16384` tokens).
- `--max-num-seqs`: Number of concurrent requests; `1` is safest for CPU use.
- `--swap-space`: Disk space (in GB) for offloading KV cache if RAM is limited.
- `--hf-token`: Hugging Face token for gated model access.

---

## 2. Run the CAG Chatbot App

### Launch the Streamlit App

Open a new terminal, activate the virtual environment, and run:

```bash
source ../vllm-openai-venv/bin/activate
cd src
python -m streamlit run app.py
```

### Test Instructions

1. Ask a random question to verify the app is working and connected to the local LLM.
2. Click the **Browse files** button and upload `MCP-intro.pdf` from the `files/` directory.
3. Ask a document-related question like:

   ```
   What is a Model Context Protocol server?
   ```

4. Check the vLLM terminal — you should see that the prompt now includes the PDF content.
5. Continue with more questions like:

   ```
   Can you provide some examples of Data and File systems MCP servers?
   ```

Observe how the CAG chatbot uses the uploaded document to answer contextually.

---

✅ **Done!** You now have a working CPU-based vLLM backend connected to a Streamlit-based local chatbot app with document context.
