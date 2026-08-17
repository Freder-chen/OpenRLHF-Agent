## Setting up Local Dense Retriever ([Search-R1](https://github.com/PeterGriffinJin/Search-R1) Local Search (WIKI))


This guide walks through setting up the **Search-R1 local dense retriever** backend and verifying it through a simple terminal test.

### 1. Create Retriever Conda Environment

Create and activate a clean environment with Python 3.10 and install dependencies:

```bash
# Create and activate environment
conda create -n retriever python=3.10 -y
conda activate retriever

# Install PyTorch with CUDA support
conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.1 -c pytorch -c nvidia -y

# Install retriever dependencies
pip install transformers datasets pyserini huggingface_hub

# Install FAISS-GPU
conda install faiss-gpu=1.8.0 -c pytorch -c nvidia -y

# Install web server packages
pip install uvicorn fastapi
```

### 2. Download Index and Corpus
Note: The local retrieval files are large. You'll need approximately 60-70 GB for download and 132 GB after extraction. Make sure you have sufficient disk space.
```bash
# Set your save path
save_path=./Index

# Download index and corpus
python examples/search/local_dense_retriever/download.py \
  --save_path $save_path

# Merge split index files into a single FAISS index
cat $save_path/part_* > $save_path/e5_Flat.index

# Decompress the corpus
gzip -d $save_path/wiki-18.jsonl.gz
```

You should now have:

- Index: `$save_path/e5_Flat.index`
- Corpus: `$save_path/wiki-18.jsonl`

### 3. Start Local Retrieval Server

```bash
# Activate retriever environment
conda activate retriever

# Set paths and retriever config
save_path=./Index
index_file=$save_path/e5_Flat.index
corpus_file=$save_path/wiki-18.jsonl
retriever_name=e5
retriever_path=intfloat/e5-base-v2

# Start the retrieval server (default port: 8000)
python examples/search/local_dense_retriever/retrieval_server.py \
  --index_path $index_file \
  --corpus_path $corpus_file \
  --topk 3 \
  --retriever_name $retriever_name \
  --retriever_model $retriever_path \
  --faiss_gpu
```

**Notes:**

- First startup downloads the model & loads the FAISS index (a few minutes)
- Subsequent startups: **~1–2 minutes**
- GPU memory: **≈5–7 GB / GPU**
- Process keeps running even if the shell closes
- To restart the server: `lsof -i :8000` to find the PID, then kill it and restart

### 4. Test the Local Search API

#### Option A — curl test

```bash
curl -X POST "http://localhost:8000/retrieve" \
  -H "Content-Type: application/json" \
  -d '{
    "queries": ["Java is?", "Python is?"],
    "topk": 2,
    "return_scores": true
  }'
```

If results return normally, the **Search-R1 local dense retriever is ready** 🚀
