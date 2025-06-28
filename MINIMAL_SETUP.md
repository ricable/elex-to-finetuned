# Minimal Flow4 CLI Setup

This document describes the minimal setup for Flow4 CLI without the complex Augmentoolkit server infrastructure.

## Quick Setup

```bash
# Set up virtual environment
uv venv
source .venv/bin/activate

# Install core dependencies
uv pip install -r requirements.txt

# Install minimal Augmentoolkit support
uv pip install openai

# Optional: For MLX support on Apple Silicon
uv pip install mlx mlx-lm
```

## Available Commands

### Basic Pipeline
```bash
# Convert HTML files to Markdown
python src/run_flow4.py --verbose convert --input data/ --output-dir output

# Chunk documents
python src/run_flow4.py --verbose chunk --input output --output-dir chunks

# Complete pipeline (convert + chunk)
python src/run_flow4.py --verbose pipeline --input data/ --output-dir output
```

### Advanced Dataset Generation
```bash
# Generate QA datasets using simplified Augmentoolkit
python src/run_flow4.py --verbose generate --input chunks --output-dir augmentoolkit_output

# With MLX model (Apple Silicon)
python src/run_flow4.py --verbose generate --input chunks --output-dir augmentoolkit_output --model mlx-community/Llama-3.2-3B-Instruct-4bit
```

### MLX Fine-tuning (Apple Silicon only)
```bash
# Fine-tune using generated datasets
python src/run_flow4.py finetune --dataset augmentoolkit_output/mlx_dataset.jsonl --model mlx-community/Llama-3.2-3B-Instruct-4bit

# With interactive chat
python src/run_flow4.py finetune --dataset augmentoolkit_output/mlx_dataset.jsonl --model mlx-community/Llama-3.2-3B-Instruct-4bit --chat
```

## Key Features

✅ **CLI-only**: No web interface or server setup required  
✅ **Minimal dependencies**: Works with basic Python packages  
✅ **Fallback modes**: Graceful degradation when optional dependencies missing  
✅ **Apple Silicon optimized**: MLX support for M1/M2/M3 processors  
✅ **Demo mode**: Works without OpenAI API key for testing  

## Output Structure

```
output/
├── markdown/                 # Converted HTML files
├── chunks/                   # Document chunks (JSON format)
├── augmentoolkit_output/     # Generated datasets
│   ├── mlx_dataset.jsonl     # Ready for MLX fine-tuning
│   ├── augmentoolkit_dataset.json
│   └── generation_summary.json
└── pipeline_summary.json     # Processing statistics
```

## Dependencies

### Core (always required)
- `beautifulsoup4` - HTML parsing
- `lxml` - XML processing  
- `html2text` - HTML to Markdown conversion
- `tiktoken` - Tokenization
- `pyyaml` - Configuration files

### Optional
- `openai` - For advanced dataset generation via API
- `mlx` + `mlx-lm` - Apple Silicon optimization (M1/M2/M3 only)
- `docling` - Advanced PDF/HTML processing (if available)

## Tested Functionality

✅ HTML to Markdown conversion (97 files processed)  
✅ Document chunking (6,968 chunks created)  
✅ Simplified Augmentoolkit generation (3 QA pairs from 3 chunks)  
✅ MLX dataset format output (ready for fine-tuning)  
✅ CLI help and command structure  

## Next Steps

1. **For OpenAI API users**: Set `OPENAI_API_KEY` environment variable
2. **For MLX users**: Ensure Apple Silicon Mac with sufficient RAM
3. **For production**: Consider scaling chunk processing and model selection
4. **For fine-tuning**: Use generated MLX datasets with appropriate model sizes

This minimal setup provides full CLI functionality without the complex server infrastructure while maintaining compatibility with advanced features when needed.