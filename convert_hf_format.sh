MODEL_DIR="/home/code/index-tts/checkpoints/Index-TTS"
VLLM_DIR="$MODEL_DIR/vllm"
python convert_hf_format.py --model_dir "$MODEL_DIR"

echo "All operations completed successfully!"