mkdir original_audio
mkdir dynamic_audio
mkdir static_audio

python3 quantization.py --audio_filename original_audio/generated_sample --generate_original --model=="latest" --sample_length=30000 --trials 100 > original_benchmarks.txt
python3 quantization.py --audio_filename dynamic_audio/generated_sample  --generate_original --model=="latest" --sample_length=30000 --trials 100 > dynamic_benchmarks.txt
python3 quantization.py --audio_filename static_audio/generated_sample   --generate_original --model=="latest" --sample_length=30000 --trials 100 > static_benchmarks.txt