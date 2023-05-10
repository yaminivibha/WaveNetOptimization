# Code for running experiments on quantization
# Portions Adapted from https://github.com/vincentherrmann/pytorch-wavenet

import argparse
import time

import librosa
import soundfile as sf
import torch.quantization

from audio_data import WavenetDataset
from wavenet_model import *
from wavenet_training import *
import warnings

warnings.filterwarnings("ignore", category=UserWarning) 


# Setting up arguments for experimentation
argparser = argparse.ArgumentParser()
argparser.add_argument(
    "--audio_filename", type=str, default="latest_generated_clip"
)
argparser.add_argument("--quantize_dynamic", type=bool, default=True)
# argparser.add_argument(
#     "--benchmark_filename", type=str, default="quantization_benchmarks.txt"
# )
argparser.add_argument("--sample_length", type=int, default=50000)
argparser.add_argument("--generate_original", type=bool, default=False)
args = argparser.parse_args()

# loading latest model from snapshot
model = load_latest_model_from("snapshots", use_cuda=False)

print("model: ", model)
print("receptive field: ", model.receptive_field)
print("parameter count: ", model.parameter_count())

# loading data
data = WavenetDataset(
    dataset_file="train_samples/bach_chaconne/dataset.npz",
    item_length=model.receptive_field + model.output_length - 1,
    target_length=model.output_length,
    file_location="train_samples/bach_chaconne",
    test_stride=20,
)
print("the dataset has " + str(len(data)) + " items")

start_data = data[250000][0]
start_data = torch.max(start_data, 0)[1]

def prog_callback(step, total_steps):
    print(str(100 * step // total_steps) + "% generated")

def generate_audio(model):
    generated = model.generate_fast(
        num_samples=args.sample_length,
        first_samples=start_data,
        progress_callback=prog_callback,
        progress_interval=1000,
        temperature=1.0,
        regularize=0.0,
    )
    return generated

# Constructing benchmarking file
print("#### QUANTIZATION BENCHMARKS ###", )

# Generating non-quantized audio
if args.generate_original:
    start = time.time()
    generated = generate_audio(model)
    regular_generation_runtime = time.time() - start

    print(generated)

    sf.write(args.audio_filename + "_regular.wav", generated, samplerate=10000)
    print(f"Regular generation time: {regular_generation_runtime} seconds")

# Running for dynamic quantization
if args.quantize_dynamic:
    start = time.time()
    quantized_model = torch.quantization.quantize_dynamic(
        model, {torch.nn.Conv1d}, dtype=torch.qint8
    )
    quantization_runtime = time.time() - start
    print(f"Time taken to Dynamically Quantize: {quantization_runtime} seconds")
    print(generated)

    start = time.time()
    quantized_generated = generate_audio(quantized_model)
    quantized_generation_runtime = time.time() - start
    sf.write(args.audio_filename + "_quantized.wav", quantized_generated, samplerate=10000)
    print(
    f"Dynamically Quantized audio generation time: {quantized_generation_runtime} seconds"
    )

if args.generate_original and args.quantize_dynamic:
    print(
        f"Speedup: {regular_generation_runtime / quantized_generation_runtime}",
        
    )

print("#### END QUANTIZATION BENCHMARKS ###")
