# Code for running experiments on quantization
# Adapted from https://github.com/vincentherrmann/pytorch-wavenet


import argparse
import time

import librosa
import soundfile as sf
import torch.quantization

from audio_data import WavenetDataset
from wavenet_model import *
from wavenet_training import *

argparser = argparse.ArgumentParser()
argparser.add_argument(
    "--audio_filename", type=str, default="latest_generated_clip"
)
argparser.add_argument("--quantize_dynamic", type=bool, default=True)
argparser.add_argument(
    "--benchmark_filename", type=str, default="quantization_benchmarks.txt"
)
argparser.add_argument("--length", type=int, default=100000)

args = argparser.parse_args()

model = load_latest_model_from("snapshots", use_cuda=False)

print("model: ", model)
print("receptive field: ", model.receptive_field)
print("parameter count: ", model.parameter_count())

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
    print(str(200 * step // total_steps) + "% generated")


start = time.time()
generated = model.generate_fast(
    num_samples=args.length,
    first_samples=start_data,
    progress_callback=prog_callback,
    progress_interval=1000,
    temperature=1.0,
    regularize=0.0,
)
regular_generation_runtime = time.time() - start
sf.write(args.audio_filename + "_regular.wav", generated, samplerate=10000)

start = time.time()
quantized_model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Conv1d}, dtype=torch.qint8
)
quantization_runtime = time.time() - start

start = time.time()
quantized_generated = quantized_model.generate_fast(
    num_samples=args.length,
    first_samples=start_data,
    progress_callback=prog_callback,
    progress_interval=1000,
    temperature=1.0,
    regularize=0.0,
)
quantized_generation_runtime = time.time() - start

print(generated)


sf.write(args.audio_filename + "_quantized.wav", quantized_generated, samplerate=10000)


outfile = open(args.benchmark_filename, "a")
print("#### QUANTIZATION BENCHMARKS ###", file=outfile)
print(f"Time taken to Quantize: {quantization_runtime} seconds", file=outfile)
print(f"Regular generation time: {regular_generation_runtime} seconds", file=outfile)
print(
    f"Quantized generation time: {quantized_generation_runtime} seconds", file=outfile
)
print(
    f"Speedup: {regular_generation_runtime / quantized_generation_runtime}",
    file=outfile,
)
print("#### END QUANTIZATION BENCHMARKS ###", file=outfile)
