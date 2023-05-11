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


# Setting up arguments for experimentation
argparser = argparse.ArgumentParser()
argparser.add_argument(
    "--audio_filename", type=str, default="latest_generated_clip"
)
argparser.add_argument("--generate_original", action="store_true", help="Generate non-quantized audio")
argparser.add_argument("--quantize_dynamic", action="store_true", help="Quantize model dynamically")
argparser.add_argument("--quantize_static", action="store_true", help="Quantize model statically")
argparser.add_argument("--model", "-m", type=str, default="latest", help="Model to use")
argparser.add_argument("--sample_length", type=int, default=50000, help="Length of generated sample")
args = argparser.parse_args()

# loading latest model from snapshot
if(args.model == "latest"):
    model = load_latest_model_from("snapshots", use_cuda=False)
else:
    model = load_to_cpu("snapshots/" + args.model)

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
    print(f"Non-quantized audio generation (inference) time: {regular_generation_runtime} seconds")

# Running for dynamic quantization
if args.quantize_dynamic:
    start = time.time()
    quantized_model = torch.quantization.quantize_dynamic(
        model, {torch.nn.Conv1d}, dtype=torch.qint8
    )
    quantization_runtime = time.time() - start
    print(f"Time taken to Dynamically Quantize: {quantization_runtime} seconds")

    param_size = 0
    for param in model.parameters():
        if param.data.is_floating_point():
            param_size += param.numel() * torch.finfo(param.data.dtype).bits
        else:
            param_size += param.numel() * torch.iinfo(param.data.dtype).bits
    buffer_size = 0
    for buffer in model.buffers():
        if buffer.data.is_floating_point():
            buffer_size += buffer.numel() * torch.finfo(buffer.data.dtype).bits
        else:
            buffer_size += buffer.numel() * torch.iinfo(buffer.data.dtype).bits

    size_all_mb = (param_size + buffer_size) / 1024**2
    print('Quantized model size: {:.3f}MB'.format(size_all_mb))

    start = time.time()
    quantized_generated = generate_audio(quantized_model)
    quantized_generation_runtime = time.time() - start
    sf.write(args.audio_filename + "_quantized.wav", quantized_generated, samplerate=10000)
    print(
    f"Dynamically Quantized audio generation (inference) time: {quantized_generation_runtime} seconds"
    )

if args.generate_original and args.quantize_dynamic:
    print(
        f"Speedup: {regular_generation_runtime / quantized_generation_runtime}",
        
    )

# Running for Static Quantization
if args.quantize_static:
    model.eval()
    model.qconfig = torch.ao.quantization.get_default_qconfig('x86')
    torch.quantization.prepare(model, inplace=True)
    
    start = time.time()
    quantized_model = torch.quantization.convert(model)
    quantization_runtime = time.time() - start
    print(f"Time taken to Statically Quantize: {quantization_runtime} seconds")

    start = time.time()
    quantized_generated = generate_audio(quantized_model)
    quantized_generation_runtime = time.time() - start
    sf.write(args.audio_filename + "_quantized.wav", quantized_generated, samplerate=10000)
    print(
    f"Statically Quantized audio generation time (inference): {quantized_generation_runtime} seconds"
    )

print("#### END QUANTIZATION BENCHMARKS ###")
