# Optimizing Wavenet
benchmarking + optimizing WaveNet, a GNN for generating raw audio
Using AIHWToolkit, Analog AI testing toolkit open-sourced by IBM

Project slides available here

IEEE report available here

#### Yamini Ananth yva2002, Stan Liao

## VM Specs and Environment Setup

Create a Linux VM on GCP with the following specifications:
- N1-highmem2 (13GB)
- 1 T1 GPU
- 100 GB boot disk running Deep Learning for Linux with CUDA 11.03 pre-installed

To setup the environment, execute the following:

```
$ virtualenv wavenet_env
$ source wavenet_env/bin/activate
$ git clone https://github.com/yaminivibha/WaveNetOptimization.git
$ cd WaveNetOptimization
$ bash setup.sh
```


## Running VM after initial setup
```
$ source wavenet_env/bin/activate
$ pip uninstall nvidia_cublas_cu11
$ cd WaveNetOptimization
```

## Setting up Wavenet with AIHWToolkit
- updated to use inputs.squeeze() in queue.enqueue function in `wavenet_model.py` as described [here](https://github.com/vincentherrmann/pytorch-wavenet/issues/21)
- changed from deprecated librosa to modern soundfile.write(soundrate=16000) in `generate_samples.py`

## References
- pytorch=wavenet repo on GitHub (for base WaveNet code)


## Attributions
