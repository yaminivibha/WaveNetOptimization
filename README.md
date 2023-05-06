# Optimizing Wavenet
benchmarking + optimizing WaveNet, a GNN for generating raw audio
Using AIHWToolkit, Analog AI testing toolkit open-sourced by IBM

Project slides available here
IEEE report available here

#### Yamini Ananth yva2002, Stan Liao

## Setup

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

## Setting up Wavenet with AIHWToolkit
- updated to use inputs.squeeze() in queue.enqueue function
- changed from deprecated librosa to modern soundfile.write(soundrate=16000)

## References


## Attributions
