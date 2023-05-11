# Optimizing Wavenet Inference using Caching and Quantization
Project slides available here
IEEE report available here

#### Yamini Ananth yva2002, Stan Liao

## Submitted Files
```
├── WaveNetOptimization
│   ├── snapshots
│   │   └── chaconne_model_2017-12-28_16-44-12 <-- pretrained CPU model
│   └──train_samples
│       └──bach_chaconne
│            └──dataset.npz                    <-- bach-chaconne training dataset
├──_Paper.pdf                                  <-- IEEE report
├── README.md                                  <-- You're here now!
├── audio_data.py                              <-- utils for audio processing
├── experiments.sh                             <-- bash script for running quantization experiments
├── generate_script.py                         <-- script for running one inference
├── model_logging.py                           <-- utils for logging during training
├── optimizers.py                              <-- model utils
├── quantization.py                            <-- utils for testing dynamic and static quantization
├── setup.sh                                   <-- bash setup script
├── train_script.py                            <-- training vanilla CPU Wavenet model
├── train_script_static.py                     <-- training CPU Wavenet model with static quantization stubs
├── wavenet_model.py                           <-- model utils for vanilla wavenet
├── wavenet_model_static.py                    <-- model utils for static quantization version
├── wavenet_modules.py                         <-- model utils
└── wavenet_training.py                        <-- model training utils
```

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

This will take care of installing all required packages and removing unused libraries from the defauly CUDA installation. 

## Running Experiments
If you wish to replicate portions of the experiments run in the paper, the following command is recommended
```
$ nohup bash experiments.sh
```
This calls on the script `quantization.py` with varying parameters in order to obtain the relevant results regarding speedup for `fast_generate`. The switches available include:
```
$ python3 quantization.py 
optional arguments:
  -h, --help            show this help message and exit
  --audio_filename AUDIO_FILENAME
  --generate_original   Generate non-quantized audio
  --quantize_dynamic    Quantize model dynamically
  --quantize_static     Quantize model statically
  --model MODEL, -m MODEL
                        Model to use
  --sample_length SAMPLE_LENGTH
                        Length of generated sample
```

To measure CPU utilization, run a single call of `quantization.py` using the `top` command in Linux, for instance:
```
$ top python3 quantization.py --dynamic_quantization 
```

## Results 

### Fast Generation

1. We use a pretrained CPU model and apply dynamic quantization. This way, we have the exact same model both before and after quantization. For the purposes of benchmarking, we generate 50,000 samples which equates to roughly 2 seconds of audio. In practice, ideally many more seconds of audio would be generated for real-world examples; however, the goal here was simply to benchmark inference.
2. For both the non quantized and quantized models, we generate 100 samples, timing the inference on each step and ensuring that all cores are synchronized.

| CPU cores | CPU memory | Mean generate runtime (secs) over 100 trials | Mean fast_generate runtime (secs) over 100 trials | Speedup  |
| --- | --- | --- | --- | --- |
| 2 vCPU | 13GB | 150.98 s | 118.88 s | 1.27002018843 |
| 4 vCPU | 26GB | 144.16 s | 113.35 s | 1.27181296868 |
| 8 vCPU | 52GB | 145.88 s | 115.83 s | 1.2594319261 |

In our model, we have a relatively small number of layers (10). According to le Paine et al, the fast generation method outperform the regular generate method most intensely for values of L=12 and above. Thus, we can observe a small quantity of speedup, however effects of speedup would be more prominent with more layers. 

We see not much speedup after scaling to 4 CPUs (and, in fact, very little even from 2→4). As mentioned below, our machine only fully utilizes 2 CPUs at a time. Thus, while we have some speedup moving from 2 → 4, adding additional CPUs does not lead to speedup.

However, it is possible that if we did hyperparameter optimization on our model to include more dataloader workers and a higher batch size, we could see some more speedup, which could be made possible with more memory.

### Comparing static and dynamic quantization

| Model Type | Mean per-epoch training time (hr) | Mean post-training  quantization - time to convert  (s) over 100 trials | Model Size (MB) | Mean fast_generate runtime (s) over 100 trials  | CPU utilization of fast_generate (%) |
| --- | --- | --- | --- | --- | --- |
| No Quantization | 9.22 | — | 6.998 | 118.293 | 200 |
| Post Training Dynamic Quantization | — | 11.09 | 6.998 | 100.234 | 200 |
| Post Training Static Quantization | — | 16.30 | 3.526 | 98.938 | 200 |

In order to implement static quantization, it was necessary to add quantization stubs to the WaveNet constructor and retrain on the Bach-Chaconne dataset. Because quantization on PyTorch is not supported over GPU, we unfortunately could not accelerate training using parallelization, distributed training, or the accelerated compute power of the GPU. Thus, trained for only one epoch; however, due to the dense convolutional layers and sample density in the training data, this process still took approximately 6.23 hours.

To measure model size, we specifically considered the size of the model parameters, inputs, and intermediate values (eg values, gradients). To compute the CPU Utilization of fast_generate for each model, we use the ***top*** Linux command after running our generation in the background. Our values are above 100% because they indicate usage compared to one core, while our machine had 4 cores; ergo, a CPU utilization of 200% would indicate a true utilization of 50%.

## Acknowledgements
Thank you to Professor Kauotar and Professor Parijat Dube for holding the course COMS6998 High Performance Machine Learning at Columbia University in Spring 2023. We learned a great deal from the course and will carry the experience through future professional work and research. 

## References
Paine, T. L., Khorrami, P., Chang, S., Zhang, Y., Ramachandran, P., Hasegawa-Johnson, M. A., & Huang, T. S. (2016). Fast Wavenet Generation Algorithm. *ArXiv [Cs.SD]*. Retrieved from http://arxiv.org/abs/1611.09482

van den Oord, A., Dieleman, S., Zen, H., Simonyan, K., Vinyals, O., Graves, A., … Kavukcuoglu, K. (2016). WaveNet: A Generative Model for Raw Audio. *ArXiv [[Cs.SD](Http://Cs.Sd/)]*. Retrieved from http://arxiv.org/abs/1609.03499

Herrmann, V. (2017). Open Source PyTorch Wavenet implementation. *Github*.

Krishnamoorthi, R. (2020, March 26). *Quantization in PyTorch*. PyTorch. Retrieved May 11, 2023, from https://pytorch.org/blog/introduction-to-quantization-on-pytorch/

Kuchaiev, O., Ginsburg, B., Gitman, I., Lavrukhin, V., Li, J., Nguyen, H., … Micikevicius, P. (2018). Mixed-Precision Training for NLP and Speech Recognition with OpenSeq2Seq. *ArXiv [Cs.CL]*. Retrieved from http://arxiv.org/abs/1805.10387

Pednekar, S., Krishnadas, A., Cho, B., & Makris, N. C. (2023). Weber’s Law of perception is a consequence of resolving the intensity of natural scintillating light and sound with the least possible error. *Proceedings of the Royal Society A: Mathematical, Physical and Engineering Sciences*, *479*(2271), 20220626. doi:10.1098/rspa.2022.0626
