# from https://github.com/szagoruyko/functional-zoo

#from graphviz import Digraph
# import torch
# from torch.autograd import Variable
import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import wave
import sys


# shows the sound waves
def visualize(dir: str, file: str):
    path = os.path.join(dir, file)
    # reading the audio file
    raw = wave.open(path)
     
    # reads all the frames
    # -1 indicates all or max frames
    signal = raw.readframes(-1)
    signal = np.frombuffer(signal, dtype ="int16")
     
    # gets the frame rate
    f_rate = raw.getframerate()
 
    # to Plot the x-axis in seconds
    # you need get the frame rate
    # and divide by size of your signal
    # to create a Time Vector
    # spaced linearly with the size
    # of the audio file
    time = np.linspace(
        0, # start
        len(signal) / f_rate,
        num = len(signal)
    )
 
    # creating new figure and labels
    plt.figure(1)
    plt.title(file)
    plt.xlabel("Time")
    
    # actual plotting
    plt.plot(time, signal)
    plt.savefig(file+"_wave.png")
    print(f"saved figure for {file}")


# def make_dot(var, params):
#     """ Produces Graphviz representation of PyTorch autograd graph

#     Blue nodes are the Variables that require grad, orange are Tensors
#     saved for backward in torch.autograd.Function

#     Args:
#         var: output Variable
#         params: dict of (name, Variable) to add names to node that
#             require grad (TODO: make optional)
#     """
#     param_map = {id(v): k for k, v in params.items()}

#     node_attr = dict(style='filled',
#                      shape='box',
#                      align='left',
#                      fontsize='12',
#                      ranksep='0.1',
#                      height='0.2')
#     dot = Digraph(node_attr=node_attr, graph_attr=dict(size="12,12"))
#     seen = set()

#     def size_to_str(size):
#         return '(' + (', ').join(['%d' % v for v in size]) + ')'

#     def add_nodes(var):
#         if var not in seen:
#             #if torch.is_tensor(var):
#             #    dot.node(str(id(var)), size_to_str(var.size()), fillcolor='orange')
#             if hasattr(var, 'variable'):
#                 u = var.variable
#                 node_name = '%s\n %s' % (param_map[id(u)], size_to_str(u.size()))
#                 dot.node(str(id(var)), node_name, fillcolor='lightblue')
#             else:
#                 dot.node(str(id(var)), str(type(var).__name__).replace('Backward', ''))
#             seen.add(var)
#             if hasattr(var, 'next_functions'):
#                 for u in var.next_functions:
#                     if u[0] is not None:
#                         dot.edge(str(id(u[0])), str(id(var)))
#                         add_nodes(u[0])
#             if hasattr(var, 'saved_tensors'):
#                 for t in var.saved_tensors:
#                     dot.edge(str(id(t)), str(id(var)))
#                     add_nodes(t)

#     add_nodes(var.grad_fn)
#     return dot


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--dir_path", type=str, default=None)
    argparser.add_argument("--file_name", type=str, default=None)

    args = argparser.parse_args()
    if args.dir_path:
        for file in os.listdir(args.dir_path):
            if file.endswith('.wav'):
                print(file)
                visualize(args.dir_path, file)

if __name__ == '__main__':
    main()