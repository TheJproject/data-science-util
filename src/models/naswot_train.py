import numpy as np
import nas_class as nas
from collections import namedtuple
import torch
from torch.utils.data import DataLoader, TensorDataset
import torchvision
import torchvision.transforms as transforms
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from network_to_tex import *
import plotly.express as px
import os
import wandb
import pickle


NUM_VERTICES = 32

ALLOWED_OPS = ['input', 'conv3x3-bn-relu', 'conv1x1-bn-relu', 'maxpool3x3', 'output']

transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

# Download and load the CIFAR-10 training dataset
train_dataset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(
    train_dataset, batch_size=32, shuffle=True, num_workers=2)

Args = namedtuple('Args', ['stem_out_channels', 'num_stacks', 'num_modules_per_stack', 'num_labels'])

args = Args(
    stem_out_channels=128,
    num_stacks=3,
    num_modules_per_stack=3,
    num_labels=10,
)


def counting_backward_hook(module, inp, out):
    module.visited_backwards = True


def hooklogdet(K, labels=None):
    s, ld = np.linalg.slogdet(K)
    return ld

def counting_forward_hook(network, module, inp, out):
    inp = inp[0].view(inp[0].size(0), -1)
    x = (inp > 0).float()  # binary indicator
    K = x @ x.t()
    K2 = (1. - x) @ (1. - x.t())
    network.K = network.K + K.cpu().numpy() + K2.cpu().numpy()  # hamming distance


searchspace = {
    'layer_types': ['conv', 'pool', 'identity'],
    'conv': {
        'num_filters': [16, 32, 64],
        'filter_size': [3, 5, 7],
        'stride': [1, 2]
    },
    'pool': {
        'type': ['max', 'average'],
        'pool_size': [2, 3],
        'stride': [1, 2]
    },
}
def is_valid_architecture(matrix, ops,generated_architectures, max_edges=9):
    # Check if the matrix is upper triangular
    if not np.all(np.tril(matrix, -1) == 0):
        return False

    # Check if the total number of edges is within the allowed limit
    if np.sum(matrix) > max_edges:
        return False

    # Check if there's a path from input to output vertex
    visited = set()

    def dfs(vertex):
        if vertex == len(matrix) - 1:
            return True
        visited.add(vertex)
        for i in range(len(matrix)):
            if matrix[vertex][i] == 1 and i not in visited:
                if dfs(i):
                    return True
        return False

    if not dfs(0):
        return False

    # Check if all vertices (except input and output) have at least one incoming and one outgoing edge
    for i in range(1, len(matrix) - 1):
        if np.sum(matrix[:, i]) == 0 or np.sum(matrix[i, :]) == 0:
            return False

    # Check if the operation labels are within the allowed range
    allowed_ops = ['input', 'conv3x3-bn-relu', 'conv1x1-bn-relu', 'maxpool3x3', 'output']
    if not all(op in allowed_ops for op in ops):
        return False
    # Check if the generated architecture has already been generated
    matrix_ops_hash = hash((matrix.tobytes(), tuple(ops)))
    if matrix_ops_hash in generated_architectures:
        return False
    return True

def generate_architecture_from_network(network, iteration):
    arch = [
        to_head('..'),
        to_cor(),
        to_begin()
    ]

    offset_x = 0
    for i, layer in enumerate(network.layers):
        layer_type = layer.__class__.__name__

        if layer_type in ['ConvBnRelu', 'Conv3x3BnRelu', 'Conv1x1BnRelu']:
            in_channels = layer.conv_bn_relu[0].in_channels
            out_channels = layer.conv_bn_relu[0].out_channels
            kernel_size = layer.conv_bn_relu[0].kernel_size[0]
            arch.append(
                to_Conv(f"conv{i+1}", kernel_size, out_channels, offset=f"({offset_x},0,0)", to=f"({offset_x},0,0)",
                        height=kernel_size, depth=kernel_size, width=2))
        elif layer_type == 'MaxPool3x3':
            arch.append(to_Pool(f"pool{i+1}", offset="(0,0,0)", to=f"(conv{i}-east)"))

        offset_x += 1

    arch.append(to_SoftMax("soft1", 10, f"({offset_x},0,0)", "(pool1-east)", caption="SOFT"))
    for i in range(1, len(network.layers)):
        arch.append(to_connection(f"pool{i}", f"conv{i + 1}"))

    arch.append(to_end())

    namefile = f"architecture_{iteration}"
    to_generate(arch, namefile + '.tex')

def evaluate_architecture(model_spec):
    network = nas.Network(model_spec, args, searchspace)

    network.K = np.zeros((NUM_VERTICES, NUM_VERTICES))
    for name, module in network.named_modules():
        if 'ReLU' in str(type(module)):
            module.register_backward_hook(counting_backward_hook)
            module.register_forward_hook(lambda module, inp, out: counting_forward_hook(network, module, inp, out))

    x, target = next(iter(trainloader))
    x2 = torch.clone(x)

    x, target = x, target
    network(x2)

    score = hooklogdet(network.K, target)
    print("ReLU based score")
    print(score)
    
    return score, network


num_architectures = 300
scores = []
generated_architectures = set()

for i in range(num_architectures):
    print(f"Architecture {i + 1}:")
    matrix, ops = nas.generate_random_architecture()
    while not is_valid_architecture(matrix, ops,generated_architectures):
        matrix, ops = nas.generate_random_architecture()
    matrix_ops_hash = hash((matrix.tobytes(), tuple(ops)))
    generated_architectures.add(matrix_ops_hash)
    print(matrix)
    print(ops)
    model_spec = nas.ModelSpec(matrix, ops)
    score, network = evaluate_architecture(model_spec)
    scores.append(score)
    print("\n")
    fig = px.imshow(network.K, 
                color_continuous_scale='Reds',
                title="K on example architecture with batch size 32")

    # Set the destination folder for WSL
    folder = "/mnt/c/Users/Jonah/OneDrive/Documents/Thesis/data-science-util/reports/figures"
    model_spec_path = os.path.join('/mnt/c/Users/Jonah/OneDrive/Documents/Thesis/nasbench/nasbench1/model-specs', f"model_spec_{i+1}.pickle")
    #save model_spec to pickle file
    with open(model_spec_path, 'wb') as f:
        pickle.dump(model_spec, f)
    # Check if the folder exists, if not create it
    if not os.path.exists(folder):
        os.makedirs(folder)

    # Save the image as a file
    image_filename = f"k_example_architecture_{i}_.png"
    image_path = os.path.join(folder, image_filename)
    fig.write_image(image_path)

print("Scores for all architectures:")
print(scores)
#pickle score
score_path = os.path.join('/mnt/c/Users/Jonah/OneDrive/Documents/Thesis/nasbench/nasbench1/', f"score.pickle")
with open(score_path, 'wb') as f:
    pickle.dump(scores, f)