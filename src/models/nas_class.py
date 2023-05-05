from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import math
import random 

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBnRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0):
        super(ConvBnRelu, self).__init__()

        self.conv_bn_relu = nn.Sequential(
            #nn.ReLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            #nn.ReLU(inplace=True)
            nn.ReLU()
        )

    def forward(self, x):
        return self.conv_bn_relu(x)

class Conv3x3BnRelu(nn.Module):
    """3x3 convolution with batch norm and ReLU activation."""
    def __init__(self, in_channels, out_channels):
        super(Conv3x3BnRelu, self).__init__()

        self.conv3x3 = ConvBnRelu(in_channels, out_channels, 3, 1, 1)

    def forward(self, x):
        x = self.conv3x3(x)
        return x

class Conv1x1BnRelu(nn.Module):
    """1x1 convolution with batch norm and ReLU activation."""
    def __init__(self, in_channels, out_channels):
        super(Conv1x1BnRelu, self).__init__()

        self.conv1x1 = ConvBnRelu(in_channels, out_channels, 1, 1, 0)

    def forward(self, x):
        x = self.conv1x1(x)
        return x

class MaxPool3x3(nn.Module):
    """3x3 max pool with no subsampling."""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(MaxPool3x3, self).__init__()

        self.maxpool = nn.MaxPool2d(kernel_size, stride, padding)
        #self.maxpool = nn.AvgPool2d(kernel_size, stride, padding)

    def forward(self, x):
        x = self.maxpool(x)
        return x

# Commas should not be used in op names
OP_MAP = {
    'conv3x3-bn-relu': Conv3x3BnRelu,
    'conv1x1-bn-relu': Conv1x1BnRelu,
    'maxpool3x3': MaxPool3x3
}
OP_MAP_INV = {
    0: 'conv1x1-bn-relu',
    1: 'conv3x3-bn-relu',
    2: 'maxpool3x3'
}
class Network(nn.Module):
    def __init__(self, spec, args, searchspace=[]):
        super(Network, self).__init__()

        self.layers = nn.ModuleList([])

        in_channels = 3
        out_channels = args.stem_out_channels

        # initial stem convolution
        stem_conv = ConvBnRelu(in_channels, out_channels, 3, 1, 1)
        self.layers.append(stem_conv)

        in_channels = out_channels
        for stack_num in range(args.num_stacks):
            if stack_num > 0:
                downsample = nn.MaxPool2d(kernel_size=2, stride=2)
                self.layers.append(downsample)

                out_channels *= 2

            for module_num in range(args.num_modules_per_stack):
                cell = Cell(spec, in_channels, out_channels)
                self.layers.append(cell)
                in_channels = out_channels

        self.classifier = nn.Linear(out_channels, args.num_labels)

        # for DARTS search
        num_edge = np.shape(spec.matrix)[0]
        self.arch_parameters = nn.Parameter(1e-3 * torch.randn(num_edge, len(searchspace)))

        self._initialize_weights()

    # Rest of the Network class code remains unchanged


    def forward(self, x, get_ints=True):
        ints = []
        for _, layer in enumerate(self.layers):
            x = layer(x)
            ints.append(x)
        out = torch.mean(x, (2, 3))
        ints.append(out)
        out = self.classifier(out)
        if get_ints:
            return out, ints[-1]
        else:
            return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                if n > 0:
                    m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def get_weights(self):
        xlist = []
        for m in self.modules():
            xlist.append(m.parameters())
        return xlist

    def get_alphas(self):
        return [self.arch_parameters]

    def genotype(self):
        return str(spec)


class Cell(nn.Module):
    """
    Builds the model using the adjacency matrix and op labels specified. Channels
    controls the module output channel count but the interior channels are
    determined via equally splitting the channel count whenever there is a
    concatenation of Tensors.
    """
    def __init__(self, spec, in_channels, out_channels):
        super(Cell, self).__init__()

        self.spec = spec
        self.num_vertices = np.shape(self.spec.matrix)[0]

        # vertex_channels[i] = number of output channels of vertex i
        self.vertex_channels = ComputeVertexChannels(in_channels, out_channels, self.spec.matrix)

        # operation for each node
        self.vertex_op = nn.ModuleList([None])
        for t in range(1, self.num_vertices-1):
            op = OP_MAP[spec.ops[t]](self.vertex_channels[t], self.vertex_channels[t])
            self.vertex_op.append(op)

        # operation for input on each vertex
        self.input_op = nn.ModuleList([None])
        for t in range(1, self.num_vertices):
            if self.spec.matrix[0, t]:
                self.input_op.append(Projection(in_channels, self.vertex_channels[t]))
            else:
                self.input_op.append(None)

    def forward(self, x):
        tensors = [x]
        out_concat = []
        for t in range(1, self.num_vertices-1):
            fan_in = [Truncate(tensors[src], self.vertex_channels[t]) for src in range(1, t) if self.spec.matrix[src, t]]
            fan_in_inds = [src for src in range(1, t) if self.spec.matrix[src, t]]

            if self.spec.matrix[0, t]:
                fan_in.append(self.input_op[t](x))
                fan_in_inds = [0] + fan_in_inds

            # perform operation on node
            vertex_input = sum(fan_in)
            vertex_output = self.vertex_op[t](vertex_input)

            tensors.append(vertex_output)
            if self.spec.matrix[t, self.num_vertices-1]:
                out_concat.append(tensors[t])

        if not out_concat: # empty list
            assert self.spec.matrix[0, self.num_vertices-1]
            outputs = self.input_op[self.num_vertices-1](tensors[0])
        else:
            if len(out_concat) == 1:
                outputs = out_concat[0]
            else:
                outputs = torch.cat(out_concat, 1)

            if self.spec.matrix[0, self.num_vertices-1]:
                outputs += self.input_op[self.num_vertices-1](tensors[0])

        return outputs

def Projection(in_channels, out_channels):
    """1x1 projection (as in ResNet) followed by batch normalization and ReLU."""
    return ConvBnRelu(in_channels, out_channels, 1)

def Truncate(inputs, channels):
    """Slice the inputs to channels if necessary."""
    input_channels = inputs.size()[1]
    if input_channels < channels:
        raise ValueError('input channel < output channels for truncate')
    elif input_channels == channels:
        return inputs   # No truncation necessary
    else:
        # Truncation should only be necessary when channel division leads to
        # vertices with +1 channels. The input vertex should always be projected to
        # the minimum channel count.
        assert input_channels - channels == 1
        return inputs[:, :channels, :, :]

def ComputeVertexChannels(in_channels, out_channels, matrix):
    """Computes the number of channels at every vertex.
    Given the input channels and output channels, this calculates the number of
    channels at each interior vertex. Interior vertices have the same number of
    channels as the max of the channels of the vertices it feeds into. The output
    channels are divided amongst the vertices that are directly connected to it.
    When the division is not even, some vertices may receive an extra channel to
    compensate.
    Returns:
        list of channel counts, in order of the vertices.
    """
    num_vertices = np.shape(matrix)[0]

    vertex_channels = [0] * num_vertices
    vertex_channels[0] = in_channels
    vertex_channels[num_vertices - 1] = out_channels

    if num_vertices == 2:
        return vertex_channels

    in_degree = np.sum(matrix[1:], axis=0)
    interior_channels = out_channels // in_degree[num_vertices - 1]
    correction = out_channels % in_degree[num_vertices - 1]

    for v in range(1, num_vertices - 1):
        if matrix[v, num_vertices - 1]:
            vertex_channels[v] = interior_channels
            if correction:
                vertex_channels[v] += 1
                correction -= 1

    for v in range(num_vertices - 3, 0, -1):
        if not matrix[v, num_vertices - 1]:
            for dst in range(v + 1, num_vertices - 1):
                if matrix[v, dst]:
                    vertex_channels[v] = max(vertex_channels[v], vertex_channels[dst])
        if vertex_channels[v] == 0:
            print(f"Vertex {v} has no incoming edges.")
            print(f"Adjacency matrix:\n{matrix}")
            print(f"Vertex channels: {vertex_channels}")

        assert vertex_channels[v] > 0, f"vertex_channels[{v}] = {vertex_channels[v]}, matrix={matrix}, in_channels={in_channels}, out_channels={out_channels}"

    final_fan_in = 0
    for v in range(1, num_vertices - 1):
        if matrix[v, num_vertices - 1]:
            final_fan_in += vertex_channels[v]
        for dst in range(v + 1, num_vertices - 1):
            if matrix[v, dst]:
                assert vertex_channels[v] >= vertex_channels[dst]
    assert final_fan_in == out_channels or num_vertices == 2

    return vertex_channels

def random_spec(num_vertices, allowed_ops):
    """Returns a random valid spec."""
    while True:
        matrix = np.random.choice([0, 1], size=(num_vertices, num_vertices))
        matrix = np.triu(matrix, 1)
        ops = np.random.choice(allowed_ops[1:-1], size=(num_vertices - 2)).tolist()
        ops = [allowed_ops[0]] + ops + [allowed_ops[-1]]
        if is_valid(matrix, ops, num_vertices):
            return matrix, ops

def generate_random_architecture(max_num_vertices=7, num_ops=3):
    generated_architectures = set()

    while True:
        num_vertices = random.randint(2, max_num_vertices)

        # Generate the adjacency matrix
        matrix = np.zeros((num_vertices, num_vertices), dtype=int)
        for i in range(num_vertices):
            for j in range(i+1, num_vertices):
                matrix[i, j] = random.choice([0, 1])

        # Check if there's a path from the first node to the last node
        visited = set()
        stack = [0]
        while stack:
            node = stack.pop()
            if node == num_vertices - 1:
                break
            if node not in visited:
                visited.add(node)
                stack.extend([v for v, edge in enumerate(matrix[node]) if edge and v not in visited])
        else:
            # The generated matrix doesn't have a path from input to output, so generate a new one
            continue

        # Generate the operation labels
        ops = ['input']
        for _ in range(1, num_vertices - 1):
            ops.append(OP_MAP_INV[random.randint(0, num_ops - 1)])
        ops.append('output')

        # Check if the generated architecture has already been generated
        matrix_ops_hash = hash((matrix.tobytes(), tuple(ops)))
        if matrix_ops_hash not in generated_architectures:
            generated_architectures.add(matrix_ops_hash)
            return matrix, ops
    
def is_valid(matrix, ops, num_vertices):
    """Check if the generated architecture is valid."""
    # Check if the input and output nodes are connected
    visited = [False] * num_vertices
    visited[0] = True
    stack = [0]
    while stack:
        curr_node = stack.pop()
        for node, edge in enumerate(matrix[curr_node]):
            if edge and not visited[node]:
                visited[node] = True
                stack.append(node)

    if not visited[-1]:
        return False

    # Check for cycles
    for row in matrix + matrix.T:
        if sum(row) > 1:
            return False

    return True

class ModelSpec:
    def __init__(self, matrix, ops):
        self.matrix = matrix
        self.ops = ops
        self.valid_spec = True

