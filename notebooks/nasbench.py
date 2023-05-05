from nats_bench import create
from models import get_cell_based_tiny_net

# Create the API for size search space
api = create(None, 'sss', fast_mode=True, verbose=True)

# Create the API for tologoy search space
api = create(None, 'tss', fast_mode=True, verbose=True)

# Query the loss / accuracy / time for 1234-th candidate architecture on CIFAR-10
# info is a dict, where you can easily figure out the meaning by key
info = api.get_more_info(1234, 'cifar10')

# Query the flops, params, latency. info is a dict.
info = api.get_cost_info(12, 'cifar10')

# Simulate the training of the 1224-th candidate:
validation_accuracy, latency, time_cost, current_total_time_cost = api.simulate_train_eval(1224, dataset='cifar10', hp='12')

# Clear the parameters of the 12-th candidate.
api.clear_params(12)

# Reload all information of the 12-th candidate.
api.reload(index=12)

# Create the instance of th 12-th candidate for CIFAR-10.
config = api.get_net_config(12, 'cifar10')
network = get_cell_based_tiny_net(config)

# Load the pre-trained weights: params is a dict, where the key is the seed and value is the weights.
params = api.get_net_param(12, 'cifar10', None)
network.load_state_dict(next(iter(params.values())))