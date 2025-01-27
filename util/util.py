import torch
from torch import nn
import numpy as np
from dataset.constants import *

def estimate_memory_training(model, sample_input, optimizer_type=torch.optim.Adam, batch_size=1, use_amp=False, device=0):
    """Predict the maximum memory usage of the model. 
    Args:
        optimizer_type (Type): the class name of the optimizer to instantiate
        model (nn.Module): the neural network model
        sample_input (torch.Tensor): A sample input to the network. It should be 
            a single item, not a batch, and it will be replicated batch_size times.
        batch_size (int): the batch size
        use_amp (bool): whether to estimate based on using mixed precision
        device (torch.device): the device to use
    """
    # Reset model and optimizer
    model.cpu()
    optimizer = optimizer_type(model.parameters(), lr=.001)
    a = torch.cuda.memory_allocated(device)
    model.to(device)
    b = torch.cuda.memory_allocated(device)
    model_memory = b - a
    model_input = torch.stack([sample_input]*batch_size, dim=0)
    # model_input = sample_input.unsqueeze(0).repeat(batch_size, 1)
    output = model(model_input.to(device))
    c = torch.cuda.memory_allocated(device)
    if use_amp:
        amp_multiplier = .5
    else:
        amp_multiplier = 1
    forward_pass_memory = (c - b)*amp_multiplier
    gradient_memory = model_memory
    if isinstance(optimizer, torch.optim.Adam):
        o = 2
    elif isinstance(optimizer, torch.optim.RMSprop):
        o = 1
    elif isinstance(optimizer, torch.optim.SGD):
        o = 0
    elif isinstance(optimizer, torch.optim.Adagrad):
        o = 1
    else:
        raise ValueError("Unsupported optimizer. Look up how many moments are" +
            "stored by your optimizer and add a case to the optimizer checker.")
    gradient_moment_memory = o*gradient_memory
    total_memory = model_memory + forward_pass_memory + gradient_memory + gradient_moment_memory

    return total_memory

def estimate_memory_inference(model, sample_input, batch_size=1, use_amp=False, device=0):
    """Predict the maximum memory usage of the model. 
    Args:
        optimizer_type (Type): the class name of the optimizer to instantiate
        model (nn.Module): the neural network model
        sample_input (torch.Tensor): A sample input to the network. It should be 
            a single item, not a batch, and it will be replicated batch_size times.
        batch_size (int): the batch size
        use_amp (bool): whether to estimate based on using mixed precision
        device (torch.device): the device to use
    """
    # Reset model and optimizer
    model.cpu()
    a = torch.cuda.memory_allocated(device)
    model.to(device)
    b = torch.cuda.memory_allocated(device)
    model_memory = b - a
    model_input = torch.stack([sample_input]*batch_size, dim=0)
    output = model(model_input.to(device))
    c = torch.cuda.memory_allocated(device)
    if use_amp:
        amp_multiplier = .5
    else:
        amp_multiplier = 1
    forward_pass_memory = (c - b)*amp_multiplier
    total_memory = model_memory+forward_pass_memory

    return total_memory

def test_memory_training(in_size=100, out_size=10, hidden_size=100, optimizer_type=torch.optim.Adam, batch_size=1, use_amp=False, device=0):
    sample_input = torch.randn(batch_size, in_size, dtype=torch.float32)
    model = nn.Sequential(nn.Linear(in_size, hidden_size),
                        *[nn.Linear(hidden_size, hidden_size) for _ in range(200)],
                        nn.Linear(hidden_size, out_size))
    max_mem_est = estimate_memory_training(model, sample_input[0], optimizer_type=optimizer_type, batch_size=batch_size, use_amp=use_amp)
    print("Maximum Memory Estimate", max_mem_est)
    optimizer = optimizer_type(model.parameters(), lr=.001)
    print("Beginning mem:", torch.cuda.memory_allocated(device), "Note - this may be higher than 0, which is due to PyTorch caching. Don't worry too much about this number")
    model.to(device)
    print("After model to device:", torch.cuda.memory_allocated(device))
    for i in range(3):
        optimizer.zero_grad()
        print("Iteration", i)
        with torch.cuda.amp.autocast(enabled=use_amp):
            a = torch.cuda.memory_allocated(device)
            out = model(sample_input.to(device)).sum() # Taking the sum here just to get a scalar output
            b = torch.cuda.memory_allocated(device)
        print("1 - After forward pass", torch.cuda.memory_allocated(device))
        print("2 - Memory consumed by forward pass", b - a)
        out.backward()
        print("3 - After backward pass", torch.cuda.memory_allocated(device))
        optimizer.step()
        print("4 - After optimizer step", torch.cuda.memory_allocated(device))

def test_memory_inference(in_size=100, out_size=10, hidden_size=100, batch_size=1, use_amp=False, device=0):
    sample_input = torch.randn(batch_size, in_size, dtype=torch.float32)
    model = nn.Sequential(nn.Linear(in_size, hidden_size),
                        *[nn.Linear(hidden_size, hidden_size) for _ in range(200)],
                        nn.Linear(hidden_size, out_size))
    max_mem_est = estimate_memory_inference(model, sample_input[0], batch_size=batch_size, use_amp=use_amp)
    print("Maximum Memory Estimate", max_mem_est)
    print("Beginning mem:", torch.cuda.memory_allocated(device), "Note - this may be higher than 0, which is due to PyTorch caching. Don't worry too much about this number")
    model.to(device)
    print("After model to device:", torch.cuda.memory_allocated(device))
    with torch.no_grad():
        for i in range(3):
            print("Iteration", i)
            with torch.cuda.amp.autocast(enabled=use_amp):
                a = torch.cuda.memory_allocated(device)
                out = model(sample_input.to(device)).sum() # Taking the sum here just to get a scalar output
                b = torch.cuda.memory_allocated(device)
            print("1 - After forward pass", torch.cuda.memory_allocated(device))
            print("2 - Memory consumed by forward pass", b - a)




def resample_2d(points, N):

    xc = points[:, 0].tolist() + [points[0, 0]]
    yc = points[:, 1].tolist() + [points[0, 1]]

    dx = np.diff(xc)
    dy = np.diff(yc)

    dS = np.sqrt(dx**2+dy**2)
    dS = np.array([0]+dS.tolist())

    d = np.cumsum(dS)

    perim = d[-1]

    ds = perim/N
    dSi = ds*np.arange(0, N)
    dSi[-1] = dSi[-1] - 0.005

    xi = np.interp(dSi, d, xc)
    yi = np.interp(dSi, d, yc)
    return xi, yi


def get_input_dim(args):
    d = args.get('dataloader')
    if d == 'SP' or d == 'SPLAP' or d == 'INPE':
        return 3
    elif d == 'SPF' or d == 'DUTS':
        return 3
    elif d in ['SPFFFT','SPGFFT','ImageNet',
                'YD', 'SPGSWIN', 'YDG',
              'ImageNet_SWIN',  'SpeedLimits', 
                'SPSpeedLimits',  'SPFRS', 'SPRS', 'SPFFT' ]:
        if args.get('ignore_phase'):
            return 6+(args.get('coeff'))+10
        else:
            return 6+(args.get('coeff')*2)+10
    elif d== 'SPGIFFT' or d == 'ImageNet_PyG':
        if args.get('ignore_phase'):
            return 6+(args.get('coeff'))
        else:
            return 6+(args.get('coeff')*2)
    elif d == 'SPG' or d == 'SPGI':
        return 3
    else:
        raise 'Unrecognized dataloader'
    

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def create_batch_grid(rows, cols):
    grid = torch.zeros([rows, cols], device='cuda')
    col_starter = 0 
    for i in range(rows):
        shifter = 0 
        for j in range(cols):
            grid[i, j] = col_starter+shifter
            if j % 2 == 1:
                shifter += 1
        if i%2 == 1:
            col_starter += cols
    
    return grid.long()

def create_edge_index(rows, cols):
    grid = torch.arange(0, rows*cols).reshape(rows, cols)
    edge_index = []
    for i in range(rows):
        for j in range(cols):
            seed_node = grid[i, j]
            for h, v in [[-1, 0], [1, 0], [0, -1], [0, 1]]:
                if 0<=i+h<rows and 0<=j+v<cols:
                    edge_index.append([seed_node, grid[i+h, j+v]])
    edge_index = torch.tensor(edge_index, dtype=torch.int64, device='cuda').permute(1, 0)

    del grid
    return edge_index

def merge_contours(contours):
    """
    Merges multiple contours by connecting their closest points and updating the order of points.

    Parameters:
    - contours: List of contours from OpenCV's findContours, where each contour is a numpy array of shape (n, 1, 2).

    Returns:
    - merged_contour: A single contour combining all input contours.
    """
    # Flatten the contours to remove hierarchy dimension
    contours = [c.reshape(-1, 2) for c in contours]

    while len(contours) > 1:
        # Find the two closest contours
        min_distance = float('inf')
        closest_pair = None

        for i in range(len(contours)):
            for j in range(i + 1, len(contours)):
                contour_a = contours[i]
                contour_b = contours[j]

                # Compute pairwise distances
                distances = np.linalg.norm(
                    contour_a[:, None, :] - contour_b[None, :, :], axis=2)
                min_idx = np.unravel_index(np.argmin(distances), distances.shape)

                distance = distances[min_idx]
                if distance < min_distance:
                    min_distance = distance
                    closest_pair = (i, j, min_idx)

        # Retrieve the indices of the two closest contours and points
        i, j, (idx_a, idx_b) = closest_pair
        contour_a, contour_b = contours[i], contours[j]

        # Reorder contour_a and contour_b to maintain continuity
        contour_a = np.roll(contour_a, -idx_a, axis=0)
        contour_b = np.roll(contour_b, -idx_b, axis=0)

        # Create a connecting line between the two contours
        connecting_line = np.array([contour_a[-1], contour_b[0]])

        # Merge the contours and connecting line
        merged_contour = np.vstack([contour_a, connecting_line, contour_b])

        # Update the contours list
        contours.pop(j)  # Remove second contour (higher index first to avoid indexing issues)
        contours.pop(i)  # Remove first contour
        contours.append(merged_contour)  # Add merged contour back to the list

    # Return the final merged contour
    return contours[0].reshape(-1, 1, 2)