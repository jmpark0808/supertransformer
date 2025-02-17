import torch
from torch import nn
import numpy as np
from dataset.constants import *
from skimage.measure import moments_central

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
    moments = args.get('moments')
    if d == 'SP' or d == 'SPLAP' or d == 'INPE':
        return 3
    elif d == 'SPF' or d == 'DUTS':
        return 3
    elif d in ['SPFFFT','SPGFFT','ImageNet',
                'YD', 'SPGSWIN', 'YDG',
              'ImageNet_SWIN',  'SpeedLimits', 
                'SPSpeedLimits',  'SPFRS', 'SPRS', 'SPFFT' ]:
        if moments:
            return 6+8+10
        else:
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


def eval_e(y_pred, y, num):
    score = torch.zeros(num).cuda()
    thlist = torch.linspace(0, 1 - 1e-10, num).cuda()
    for i in range(num):
        y_pred_th = (y_pred >= thlist[i]).float()
        fm = y_pred_th - y_pred_th.mean()
        gt = y - y.mean()
        align_matrix = 2 * gt * fm / (gt * gt + fm * fm + 1e-20)
        enhanced = ((align_matrix + 1) * (align_matrix + 1)) / 4
        score[i] = torch.sum(enhanced) / (y.numel() - 1 + 1e-20)
    return score

def detect_object(pred, gt):
    temp = pred[gt == 1]
    x = temp.mean()
    sigma_x = temp.std()
    score = 2.0 * x / (x * x + 1.0 + sigma_x + 1e-20)
    
    return score

def S_object(pred, gt):
    fg = torch.where(gt==0, torch.zeros_like(pred), pred)
    bg = torch.where(gt==1, torch.zeros_like(pred), 1-pred)
    o_fg = detect_object(fg, gt)
    o_bg = detect_object(bg, 1-gt)
    u = gt.mean()
    Q = u * o_fg + (1-u) * o_bg
    return Q


def centroid(gt):
    rows, cols = gt.size()[-2:]
    gt = gt.view(rows, cols)
    if gt.sum() == 0:
        
        X = torch.eye(1).cuda() * round(cols / 2)
        Y = torch.eye(1).cuda() * round(rows / 2)
        
    else:
        total = gt.sum()
       
        i = torch.from_numpy(np.arange(0,cols)).cuda().float()
        j = torch.from_numpy(np.arange(0,rows)).cuda().float()
        
        X = torch.round((gt.sum(dim=0)*i).sum() / total)
        Y = torch.round((gt.sum(dim=1)*j).sum() / total)
    return X.long(), Y.long()

def divideGT(gt, X, Y):
    h, w = gt.size()[-2:]
    area = h*w
    gt = gt.view(h, w)
    LT = gt[:Y, :X]
    RT = gt[:Y, X:w]
    LB = gt[Y:h, :X]
    RB = gt[Y:h, X:w]
    X = X.float()
    Y = Y.float()
    w1 = X * Y / area
    w2 = (w - X) * Y / area
    w3 = X * (h - Y) / area
    w4 = 1 - w1 - w2 - w3
    return LT, RT, LB, RB, w1, w2, w3, w4

def dividePrediction(pred, X, Y):
    h, w = pred.size()[-2:]
    pred = pred.view(h, w)
    LT = pred[:Y, :X]
    RT = pred[:Y, X:w]
    LB = pred[Y:h, :X]
    RB = pred[Y:h, X:w]
    return LT, RT, LB, RB

def ssim(pred, gt):
    gt = gt.float()
    h, w = pred.size()[-2:]
    N = h*w
    x = pred.mean()
    y = gt.mean()
    sigma_x2 = ((pred - x)*(pred - x)).sum() / (N - 1 + 1e-20)
    sigma_y2 = ((gt - y)*(gt - y)).sum() / (N - 1 + 1e-20)
    sigma_xy = ((pred - x)*(gt - y)).sum() / (N - 1 + 1e-20)
    
    aplha = 4 * x * y *sigma_xy
    beta = (x*x + y*y) * (sigma_x2 + sigma_y2)

    if aplha != 0:
        Q = aplha / (beta + 1e-20)
    elif aplha == 0 and beta == 0:
        Q = 1.0
    else:
        Q = 0
    return Q


def S_region(pred, gt):
    X, Y = centroid(gt)
    gt1, gt2, gt3, gt4, w1, w2, w3, w4 = divideGT(gt, X, Y)
    p1, p2, p3, p4 = dividePrediction(pred, X, Y)
    Q1 = ssim(p1, gt1)
    Q2 = ssim(p2, gt2)
    Q3 = ssim(p3, gt3)
    Q4 = ssim(p4, gt4)
    Q = w1*Q1 + w2*Q2 + w3*Q3 + w4*Q4
    # print(Q)
    return Q


def compute_central_moments(binary_image):
    """
    Computes central moments of a binary image using skimage.measure.regionprops.
    
    Parameters:
    - binary_image: (2D numpy array) Binary image.

    Returns:
    - central_moments (numpy array): The computed central moments.
    """
    # label_image = measure.label(binary_image)  # Label connected components
    # props = measure.regionprops(label_image)
    # moments = props[0].moments_central 
    moments_ = moments_central(binary_image)
    # moments_ = np.sign(moments)*np.log(np.abs(moments)+1e-10)
    moments_ = np.array([moments_[0,0], moments_[1, 1], moments_[2, 0],
                          moments_[0, 2], moments_[2, 1], moments_[1, 2], moments_[3, 0], moments_[0, 3]])
    return moments_


