
import sys
sys.path.insert(0, '/home/eddie/waterloo/supertransformer')
from Models.SP_TFM import SP_TFM, SP_TFM_TFM
from Models.ITSD import baseline
from Models.EGNet import build_model
import math
import torch
from ptflops import get_model_complexity_info
from util import util
import matplotlib.pyplot as plt
import numpy as np

def prepare_input_gat(time_steps):
    x1 = torch.FloatTensor(time_steps[0],time_steps[1] , 11)
    x2 = torch.FloatTensor(time_steps[0], time_steps[1], time_steps[1])
    return dict(input = [x1, x2])

# flops, params = get_model_complexity_info(self.model, input_res=(1, self.num_seg), input_constructor=prepare_input_gat,
#                                               as_strings=False, print_per_layer_stat=False)

resolutions = [224, 256, 512, 1024]

fig, axs = plt.subplots(3, figsize=(10, 10))

flops = []
training_mems = []
inference_mems = []
for res in resolutions:
    forward = SP_TFM_TFM(11, 16, 1, 0, 8, 6, 'bn')
    input_shape = (res, 11)
    adj_shape = (res, res)

    
    input = torch.rand(input_shape)
    # input_adj = np.random.binomial(1, 9/res, (res, res))
    input_adj = np.ones((res, res))
    input_adj = torch.from_numpy(input_adj)
    device = 0
    use_amp = False
    optimizer_type = torch.optim.Adam
    forward.cpu()
    optimizer = optimizer_type(forward.parameters(), lr=.001)
    a = torch.cuda.memory_allocated(device)
    forward.to(device)
    b = torch.cuda.memory_allocated(device)
    model_memory = b - a
    model_input = torch.stack([input]*1, dim=0)
    model_adj_input = torch.stack([input_adj]*1, dim=0)
    # model_input = sample_input.unsqueeze(0).repeat(batch_size, 1)
    output = forward([model_input.to(device), model_adj_input.to(device)])
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

    training_mems.append(math.log10(total_memory))

    with torch.no_grad():
        forward.cpu()
        forward.eval()
        # forward.cpu()
        # a = torch.cuda.memory_allocated(0)
        # forward.to('cuda')
        # b = torch.cuda.memory_allocated(0)
        # model_memory = b - a

        flop, params = get_model_complexity_info(forward, input_res=(1, res), input_constructor=prepare_input_gat,
                                              as_strings=False, print_per_layer_stat=False)
        
        flops.append(math.log10(flop))


        forward.cpu()
        a = torch.cuda.memory_allocated(device)
        forward.to(device)
        b = torch.cuda.memory_allocated(device)
        model_memory = b - a
        model_input = torch.stack([input]*1, dim=0)
        model_adj_input = torch.stack([input_adj]*1, dim=0)
        output = forward([model_input.to(device), model_adj_input.to(device)])
        c = torch.cuda.memory_allocated(device)
        if use_amp:
            amp_multiplier = .5
        else:
            amp_multiplier = 1
        forward_pass_memory = (c - b)*amp_multiplier
        total_memory = model_memory+forward_pass_memory

        
        inference_mems.append(total_memory)
    # del input
    # del forward
    # torch.cuda.empty_cache()

axs[0].plot(resolutions, flops)
axs[0].scatter(resolutions, flops, c='red')
axs[0].set_xticks([])
axs[0].set_ylabel('MACs (log scale)', fontsize=20)
# axs[0].legend()

axs[1].plot(resolutions, training_mems)
axs[1].scatter(resolutions, training_mems, c='red')
axs[1].set_xticks([])
axs[1].set_ylabel('Training Mem bytes (log scale)', fontsize=20)
axs[1].legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=15)

axs[2].plot(resolutions, inference_mems)
axs[2].scatter(resolutions, inference_mems, c='red')
axs[2].set_ylabel('Inference Mem bytes', fontsize=20)
axs[2].set_xticks(resolutions)
# axs[2].legend()
axs[2].set_xlabel('Resolution', fontsize=20)
# fig.supylabel('Log scale')
print(flops)
print(training_mems)
print(inference_mems)


fig.tight_layout()
fig.savefig('/home/eddie/waterloo/supertransformer/results/sgct_res_vs_computation.png')
# plt.show()


    


