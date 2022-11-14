
import sys
sys.path.insert(0, '/home/eddie/waterloo/supertransformer')
from Models.SP_TFM import SP_TFM
from Models.ITSD import baseline
from Models.EGNet import build_model
import math
import torch
from ptflops import get_model_complexity_info
from util import util
import matplotlib.pyplot as plt


models = ['SGT', 'ITSD', 'EGNet']
resolutions = [224, 256, 512, 1024]

fig, axs = plt.subplots(3, figsize=(10, 10))
for ind, model in enumerate(models):
    flops = []
    training_mems = []
    inference_mems = []
    for res in resolutions:
        if model == 'SGT':
            forward = SP_TFM(11, 16, 8, 6)
            input_shape = (res, 11)
        elif model == 'ITSD':
            forward = baseline('resnet')
            input_shape = (3, res, res)
        elif model == 'EGNet':
            forward = build_model('resnet')
            input_shape = (3, res, res)
        


        input = torch.rand(input_shape)
        
        # forward.cpu()
        # optimizer = torch.optim.Adam(forward.parameters(), lr=.001)
        # a = torch.cuda.memory_allocated(0)
        # forward.to('cuda')
        # b = torch.cuda.memory_allocated(0)
        # model_memory = b - a
        # input = torch.stack([input]*1, dim=0).to('cuda')
        # # model_input = sample_input.unsqueeze(0).repeat(batch_size, 1)
        # output = forward(input)
        # c = torch.cuda.memory_allocated(0)

        # forward_pass_memory = (c - b)
        # gradient_memory = model_memory
        # if isinstance(optimizer, torch.optim.Adam):
        #     o = 2
        # elif isinstance(optimizer, torch.optim.RMSprop):
        #     o = 1
        # elif isinstance(optimizer, torch.optim.SGD):
        #     o = 0
        # elif isinstance(optimizer, torch.optim.Adagrad):
        #     o = 1
        # else:
        #     raise ValueError("Unsupported optimizer. Look up how many moments are" +
        #         "stored by your optimizer and add a case to the optimizer checker.")
        # gradient_moment_memory = o*gradient_memory
        # total_memory = model_memory + forward_pass_memory + gradient_memory + gradient_moment_memory
        total_memory = util.estimate_memory_training(forward, input)
        training_mems.append(math.log10(total_memory))

        with torch.no_grad():
            forward.eval()
            # forward.cpu()
            # a = torch.cuda.memory_allocated(0)
            # forward.to('cuda')
            # b = torch.cuda.memory_allocated(0)
            # model_memory = b - a

            flop, params = get_model_complexity_info(forward, input_res=input_shape, as_strings=False)
            
            flops.append(math.log10(flop))

            model_memory = util.estimate_memory_inference(forward, input)
            inference_mems.append(math.log10(model_memory))
        # del input
        # del forward
        # torch.cuda.empty_cache()

    axs[0].plot(resolutions, flops, label=model)
    axs[0].scatter(resolutions, flops, c='red')
    axs[0].set_xticks([])
    axs[0].set_ylabel('MACs')
    # axs[0].legend()

    axs[1].plot(resolutions, training_mems, label=model)
    axs[1].scatter(resolutions, training_mems, c='red')
    axs[1].set_xticks([])
    axs[1].set_ylabel('Training Mem')
    axs[1].legend(loc='center left', bbox_to_anchor=(1, 0.5))

    axs[2].plot(resolutions, inference_mems, label=model)
    axs[2].scatter(resolutions, inference_mems, c='red')
    axs[2].set_ylabel('Inference Mem')
    axs[2].set_xticks(resolutions)
    # axs[2].legend()
    axs[2].set_xlabel('Resolution')
    fig.supylabel('Log scale')
    print(flops)
    print(training_mems)
    print(inference_mems)
    assert(0)

fig.tight_layout()
fig.savefig('/home/eddie/waterloo/supertransformer/results/image_res_vs_computation.png')
# plt.show()


    


