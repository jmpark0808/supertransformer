import torch
import torch.nn as nn

max_relative_position = 5
num_units = 3


embeddings_table = nn.Parameter(torch.Tensor(max_relative_position * 2 + 1, num_units)).cuda()

range_vec_q = torch.arange(10)
range_vec_k = torch.arange(10)
distance_mat = range_vec_k[None, :] - range_vec_q[:, None]
distance_mat_clipped = torch.clamp(distance_mat, -max_relative_position, max_relative_position)
final_mat = distance_mat_clipped + max_relative_position
final_mat = torch.LongTensor(final_mat).cuda()
print(final_mat)
embeddings = embeddings_table[final_mat].cuda()

print(embeddings.size())