import torch.nn as nn
import torch
class GroupedLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        groups=1,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()

        assert in_features % groups == 0
        assert out_features % groups == 0

        self.in_features = in_features
        self.out_features = out_features
        self.groups = groups

        self._linear_layers = nn.ModuleList(
            [
                nn.Linear(
                    in_features // groups,
                    out_features // groups,
                    bias=bias,
                    device=device,
                    dtype=dtype,
                )
                for _ in range(groups)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.shape[:-1]+(-1, self.groups))

        result = [
            l(x[..., i])
            for i, l in enumerate(self._linear_layers)
        ]
        return torch.cat(result, dim=-1)
    


conv1d = nn.Conv1d(256, 256, 1, groups=8)
grouped = GroupedLinear(256, 256, groups=8) 

random_input = torch.randn(1, 256, 1024)

import time
start = time.time()
for _ in range(1000):
    conv1d(random_input)
end = time.time()
print('Conv1d', end-start)

random_input = random_input.permute(0, 2, 1)
start = time.time()
for _ in range(1000):
    grouped(random_input)
end = time.time()
print('Grouped', end-start)

