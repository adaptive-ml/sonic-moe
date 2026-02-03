# ********************************************************************************
# Copyright (c) 2025, Wentao Guo, Mayank Mishra, Xinle Cheng, Ion Stoica, Tri Dao
# ********************************************************************************

import torch

try:
    from harmony_kernels import count_cumsum_cuda as _harmony_count_cumsum_cuda
except ImportError as e:
    raise ImportError(
        "harmony_kernels is required for SonicMoE. "
        "Please install harmony_kernels to use this library. "
        "count_cumsum_cuda must be available from harmony_kernels."
    ) from e


@torch.compiler.disable
def count_cumsum_cuda(x: torch.Tensor, count_output: torch.Tensor, cumsum_output: torch.Tensor | None, stream: int) -> None:
    if cumsum_output is None:
        cumsum_output = torch.empty(count_output.shape[0], dtype=torch.int32, device=count_output.device)
    _harmony_count_cumsum_cuda(x, count_output, cumsum_output)


@torch.no_grad()
def count_cumsum(x: torch.Tensor, E: int, do_cumsum: bool = True) -> torch.Tensor:
    assert x.dim() == 1, "x should be 1-dimensional"
    assert x.dtype in [torch.int32, torch.long]

    count_output = torch.empty(E, dtype=torch.int32, device=x.device)
    cumsum_output = torch.empty(E, dtype=torch.int32, device=x.device) if do_cumsum else None
    stream = torch.cuda.current_stream(x.device).cuda_stream

    count_cumsum_cuda(x=x, count_output=count_output, cumsum_output=cumsum_output, stream=stream)

    if do_cumsum:
        return count_output, cumsum_output
    else:
        return count_output
