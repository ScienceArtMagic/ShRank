import copy
from dataclasses import DataClass
import torch
import torch.nn as nn
from .lr_module import LED, CED
import warnings
from transformers.modeling_utils import Conv1D as HFConv1D

r"""
Input:
    weight - weight of the original nn.module to be factorized
    rank - the rank to be applied for low-rank factorization
Output:
   low-rank factorization weight matrix (U.S) and V
"""


def lr_svd(weight, rank: int):
    U, S, Vh = torch.linalg.svd(weight.T, full_matrices=True)
    return U[:, :rank] @ torch.diag(S[:rank]), Vh[:rank, :].T


r"""
Input:
    module - nn.module to be factorized
    rank - the rank to be applied for low-rank factorization
    groups_out - (convolution only) make ced_unit[1] a grouped convolution if True (and skip_high_groups == False)
    skip_high_groups - (convolution only) don't factorize grouped convolution if out_channels // groups > rank
    fact_led_unit - flag for skipping factorization on LED and CED unit
    
Output:
    low-rank version of the given module
"""


def factorize_module(
    module: nn.Linear | HFConv1D | nn.Conv1d | nn.Conv2d | nn.Conv3d,
    rank: int,
    groups_out: bool,
    skip_high_groups: bool,
    # fact_led_unit: bool,
):
    # def get_fractional_rank(rank: int, limit_rank: int) -> Tuple[int, int]:
    #     # Define rank from the given rank percentage
    #     if rank < 1:
    #         rank = int(limit_rank * rank)
    #         if rank == 0:
    #             return module
    #     rank = int(rank)

    if type(module) in [nn.Linear, HFConv1D]:
        in_features, out_features = (
            (module.in_features, module.out_features)
            if type(module) is nn.Linear
            else module.weight.shape
        )
        limit_rank = int((in_features * out_features) / (in_features + out_features))
        if limit_rank <= rank:
            warnings.warn(
                f"skipping {type(module)} with in: {module.in_features}, out: {module.out_features}, rank: {rank}"
            )
            # Ignore if input/output features are smaller than rank to prevent factorization on low dimensional input/output vector
            return module

        # Extract module weight
        weight = module.weight if type(module) is nn.Linear else module.weight.T

        # Create LED unit
        led_module = LED(
            module.in_features,
            module.out_features,
            r=rank,
            bias=module.bias is not None,
            device=module.weight.device,
        )

        # Initialize matrix
        U, V = lr_svd(weight.T, rank)
        led_module.led_unit[0].weight.data = U.T  # Initialize U
        led_module.led_unit[1].weight.data = V.T  # Initialize V
        if module.bias is not None:
            led_module.led_unit[1].bias = module.bias

        # Return module
        return led_module
    elif type(module) in [nn.Conv1d, nn.Conv2d, nn.Conv3d]:
        out_limit = (
            module.out_channels // module.groups
            if skip_high_groups
            else module.out_channels
        )
        limit_rank = int(
            (module.in_channels * (out_limit)) / (module.in_channels + (out_limit))
        )
        if limit_rank <= rank:
            warnings.warn(
                f"skipping {type(module)} with in: {module.in_channels}, out: {module.out_channels // module.groups}, rank: {rank}"
            )
            # Ignore if input/output channels are smaller than rank to prevent factorization on low dimensional input/output vector
            return module

        # Extract layer weight
        weight = module.weight.view(module.out_channels, -1)

        # Replace with CED unit
        ced_module = CED(
            module.in_channels,
            module.out_channels,
            r=rank,
            kernel_size=module.kernel_size,
            stride=module.stride,
            padding=module.padding,
            dilation=module.dilation,
            padding_mode=module.padding_mode,
            groups=module.groups,
            groups_out=groups_out,
            bias=module.bias is not None,
            device=module.weight.device,
        )

        # # Initialize matrix
        U, V = lr_svd(
            weight,
            rank,
        )
        ced_module.ced_unit[0].weight.data = U.T
        # .view_as(
        #     ced_module.ced_unit[0].weight
        # )  # Initialize U
        ced_module.ced_unit[1].weight.data = V.T
        # .view_as(
        #     ced_module.ced_unit[1].weight
        # )  # Initialize Vh
        if module.bias is not None:
            ced_module.ced_unit[1].bias.data = module.bias.data
        return ced_module


r"""
Input:
    module - the module (nn.Module) to be factorized (required)
    rank - the rank to be applied for low-rank factorization (required)
    deepcopy - deepcopy module before factorization, return new factorized copy of the model (default: False)
    submodules - submodules of model of which the factorization will be applied (default: None)
    fact_led_unit - flag for skipping factorization on LED and CED unit (default: False)
    
Output:
    low-rank version of the given module (will create a model copy if `deep_copy=True`)
"""


def auto_fact(
    module: nn.Linear | HFConv1D | nn.Conv1d | nn.Conv2d | nn.Conv3d | LED | CED,
    rank: int,
    groups_out=False,
    skip_high_groups=True,
    submodules=None,
    deepcopy=False,
    fact_led_unit=False,
):
    if deepcopy:
        copy_module = copy.deepcopy(module)
    else:
        copy_module = module

    def auto_fact_recursive(
        module: nn.Linear | HFConv1D | nn.Conv1d | nn.Conv2d | nn.Conv3d,
        reference_module: nn.Linear | HFConv1D | nn.Conv1d | nn.Conv2d | nn.Conv3d,
        rank: int,
        groups_out: bool,
        skip_high_groups: bool,
        submodules,
        fact_led_unit: bool,
        factorize_child: bool,
    ):
        # If the top module is Linear or Conv, return the factorized module directly
        if type(reference_module) in [
            nn.Linear,
            nn.Conv1d,
            nn.Conv2d,
            nn.Conv3d,
            HFConv1D,
        ]:
            return factorize_module(
                module,
                rank,
                groups_out,
                skip_high_groups,
                # fact_led_unit
            )

        for key, reference_key in zip(module._modules, reference_module._modules):
            # Skip LED or CED units if `fact_led_unit` is True
            if not fact_led_unit and type(reference_module._modules[reference_key]) in [
                LED,
                CED,
            ]:
                continue

            if type(reference_module._modules[reference_key]) in [
                nn.Linear,
                nn.Conv1d,
                nn.Conv2d,
                nn.Conv3d,
                HFConv1D,
            ] and (
                factorize_child
                or reference_module._modules[reference_key]
                in ([] if submodules is None else submodules)
            ):
                # Factorize Linear to LED and Convolution to CED
                module._modules[key] = factorize_module(
                    module._modules[key],
                    rank,
                    groups_out,
                    skip_high_groups,
                    # fact_led_unit,
                )
            else:
                # Perform recursive tracing
                if len(reference_module._modules[reference_key]._modules.items()) > 0:
                    if (
                        submodules is None
                        or reference_module._modules[reference_key] in submodules
                    ):
                        module._modules[key] = auto_fact_recursive(
                            module._modules[key],
                            reference_module._modules[reference_key],
                            rank,
                            groups_out,
                            skip_high_groups,
                            submodules,
                            fact_led_unit=fact_led_unit,
                            factorize_child=True,
                        )
                    else:
                        module._modules[key] = auto_fact_recursive(
                            module._modules[key],
                            reference_module._modules[reference_key],
                            rank,
                            groups_out,
                            skip_high_groups,
                            submodules,
                            fact_led_unit=fact_led_unit,
                            factorize_child=factorize_child,
                        )
        return module

    # Perform recursive factorization
    return auto_fact_recursive(
        copy_module,
        module,
        rank,
        groups_out,
        skip_high_groups,
        submodules,
        fact_led_unit,
        False,
    )
