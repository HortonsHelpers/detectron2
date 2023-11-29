# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import numpy as np
import os
import subprocess
import sys
from collections import defaultdict
import PIL
import torch
import torchvision
from tabulate import tabulate

__all__ = ["collect_env_info"]


def collect_torch_env():
    try:
        import torch.__config__

        return torch.__config__.show()
    except ImportError:
        # compatible with older versions of pytorch
        from torch.utils.collect_env import get_pretty_env_info

        return get_pretty_env_info()


def get_env_module():
    var_name = "DETECTRON2_ENV_MODULE"
    return var_name, os.environ.get(var_name, "<not set>")


def collect_env_info():
    data = [
        ("sys.platform", sys.platform),
        ("Python", sys.version.replace("\n", "")),
        ("Numpy", np.__version__),
    ]
    try:
        from detectron2 import _C
    except ImportError:
        data.append(("detectron2._C", "failed to import"))
    else:
        data.append(("Detectron2 Compiler", _C.get_compiler_version()))
        data.append(("Detectron2 CUDA Compiler", _C.get_cuda_version()))

    data.extend(
        (
            get_env_module(),
            ("PyTorch", torch.__version__),
            ("PyTorch Debug Build", torch.version.debug),
        )
    )
    try:
        data.append(("torchvision", torchvision.__version__))
    except AttributeError:
        data.append(("torchvision", "unknown"))

    has_cuda = torch.cuda.is_available()
    data.append(("CUDA available", has_cuda))
    if has_cuda:
        devices = defaultdict(list)
        for k in range(torch.cuda.device_count()):
            devices[torch.cuda.get_device_name(k)].append(str(k))
        data.extend(
            ("GPU " + ",".join(devids), name)
            for name, devids in devices.items()
        )
        from torch.utils.cpp_extension import CUDA_HOME

        data.append(("CUDA_HOME", str(CUDA_HOME)))

        if CUDA_HOME is not None and os.path.isdir(CUDA_HOME):
            try:
                nvcc = os.path.join(CUDA_HOME, "bin", "nvcc")
                nvcc = subprocess.check_output(f"'{nvcc}' -V | tail -n1", shell=True)
                nvcc = nvcc.decode("utf-8").strip()
            except subprocess.SubprocessError:
                nvcc = "Not Available"
            data.append(("NVCC", nvcc))

        if cuda_arch_list := os.environ.get("TORCH_CUDA_ARCH_LIST", None):
            data.append(("TORCH_CUDA_ARCH_LIST", cuda_arch_list))
    data.append(("Pillow", PIL.__version__))

    try:
        import cv2

        data.append(("cv2", cv2.__version__))
    except ImportError:
        pass
    env_str = tabulate(data) + "\n"
    env_str += collect_torch_env()
    return env_str


if __name__ == "__main__":
    print(collect_env_info())
