import importlib
from copy import deepcopy
from os import path as osp
import inspect
from basicsr.utils import get_root_logger, scandir
from basicsr.utils.registry import ARCH_REGISTRY
import shutil

__all__ = ["build_network"]

# automatically scan and import arch modules for registry
# scan all the files under the 'archs' folder and collect files ending with
# '_arch.py'
arch_folder = osp.dirname(osp.abspath(__file__))
arch_filenames = [
    osp.splitext(osp.basename(v))[0]
    for v in scandir(arch_folder)
    if v.endswith("_arch.py")
]
# import all the arch modules
_arch_modules = [
    importlib.import_module(f"basicsr.archs.{file_name}")
    for file_name in arch_filenames
]


def build_network(opt, type):
    opt = deepcopy(opt)
    if type == "network_g":
        network_type = opt["network_g"].pop("type")
        net = ARCH_REGISTRY.get(network_type)(**opt["network_g"])
    else:
        network_type = opt["network_d"].pop("type")
        net = ARCH_REGISTRY.get(network_type)(**opt["network_d"])
    logger = get_root_logger()
    arch_file_path = inspect.getfile(net.__class__)
    shutil.copy(
        arch_file_path,
        osp.join(
            opt["path"]["root_path"],
            "experiments",
            opt["name"],
            osp.basename(arch_file_path),
        ),
    )
    logger.info(f"Network [{net.__class__.__name__}] is created.")
    logger.info(net)
    return net
