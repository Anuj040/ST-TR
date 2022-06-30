import argparse
import os

import yaml


def save_arg(arg):
    # save arg
    arg_dict = vars(arg)
    os.makedirs(arg.work_dir, exist_ok=True)
    with open(f"{arg.work_dir}/config.yaml", "w") as f:
        yaml.dump(arg_dict, f)


def str2bool(v):
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def import_class(name):
    components = name.split(".")
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod
