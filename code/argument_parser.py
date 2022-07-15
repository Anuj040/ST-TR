import argparse
import multiprocessing
import re
from collections import OrderedDict

import configargparse as argparse
from utils.misc import str2bool


def arg_boolean(v):
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def arg_tuple(cast_type):
    regex = re.compile(r"\d+\.\d+|\d+")

    def parse_tuple(v):
        vals = regex.findall(v)
        return [cast_type(val) for val in vals]

    return parse_tuple


def arg_list_tuple(cast_type):
    regex = re.compile(r"\([^\)]*\)")
    tuple_parser = arg_tuple(cast_type)

    def parse_list(v):
        tuples = regex.findall(v)
        return [tuple_parser(t) for t in tuples]

    return parse_list


def arg_dict(cast_type):
    regex_pairs = re.compile(r"[^\ ]+=[^\ ]+")
    regex_keyvals = re.compile(r"([^\ ]+)=([^\ ]+)")

    def parse_dict(v):
        d = OrderedDict()
        for keyval in regex_pairs.findall(v):
            key, val = regex_keyvals.match(keyval).groups()
            d.update({key: cast_type(val)})
        return d

    return parse_dict


def get_parser(exp_name: str):
    # parameter priority: command line > config > default

    parser = argparse.ArgumentParser(
        description="Spatial Temporal Graph Convolution Network"
    )
    parser.add_argument("--val_split", type=int, default=0.2)
    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--log_dir", type=str, default=f"checkpoints/{exp_name}")
    parser.add_argument("--exp_name", type=str, default=exp_name)
    parser.add_argument(
        "--num_workers",
        type=int,
        default=multiprocessing.cpu_count(),
        help="the number of worker for data loader",
    )
    parser.add_argument("--clip_grad_norm", type=float, default=None)
    parser.add_argument("--writer_enabled", type=arg_boolean, default=True)
    parser.add_argument("--gcn0_flag", type=arg_boolean, default=False)
    parser.add_argument("--scheduling_lr", type=arg_boolean, default=True)
    parser.add_argument("--complete", type=arg_boolean, default=True)
    parser.add_argument("--bn_flag", type=arg_boolean, default=True)
    parser.add_argument("--accumulating_gradients", type=arg_boolean, default=True)
    parser.add_argument("--optimize_every", type=int, default=2)
    parser.add_argument("--validation_split", type=arg_boolean, default=False)
    parser.add_argument("--data_mirroring", type=arg_boolean, default=False)
    parser.add_argument("--local_rank", type=int, default=0)

    parser.add_argument(
        "--work-dir",
        default=f"./{exp_name}",
        help="the work folder for storing results",
    )
    parser.add_argument(
        "--config",
        default="config/st_gcn/custom_data/train.yaml",
        help="path to the configuration file",
    )

    # processor
    parser.add_argument("--phase", default="train", help="must be train or test")
    parser.add_argument(
        "--save_score",
        type=str2bool,
        default=True,
        help="if ture, the classification score will be stored",
    )

    # visulize and debug
    parser.add_argument(
        "--seed", type=int, default=13696642, help="random seed for pytorch"
    )
    parser.add_argument(
        "--training", type=str2bool, default=True, help="training or testing mode"
    )

    parser.add_argument(
        "--log-interval",
        type=int,
        default=100,
        help="the interval for printing messages (#iteration)",
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=1,
        help="the interval for storing models (#iteration)",
    )
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=10,
        help="the interval for evaluating models (#iteration)",
    )
    parser.add_argument(
        "--print-log", type=str2bool, default=True, help="print logging or not"
    )
    parser.add_argument(
        "--show-topk",
        type=int,
        default=[1, 5],
        nargs="+",
        help="which Top K accuracy will be shown",
    )

    # feeder
    parser.add_argument(
        "--feeder", default="feeder.Feeder", help="data loader will be used"
    )
    parser.add_argument(
        "--feeder_augmented",
        default="feeder.feeder_augmented",
        help="data loader will be used",
    )
    parser.add_argument(
        "--train-feeder-args",
        default=dict(),
        help="the arguments of data loader for training",
    )
    parser.add_argument(
        "--test-feeder-args",
        default=dict(),
        help="the arguments of data loader for test",
    )

    parser.add_argument(
        "--train_feeder_args_new",
        default=dict(),
        help="the arguments of data loader for training",
    )
    parser.add_argument(
        "--test_feeder_args_new",
        default=dict(),
        help="the arguments of data loader for test",
    )
    # model
    parser.add_argument("--model", default=None, help="the model will be used")
    parser.add_argument(
        "--model-args", type=dict, default=dict(), help="the arguments of model"
    )
    parser.add_argument(
        "--weights", default=None, help="the weights for network initialization"
    )
    parser.add_argument(
        "--ignore-weights",
        type=str,
        default=[],
        nargs="+",
        help="the name of weights which will be ignored in the initialization",
    )

    # optim
    parser.add_argument(
        "--scheduler", type=float, default=0, help="initial learning rate"
    )
    parser.add_argument(
        "--base-lr", type=float, default=0.1, help="initial learning rate"
    )
    parser.add_argument(
        "--step",
        type=int,
        default=[20, 40, 60],
        nargs="+",
        help="the epoch where optimizer reduce the learning rate",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=0,
        nargs="+",
        help="the indexes of GPUs for training or testing",
    )
    parser.add_argument("--optimizer", default="SGD", help="type of optimizer")
    parser.add_argument(
        "--nesterov", type=str2bool, default=False, help="use nesterov or not"
    )
    parser.add_argument(
        "--batch-size", type=int, default=256, help="training batch size"
    )
    parser.add_argument(
        "--test-batch-size", type=int, default=256, help="test batch size"
    )
    parser.add_argument(
        "--start-epoch", type=int, default=0, help="start training from which epoch"
    )
    parser.add_argument(
        "--num_epoch", type=int, default=120, help="stop training in which epoch"
    )
    parser.add_argument(
        "--weight-decay", type=float, default=0.0005, help="weight decay for optimizer"
    )
    parser.add_argument(
        "--display_by_category",
        type=str2bool,
        default=False,
        help="if ture, the top k accuracy by category  will be displayed",
    )
    parser.add_argument(
        "--display_recall_precision",
        type=str2bool,
        default=False,
        help="if ture, recall and precision by category  will be displayed",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default=None,
        help="mixed",
    )
    parser.add_argument(
        "--loss_fn",
        type=str,
        default=None,
        help="loss function to use for optimization",
    )
    return parser
