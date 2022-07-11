import pickle

import numpy as np
from st_gcn.feeder import tools
from torch.utils.data import Dataset
from yaml import Loader, load


class FeederCustom(Dataset):
    def __init__(
        self,
        data_path,
        label_path,
        random_choose=False,
        random_shift=False,
        random_move=False,
        window_size=-1,
        normalization=False,
        debug=False,
        use_mmap=True,
        channel=3,
    ):
        """

        :param data_path:
        :param label_path:
        :param random_choose: If true, randomly choose a portion of the input sequence
        :param random_shift: If true, randomly pad zeros at the begining or end of sequence
        :param random_move:
        :param window_size: The length of the output sequence
        :param normalization: If true, normalize input sequence
        :param debug: If true, only use the first 100 samples
        :param use_mmap: If true, use mmap mode to load data, which can save the running memory
        """
        self.debug = debug
        self.data_path = data_path
        self.label_path = label_path
        self.random_choose = random_choose
        self.random_shift = random_shift
        self.random_move = random_move
        self.window_size = window_size
        self.normalization = normalization
        self.use_mmap = use_mmap
        self.channel = channel
        with open("custom_data/config.yaml", "r") as f:
            self.config = load(f, Loader=Loader)

        self.load_data()
        if normalization:
            self.get_mean_map()

    def zero_runs(self, a):
        # Create an array that is 1 where a is 0, and pad each end with an extra 0.
        print(np.equal(a, 0))
        iszero = np.concatenate(([0], np.equal(a, 0).view(np.int8), [0]))
        absdiff = np.abs(np.diff(iszero))
        # Runs start and end where absdiff is 1.
        return np.where(absdiff == 1)[0].reshape(-1, 2)

    def load_data(self):
        # data: N C V T M

        try:
            with open(self.label_path) as f:
                self.activity_label, self.hand_activity_label = pickle.load(f)
        except Exception:
            # for pickle file from python2
            with open(self.label_path, "rb") as f:
                self.activity_label, self.hand_activity_label = pickle.load(
                    f, encoding="latin1"
                )

        # load data
        if self.use_mmap:
            self.data = np.memmap(
                self.data_path,
                mode="r",
                shape=(
                    len(self.activity_label),
                    self.channel,
                    self.config["max_frames"],
                    sum(list(self.config["feature_length"].values())),
                    1,
                ),
            )
            print(len(self.activity_label), self.data.shape)
        else:
            self.data = np.load(self.data_path)
        if self.debug:
            self.hand_activity_label = self.hand_activity_label[:100]
            self.data = self.data[:100]
            self.activity_label = self.activity_label[:100]
        self.N, self.C, self.T, self.V, self.M = self.data.shape

    def get_mean_map(self):
        data = self.data
        N, C, T, V, M = data.shape
        self.mean_map = (
            data.mean(axis=2, keepdims=True).mean(axis=4, keepdims=True).mean(axis=0)
        )
        self.std_map = (
            data.transpose((0, 2, 4, 1, 3))
            .reshape((N * T * M, C * V))
            .std(axis=0)
            .reshape((C, 1, V, 1))
        )

    def __len__(self):
        return len(self.hand_activity_label)

    def __iter__(self):
        return self

    def __getitem__(self, index):
        activity_label = self.activity_label[index]

        data_numpy = self.data[index]
        hand_activity_label = self.hand_activity_label[index]
        data_numpy = np.array(data_numpy)

        if self.normalization:
            data_numpy = (data_numpy - self.mean_map) / self.std_map
        if self.random_shift:
            data_numpy = tools.random_shift(data_numpy)
        if self.random_choose:
            data_numpy = tools.random_choose(data_numpy, self.window_size)
        elif self.window_size > 0:
            data_numpy = tools.auto_pading(data_numpy, self.window_size)
        if self.random_move:
            data_numpy = tools.random_move(data_numpy)
        return data_numpy, activity_label, hand_activity_label

    def top_k(self, score, top_k, label_type):
        label = (
            self.hand_activity_label if label_type == "hand" else self.activity_label
        )
        rank = score.argsort()
        hit_top_k = [l in rank[i, -top_k:] for i, l in enumerate(label)]
        return sum(hit_top_k) * 1.0 / len(hit_top_k)
