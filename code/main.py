#!/usr/bin/env python

import os
import pickle
from typing import List

import adamod
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from argument_parser import get_parser
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    f1_score,
    multilabel_confusion_matrix,
)
from tensorboardX import SummaryWriter
from tools.accuracy import multi_label_accuracy, single_label_accuracy
from tools.seesaw import SeesawLoss
from tools.weighted_BCE import focal_weighted_bce
from torch.cuda.amp import GradScaler
from utils.misc import import_class, save_arg
from utils.time_utils import TimeKeeper
from utils.train_utils import (
    adjust_learning_rate,
    opt_update,
    save_checkpoint,
    scaled_opt_update,
)
from yaml import Loader, load

NAME_EXP = "NTT_test"
CURRENT_EXP = "focal_(len_wt)^0.5"
writer = SummaryWriter(f"./{NAME_EXP}")
np.random.seed(13696641)
torch.manual_seed(13696641)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Processor:
    """
    Processor for Skeleton-based Action Recgnition
    """

    def __init__(self, arg):
        self.arg = arg
        self.load_data()

        with open("custom_data/config.yaml", "r") as f:
            self.data_config = load(f, Loader=Loader)
        self.num_points = sum(list(self.data_config["feature_length"].values()))

        self.time_keeper = TimeKeeper(arg)
        self.load_model()
        self.load_optimizer()
        self.graph = nx.Graph()

        save_arg(arg)

        self.best_epoch = 0
        self.best_accuracy = 0
        self.params = arg

    def load_checkpoint(self, path, filename):
        ckpt_path = os.path.join(path, filename)

        checkpoint = torch.load(ckpt_path, map_location=torch.device(DEVICE))
        self.epoch = checkpoint["epoch"]
        self.best_epoch = checkpoint["best_epoch"]
        self.best_epoch_score = checkpoint["best_epoch_score"]
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    def load_data(self):
        Feeder = import_class(self.arg.feeder)

        self.data_loader = {}
        self.trainLoader = Feeder(
            **self.arg.train_feeder_args,
            channel=self.arg.model_args["channel"],
        )
        self.sample_weights = torch.from_numpy(self.trainLoader.get_weights()).to(
            DEVICE
        )
        self.testLoader = Feeder(
            **self.arg.test_feeder_args,
            channel=self.arg.model_args["channel"],
        )

        if arg.validation_split:
            val_size = int(0.2 * len(self.trainLoader))

            self.trainLoader, self.valLoader = torch.utils.data.random_split(
                self.trainLoader, [len(self.trainLoader) - val_size, val_size]
            )

        # FIX ME SHUFFLE
        if self.arg.phase == "train":
            self.data_loader["train"] = torch.utils.data.DataLoader(
                dataset=self.trainLoader,
                batch_size=self.arg.batch_size,
                shuffle=True,
                num_workers=self.arg.num_workers,
            )

        self.data_loader["val"] = torch.utils.data.DataLoader(
            dataset=self.testLoader,
            batch_size=self.arg.test_batch_size,
            shuffle=False,
            num_workers=self.arg.num_workers,
        )
        self.data_loader["test"] = torch.utils.data.DataLoader(
            dataset=self.testLoader,
            batch_size=self.arg.test_batch_size,
            shuffle=False,
            num_workers=self.arg.num_workers,
        )

    def load_model(self) -> None:
        assert type(self.arg.model_args["num_class"]) == list, (
            "Num of classes for each classifier head"
            "(single as well) should be provided as a list"
        )
        Model = import_class(self.arg.model)
        # self.model = Model(**self.arg.model_args).to(DEVICE)
        self.model = nn.DataParallel(
            Model(**self.arg.model_args, num_point=self.num_points).to(DEVICE)
        )
        loss_fn = self.arg.model_args["loss_fn"]
        if loss_fn == "cce":
            self.loss = [
                nn.CrossEntropyLoss().to(DEVICE)
                for _ in self.arg.model_args["num_class"]
            ]
        elif loss_fn == "seesaw":
            self.loss = [
                SeesawLoss(num_classes=num_class).to(DEVICE)
                for num_class in self.arg.model_args["num_class"]
            ]
        elif loss_fn == "multilabel":
            # self.loss = [
            #     torch.nn.BCELoss().to(DEVICE) for _ in self.arg.model_args["num_class"]
            # ]
            self.loss = [
                focal_weighted_bce(self.sample_weights)
                for _ in self.arg.model_args["num_class"]
            ]
        else:
            raise ValueError(f"loss_fn type '{loss_fn}' is not a valid option")

        if self.arg.weights:
            self.time_keeper.print_log(f"Load weights from {self.arg.weights}.")
            if ".pkl" in self.arg.weights:
                with open(self.arg.weights, "r") as f:
                    weights = pickle.load(f)
            else:
                weights = torch.load(
                    self.arg.weights, map_location=torch.device(DEVICE)
                )

            for w in self.arg.ignore_weights:
                if weights.pop(w, None) is not None:
                    self.time_keeper.print_log(f"Sucessfully Remove Weights: {w}.")
                else:
                    self.time_keeper.print_log(f"Can Not Remove Weights: {w}.")

            try:
                self.model.load_state_dict(weights)
            except Exception:
                state = self.model.state_dict()
                diff = list(set(state.keys()).difference(set(weights.keys())))
                print("Can not find these weights:")
                for d in diff:
                    print(f"  {d}")
                state.update(weights)
                self.model.load_state_dict(state)

            # If the pretrained models are without the DataParallel wrapper
            # self.model = nn.DataParallel(self.model)
            # torch.save(self.model.state_dict(), self.arg.weights.replace("_temporal.pt", "_temporal_data.pt"))

    def load_optimizer(self):
        if self.arg.optimizer == "SGD":
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.arg.base_lr,
                momentum=0.9,
                nesterov=self.arg.nesterov,
                weight_decay=self.arg.weight_decay,
            )

            optim.SGD
        elif self.arg.optimizer == "Adam":
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.arg.base_lr,
                weight_decay=self.arg.weight_decay,
            )
        elif self.arg.optimizer == "Adamod":
            self.optimizer = adamod.AdaMod(
                self.model.parameters(), lr=self.arg.base_lr, beta3=0.999
            )
            print("Using Adamod")

        else:
            raise ValueError()

    def train(
        self, epoch: int, save_model: bool = True, scaler: GradScaler = None
    ) -> None:

        self.model.train()
        self.time_keeper.print_log(f"Training epoch: [{epoch+1}/{self.arg.num_epoch}]")
        loader = self.data_loader["train"]
        lr = adjust_learning_rate(self.arg, self.optimizer, epoch)
        loss_value = []
        train_total = 0
        train_correct = [0] * len(self.arg.model_args["num_class"])

        # Set the current_time
        self.time_keeper.record_time()
        timer = dict(dataloader=0.001, model=0.001, statistics=0.001)

        if not arg.accumulating_gradients:
            arg.optimize_every == 1

        running_batches = 0
        real_batch_index = 0
        tot_num_batches = len(loader)
        running_optimize_every = min(
            arg.optimize_every, tot_num_batches - running_batches
        )
        self.optimizer.zero_grad()
        for batch_idx, (data, labels) in enumerate(loader):

            data = data.float().to(DEVICE)
            labels = [label.to(DEVICE) for label in labels]
            if arg.model_args["loss_fn"] == "multilabel":
                labels = [label.float() for label in labels]
            timer["dataloader"] += self.time_keeper.split_time()

            # forward
            if scaler is not None:
                with torch.cuda.amp.autocast(enabled=True):
                    outputs = self.model(data)
                    loss = sum(
                        loss_fn(output, label)
                        for output, label, loss_fn in zip(outputs, labels, self.loss)
                    )
            else:
                outputs = self.model(data)
                loss = sum(
                    loss_fn(output, label)
                    for output, label, loss_fn in zip(outputs, labels, self.loss)
                )
            loss_value.append(loss.item())

            # Metrics
            if arg.model_args["loss_fn"] == "multilabel":
                acc, predictions, train_total, train_correct = multi_label_accuracy(
                    outputs, labels, train_total, train_correct
                )
                acc /= sum(self.arg.model_args["num_class"])
            else:
                acc, predictions, train_total, train_correct = single_label_accuracy(
                    outputs, labels, train_total, train_correct
                )
                acc /= len(self.arg.model_args["num_class"])

            # backward
            loss /= running_optimize_every
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            timer["model"] += self.time_keeper.split_time()

            if (batch_idx + 1) % running_optimize_every == 0:
                # Weight update
                if scaler is not None:
                    scaled_opt_update(
                        scaler, self.optimizer, self.model, arg.clip_grad_norm
                    )
                else:
                    opt_update(self.optimizer, self.model, arg.clip_grad_norm)

                # Print statistics every 'n' batches
                if (real_batch_index + 1) % self.arg.log_interval == 0:
                    step = epoch * (len(loader) / arg.optimize_every) + real_batch_index

                    self.train_logging(
                        epoch,
                        batch_idx,
                        step,
                        tot_num_batches,
                        loss,
                        acc,
                        lr,
                    )

                running_optimize_every = min(
                    arg.optimize_every, tot_num_batches - (batch_idx + 1)
                )
                real_batch_index += 1

            timer["statistics"] += self.time_keeper.split_time()

        self.train_logging(
            epoch,
            batch_idx,
            step,
            tot_num_batches,
            loss,
            acc,
            lr,
            timer,
            loss_value,
        )
        for ind, predicts in enumerate(predictions):
            print(f"Here are the just predicted labels{ind+1}: {predicts.detach()}")
            print(f"Here are the correct labels{ind+1}: {labels[ind].detach()}")

    def train_logging(
        self,
        epoch: int,
        batch_idx: int,
        step: int,
        total_batches: int,
        loss,
        acc: float,
        lr: float,
        timer: dict = None,
        loss_value: list = None,
    ) -> None:

        # Get training statistics.
        self.time_keeper.print_log(
            (
                f"\tBatch({batch_idx + 1}/{total_batches}) done. Loss: {loss.item():.4f},"
                f"Training Accuracy: {acc:.4f}  lr:{lr:.6f}"
            )
        )

        # Print tensorboard info
        info = {"loss-train": loss, "accuracy-train": acc}
        for tag, value in info.items():
            writer.add_scalar(tag, value, step)

        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                writer.add_scalar(f"gradients/{name}", param.grad.norm(2).item(), step)
        # statistics of time consumption and loss
        if loss_value is not None:
            self.time_keeper.print_log(
                f"\tMean training loss: {np.mean(loss_value):.4f}."
            )
        if timer is not None:
            proportion = {
                k: f"{int(round(v * 100 / sum(timer.values()))):02d}%"
                for k, v in timer.items()
            }
            self.time_keeper.print_log(
                "\tTime consumption: {}, [Data]{dataloader}, [Network]{model}".format(
                    sum(timer.values()), **proportion
                )
            )
            print("saving!")
            model_path = f"{self.arg.work_dir}/epoch{epoch + 1}_model.pt"
            state_dict = {
                "epoch": epoch,
                "best_epoch": self.best_epoch,
                "best_epoch_score": self.best_accuracy,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            }
            save_checkpoint(self.arg.work_dir, f"epoch{epoch+1}.ckpt", state_dict)
            torch.save(self.model.state_dict(), model_path)

    def test(self, epoch: int, save_score: bool = True, loader_name: List[str] = None):
        if loader_name is None:
            loader_name = ["test"]
        self.model.eval()
        self.time_keeper.print_log(f"Eval epoch: {epoch + 1}")

        val_total = 0
        val_correct = [0] * len(self.arg.model_args["num_class"])
        class_correct = [
            [0.0] * num_classes for num_classes in self.arg.model_args["num_class"]
        ]
        class_total = [
            [0.0] * num_classes for num_classes in self.arg.model_args["num_class"]
        ]
        class_labels = [[]] * len(self.arg.model_args["num_class"])
        class_outputs = [[]] * len(self.arg.model_args["num_class"])

        with torch.no_grad():
            for ln in loader_name:
                for data, labels in self.data_loader[ln]:
                    data = data.float().to(DEVICE)
                    labels = [label.to(DEVICE) for label in labels]

                    outputs = self.model(data)
                    (
                        val_accuracy,
                        predictions,
                        val_total,
                        val_correct,
                    ) = multi_label_accuracy(outputs, labels, val_total, val_correct)
                    val_accuracy /= sum(self.arg.model_args["num_class"])
                    # Storing validation set labels and targets
                    for ind, label in enumerate(labels):
                        class_labels[ind].append(label.cpu())
                        class_outputs[ind].append(outputs[ind].cpu())

        info = {"accuracy-test": val_accuracy}
        for ind, num_classes in enumerate(self.arg.model_args["num_class"]):
            conf_matrix_test = multilabel_confusion_matrix(
                np.concatenate(class_labels[ind], axis=0),
                np.round(np.concatenate(class_outputs[ind], axis=0)),
                labels=np.arange(num_classes),
            )
            path = os.path.join(NAME_EXP, CURRENT_EXP)
            os.makedirs(path, exist_ok=True)
            np.save(
                os.path.join(path, f"confusion_test_{epoch}_{ind + 1}"),
                conf_matrix_test,
            )
            f, axes = plt.subplots(
                num_classes // 5 if num_classes % 5 == 0 else num_classes // 5 + 1,
                5,
                figsize=(
                    2
                    * (
                        num_classes // 5
                        if num_classes % 5 == 0
                        else num_classes // 5 + 1
                    ),
                    5 + 4,
                ),
            )
            axes = axes.ravel()
            for i, ax in enumerate(axes):
                if i < num_classes:
                    matrix = conf_matrix_test[i] / np.expand_dims(
                        np.sum(conf_matrix_test[i], axis=-1), -1
                    )
                    matrix = np.nan_to_num(matrix, posinf=0)
                    if matrix[0, 0] == 1:
                        ax.set_visible(False)
                        continue

                    disp = ConfusionMatrixDisplay(
                        np.round_(matrix, decimals=2), display_labels=[0, i + 1]
                    )
                    disp.plot(ax=ax, values_format=".4g")
                    disp.ax_.set_title(f"class {i + 1}")

                    disp.ax_.set_xlabel("")
                    disp.ax_.set_ylabel("")
                    disp.im_.colorbar.remove()
                else:
                    ax.set_visible(False)

            plt.subplots_adjust(wspace=0.35, hspace=0.01)
            f.colorbar(disp.im_, ax=axes, fraction=0.046, pad=0.04)
            f.suptitle(f"{CURRENT_EXP}", fontsize=16)
            f.supxlabel("Predicted Labels")
            f.supylabel("True Labels")
            plt.savefig(
                os.path.join(path, f"confusion_test_{epoch}_{ind + 1}.jpg"), dpi=150
            )

        # if arg.display_recall_precision:
        #     precision, recall = self.data_loader[
        #         ln
        #     ].dataset.calculate_recall_precision(score)
        #     for i in range(len(precision)):
        #         self.time_keeper.print_log(
        #             f"\tClass{i + 1} Precision: {100 * precision[i]:.2f}%, Recall: {100 * recall[i]:.2f}%"
        #         )

        # for k in self.arg.show_topk:
        #     if arg.display_by_category:
        #         accuracy = self.data_loader[ln].dataset.top_k_by_category(
        #             score, k
        #         )
        #         for i in range(score.shape[1]):
        #             self.time_keeper.print_log(
        #                 f"\tClass{i + 1} Top{k}: {100 * accuracy[i]:.2f}%"
        #             )
        #         self.time_keeper.print_log(
        #             f"\tTop{k}: {100 * sum(accuracy) / len(accuracy):.2f}%"
        #         )
        #     else:
        #         self.time_keeper.print_log(
        #             f"\tTop{k}: {100 * self.data_loader[ln].dataset.top_k(score, k):.2f}%"
        #         )
        stats_val = (
            f"Validation: Epoch [{epoch}/{self.arg.num_epoch}]"
            f"Validation Accuracy: {val_accuracy}"
        )

        print(f"\n{stats_val}")

    def val(self, epoch: int, save_score: bool = False, loader_name: list = None):
        if loader_name is None:
            loader_name = ["val"]
        self.model.eval()
        self.time_keeper.print_log(f"Eval epoch: {epoch + 1}")

        val_total = 0
        val_correct = [0] * len(self.arg.model_args["num_class"])
        class_correct = [
            [0.0] * num_classes for num_classes in self.arg.model_args["num_class"]
        ]
        class_total = [
            [0.0] * num_classes for num_classes in self.arg.model_args["num_class"]
        ]
        class_labels = [[]] * len(self.arg.model_args["num_class"])
        class_outputs = [[]] * len(self.arg.model_args["num_class"])

        with torch.no_grad():
            for ln in loader_name:
                loss_value = []
                for data, labels in self.data_loader[ln]:
                    data = data.float().to(DEVICE)
                    labels = [label.to(DEVICE) for label in labels]
                    if arg.model_args["loss_fn"] == "multilabel":
                        labels = [label.float() for label in labels]

                    outputs = self.model(data)
                    loss = sum(
                        loss_fn(output, label)
                        for output, label, loss_fn in zip(outputs, labels, self.loss)
                    )
                    loss_value.append(loss.item())

                    # Metrics
                    if arg.model_args["loss_fn"] == "multilabel":
                        (
                            val_accuracy,
                            predictions,
                            val_total,
                            val_correct,
                        ) = multi_label_accuracy(
                            outputs, labels, val_total, val_correct
                        )
                        val_accuracy /= sum(self.arg.model_args["num_class"])
                        # Storing validation set labels and targets
                        for ind, label in enumerate(labels):
                            class_labels[ind].append(label.cpu())
                            class_outputs[ind].append(outputs[ind].cpu())
                    else:
                        (
                            val_accuracy,
                            predictions,
                            val_total,
                            val_correct,
                        ) = single_label_accuracy(
                            outputs, labels, val_total, val_correct
                        )
                        val_accuracy /= len(self.arg.model_args["num_class"])

                        c = [
                            (label == predicts.squeeze()).float()
                            for label, predicts in zip(labels, predictions)
                        ]

                        # Calculating validation accuracy for each class
                        for ind, label in enumerate(labels):
                            for l in range(label.size(0)):
                                class_label = label[l]
                                class_correct[ind][class_label - 1] += c[ind][l]
                                class_total[ind][class_label - 1] += 1

                info = {"loss-val": loss, "accuracy-val": val_accuracy}

                self.time_keeper.print_log(
                    f"\tMean {ln} loss of {len(self.data_loader[ln])} batches: {np.mean(loss_value)}."
                )

                for ind, predicts in enumerate(predictions):
                    print(f"Here are the just predicted labels{ind+1}: {predicts}")
                    print(f"Here are the correct labels{ind+1}: {labels[ind]}")
                print("Total samples seen so far: ", val_total)

                stats_val = (
                    f"Validation: Epoch [{epoch}/{self.arg.num_epoch}], Loss: {loss.item()}"
                    f"Validation Accuracy: {val_accuracy}"
                )

                print(f"\n{stats_val}")
                if arg.model_args["loss_fn"] == "multilabel":
                    for ind in range(len(self.arg.model_args["num_class"])):
                        print(f"f1-scores for {ind+1}th classifier:")
                        f1_vals = f1_score(
                            np.concatenate(class_labels[ind], axis=0),
                            np.round(np.concatenate(class_outputs[ind], axis=0)),
                            average=None,
                        )
                        vals = {"exp": [CURRENT_EXP]}
                        vals.update(
                            {f"f1_{i + 1}": [val] for i, val in enumerate(f1_vals)}
                        )
                        df = pd.DataFrame(vals)
                        csv_path = os.path.join(NAME_EXP, f"results_{ind + 1}.csv")
                        df.to_csv(
                            csv_path,
                            index=False,
                            mode="a",
                            header=not os.path.exists(csv_path),
                        )
                else:
                    for ind, num_classes in enumerate(self.arg.model_args["num_class"]):
                        print(f"class accuracies for {ind}th head")
                        for i in range(num_classes):
                            if class_total[ind][i] != 0:
                                print(
                                    (
                                        f"Accuracy of {i + 1} : {int(class_correct[ind][i])} / {int(class_total[ind][i])} ="
                                        f" {int(100 * class_correct[ind][i] / class_total[ind][i])} %"
                                    )
                                )

                step = (epoch + 1) * (
                    len(self.data_loader["train"]) / (arg.optimize_every)
                )

                for tag, value in info.items():
                    writer.add_scalar(tag, value, step)

        return val_accuracy

    def start(self) -> None:

        if not self.arg.training:
            self.test(epoch=0, save_score=self.arg.save_score, loader_name=["test"])

        patient_counter = 0

        pytorch_total_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        layer_params = sum(p.view(-1).shape[0] for p in self.model.parameters())
        print(f"Params: {pytorch_total_params}")
        print(f"Layer params: {layer_params}")

        if self.arg.phase == "train":
            patience = 50
            self.time_keeper.print_log(f"Parameters:\n{vars(self.arg)}\n")
            scaler = (
                torch.cuda.amp.GradScaler() if self.arg.precision == "amp" else None
            )
            for epoch in range(self.arg.start_epoch, self.arg.num_epoch):

                save_model = ((epoch + 1) % self.arg.save_interval == 0) or (
                    epoch + 1 == self.arg.num_epoch
                )
                eval_model = ((epoch + 1) % self.arg.eval_interval == 0) or (
                    epoch + 1 == self.arg.num_epoch
                )

                self.train(epoch, save_model=save_model, scaler=scaler)
                if eval_model:
                    accuracy = self.val(
                        epoch, save_score=self.arg.save_score, loader_name=["val"]
                    )
                    if accuracy <= self.best_accuracy:
                        patient_counter += 1
                    else:
                        self.best_epoch = epoch
                        self.best_accuracy = accuracy
                        patient_counter = 0
                    if patient_counter == patience:
                        print("Early stopped!")
                        break

            load_model_path = os.path.join(NAME_EXP, f"epoch{epoch+1}_model.pt")
            self.time_keeper.print_log(f"Load weights from {load_model_path}.")
            weights = torch.load(load_model_path)

            for w in self.arg.ignore_weights:
                if weights.pop(w, None) is not None:
                    self.time_keeper.print_log(f"Sucessfully Remove Weights: {w}.")
                else:
                    self.time_keeper.print_log(f"Can Not Remove Weights: {w}.")

            try:
                self.model.load_state_dict(weights)
            except Exception:
                state = self.model.state_dict()
                diff = list(set(state.keys()).difference(set(weights.keys())))
                print("Can not find these weights:")
                for d in diff:
                    print(f"  {d}")
                state.update(weights)
                self.model.load_state_dict(state)
            # self.test(epoch=0, save_score=self.arg.save_score, loader_name=["test"])
            self.time_keeper.print_log("Done.\n")

        elif self.arg.phase == "test":
            if self.arg.weights is None:
                raise ValueError("Please appoint --weights.")
            self.arg.print_log = False
            self.time_keeper.print_log(f"Model:   {self.arg.model}.")
            self.time_keeper.print_log(f"Weights: {self.arg.weights}.")

            self.test(epoch=0, save_score=self.arg.save_score, loader_name=["test"])
            self.time_keeper.print_log("Done.\n")


if __name__ == "__main__":

    parser = get_parser(NAME_EXP)
    p = parser.parse_args()
    if p.config is not None:
        with open(p.config, "r") as f:
            default_arg = yaml.load(f, Loader=Loader)
        key = vars(p).keys()
        for k in default_arg.keys():
            if k not in key:
                print(f"WRONG ARG: {k}")
                assert k in key
        parser.set_defaults(**default_arg)

    arg = parser.parse_args()
    processor = Processor(arg)
    processor.start()
