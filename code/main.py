#!/usr/bin/env python

import os
import pickle

import adamod
import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from argument_parser import get_parser
from sklearn.metrics import confusion_matrix
from tensorboardX import SummaryWriter
from utils.misc import import_class, save_arg
from utils.time_utils import TimeKeeper
from utils.train_utils import adjust_learning_rate, save_checkpoint

NAME_EXP = "NTT_test"
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
        self.load_model()
        self.load_optimizer()
        self.graph = nx.Graph()

        self.time_keeper = TimeKeeper(arg)
        save_arg(arg)

        self.best_epoch = 0
        self.seen = 0
        self.best_accuracy = 0
        self.params = arg
        self.num_joints = 25

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
        self.trainLoader = Feeder(**self.arg.train_feeder_args)
        self.testLoader = Feeder(**self.arg.test_feeder_args)

        print(self.trainLoader == self.testLoader)
        if arg.validation_split:
            val_size = int(0.2 * len(self.trainLoader))

            self.trainLoader, self.valLoader = torch.utils.data.random_split(
                self.trainLoader, [len(self.trainLoader) - val_size, val_size]
            )

        # print("Train size: ", len(self.trainLoaderNew))
        # print("Test size: ", len(self.testLoaderNew))

        # FIX ME SHUFFLE

        if self.arg.phase == "train":
            self.data_loader["train"] = torch.utils.data.DataLoader(
                dataset=self.trainLoader,
                batch_size=self.arg.batch_size,
                shuffle=True,
                num_workers=self.arg.num_worker,
            )

        self.data_loader["val"] = torch.utils.data.DataLoader(
            dataset=self.testLoader,
            batch_size=self.arg.test_batch_size,
            shuffle=False,
            num_workers=self.arg.num_worker,
        )
        self.data_loader["test"] = torch.utils.data.DataLoader(
            dataset=self.testLoader,
            batch_size=self.arg.test_batch_size,
            shuffle=False,
            num_workers=self.arg.num_worker,
        )

    def load_model(self):
        Model = import_class(self.arg.model)
        # self.model = Model(**self.arg.model_args).to(DEVICE)
        self.model = nn.DataParallel(Model(**self.arg.model_args).to(DEVICE))
        self.loss = nn.CrossEntropyLoss().to(DEVICE)

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
            except:
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

    def train(self, epoch, save_model=True):

        self.model.train()
        self.time_keeper.print_log(f"Training epoch: {epoch + 1}")
        loader = self.data_loader["train"]
        lr = adjust_learning_rate(self.arg, self.optimizer, epoch)
        loss_value = []
        train_total = 0
        train_correct = 0

        # Set the current_time
        self.time_keeper.record_time()
        timer = dict(dataloader=0.001, model=0.001, statistics=0.001)

        if not arg.accumulating_gradients:
            for batch_idx, (data, label, name) in enumerate(loader):
                data = data.float().to(DEVICE)
                label = label.long().to(DEVICE)

                timer["dataloader"] += self.time_keeper.split_time()

                name = name[0]
                output = self.model(data, label, name)
                loss = self.loss(output, label)

                # backward
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                loss_value.append(loss.data.item())
                _, predictions = torch.max(output, 1)
                train_total = train_total + label.size(0)
                train_correct = train_correct + (predictions == label).sum().item()
                acc = 100 * train_correct / train_total
                timer["model"] += self.time_keeper.split_time()

                info = {"loss-Train": loss, "accuracy-Train": acc}

                # Print statistics every 100 batches
                if (batch_idx + 1) % 200 == 0:
                    print("Total samples seen so far: ", train_total)
                    print("Here are the just predicted labels: ", predictions)
                    print("Here are the correct labels: ", label)

                    # Get training statistics.
                    stats_train = f"Training: Epoch [{epoch}/{self.arg.num_epoch}], Step [{batch_idx}], Loss: {loss.item()}, Training Accuracy: {acc}"
                    print(f"\n{stats_train}")

                    step = epoch * len(loader) + batch_idx

                    # Print tensorboard info
                    for tag, value in info.items():
                        writer.add_scalar(tag, value, step)

                    for name, param in self.model.named_parameters():
                        if param.requires_grad and param.grad is not None:
                            writer.add_scalar(
                                "gradients/" + name, param.grad.norm(2).item(), step
                            )

                # statistics
                if batch_idx % self.arg.log_interval == 0:
                    self.time_keeper.print_log(
                        f"\tBatch({batch_idx}/{len(loader)}) done. Loss: {loss.data.item():.4f}  lr:{lr:.6f}"
                    )
                timer["statistics"] += self.time_keeper.split_time()

            # statistics of time consumption and loss
            proportion = {
                k: f"{int(round(v * 100 / sum(timer.values()))):02d}%"
                for k, v in timer.items()
            }
            self.time_keeper.print_log(
                f"\tMean training loss: {np.mean(loss_value):.4f}."
            )
            self.time_keeper.print_log(
                "\tTime consumption: [Data]{dataloader}, [Network]{model}".format(
                    **proportion
                )
            )

            if True:
                print("saving!")
                model_path = f"{self.arg.work_dir}/epoch{epoch + 1}_model.pt"
                torch.save(self.model.state_dict(), model_path)
        else:
            running_loss = 0
            running_batches = 0
            real_batch_index = 0
            running_samples = 0
            tot_num_batches = len(loader)
            running_optimize_every = min(
                arg.optimize_every, tot_num_batches - running_batches
            )
            self.optimizer.zero_grad()

            for batch_idx, (data, label, name) in enumerate(loader):
                if batch_idx > 10:
                    break

                data = data.float().to(DEVICE)
                label = label.long().to(DEVICE)

                timer["dataloader"] += self.time_keeper.split_time()

                # forward

                output = self.model(data, label, name)
                loss = self.loss(output, label)
                loss_norm = loss / running_optimize_every

                # backward
                loss_norm.backward()

                _, predictions = torch.max(output, 1)
                train_total = train_total + label.size(0)
                train_correct = train_correct + (predictions == label).sum().item()
                acc = 100 * train_correct / train_total
                timer["model"] += self.time_keeper.split_time()

                info = {"loss-Train": loss, "accuracy-Train": acc}

                # Updating running_loss and seen samples
                running_loss = running_loss + loss.item()
                running_batches = running_batches + 1
                self.seen = self.seen + label.size(0)
                running_samples = running_samples + label.size(0)

                if running_batches % running_optimize_every == 0:

                    if arg.clip:
                        torch.nn.utils.clip_grad.clip_grad_norm_(
                            self.model.parameters(), 1
                        )

                    # Step
                    self.optimizer.step()
                    loss_value.append(loss.data.item())

                    step = (
                        epoch * (len(loader) / (arg.optimize_every)) + real_batch_index
                    )

                    for name, param in self.model.named_parameters():
                        if param.requires_grad and param.grad is not None:
                            # logger.scalar_summary("gradients/" + name, param.grad.norm(2).item(), global_step)
                            writer.add_scalar(
                                "gradients/" + name, param.grad.norm(2).item(), step
                            )

                    # Print statistics every 100 batches
                    if (real_batch_index + 1) % 200 == 0:
                        # accuracy = self.evaluate(epoch, epochs, self.sub_dataVal)

                        print("Total samples seen so far: ", train_total)
                        print("Here are the just predicted labels: ", predictions)
                        print("Here are the correct labels: ", label)

                        # Get training statistics.
                        stats_train = f"Training: Epoch [{epoch}/{self.arg.num_epoch}], Step [{batch_idx}], Loss: {loss.item()}, Training Accuracy: {acc}"
                        print("\n" + stats_train)
                        step = (
                            epoch * (len(loader) / arg.optimize_every)
                            + real_batch_index
                        )

                        if True:
                            print("saving!")
                            model_path = (
                                f"{self.arg.work_dir}/epoch{epoch + 1}_model.pt"
                            )
                            state_dict = {
                                "epoch": epoch,
                                "best_epoch": self.best_epoch,
                                "best_epoch_score": self.best_accuracy,
                                "model_state_dict": self.model.state_dict(),
                                "optimizer_state_dict": self.optimizer.state_dict(),
                            }
                            save_checkpoint(
                                self.arg.work_dir, f"epoch{epoch+1}.ckpt", state_dict
                            )
                            torch.save(self.model.state_dict(), model_path)

                        # Print tensorboard info
                        for tag, value in info.items():
                            writer.add_scalar(tag, value, step)

                        for name, param in self.model.named_parameters():
                            if param.requires_grad and param.grad is not None:
                                writer.add_scalar(
                                    "gradients/" + name, param.grad.norm(2).item(), step
                                )

                    self.optimizer.zero_grad()
                    running_optimize_every = min(
                        arg.optimize_every, tot_num_batches - running_batches
                    )
                    real_batch_index = real_batch_index + 1

                    # statistics
                if batch_idx % self.arg.log_interval == 0:
                    self.time_keeper.print_log(
                        f"\tBatch({batch_idx}/{len(loader)}) done. Loss: {loss.data.item():.4f}  lr:{lr:.6f}"
                    )
                timer["statistics"] += self.time_keeper.split_time()

            # statistics of time consumption and loss
            proportion = {
                k: f"{int(round(v * 100 / sum(timer.values()))):02d}%"
                for k, v in timer.items()
            }
            self.time_keeper.print_log(
                f"\tMean training loss: {np.mean(loss_value):.4f}."
            )
            self.time_keeper.print_log(
                "\tTime consumption: [Data]{dataloader}, [Network]{model}".format(
                    **proportion
                )
            )

            if True:
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

    def test(self, epoch, save_score=True, loader_name=["test"]):
        self.model.eval()
        self.time_keeper.print_log(f"Eval epoch: {epoch + 1}")
        val_correct = 0
        val_total = 0
        conf_matrix_test = 0
        class_correct = [0.0] * self.arg.model_args["num_class"]
        class_total = [0.0] * self.arg.model_args["num_class"]

        with torch.no_grad():
            for ln in loader_name:
                loss_value = []
                score_frag = []
                for batch_idx, (data, label, name) in enumerate(self.data_loader[ln]):
                    data = data.float().to(DEVICE)
                    label = label.long().to(DEVICE)

                    name = name[0]
                    output = self.model(data, label, name)
                    loss = self.loss(output, label)
                    score_frag.append(output.data.cpu().numpy())
                    loss_value.append(loss.data.item())

                    _, predictions = torch.max(output, 1)
                    val_total = val_total + label.size(0)
                    val_correct = (
                        val_correct + (predictions == label).double().sum().item()
                    )
                    val_accuracy = (val_correct / val_total) * 100
                    c = (label == predictions.squeeze()).float()
                    c.float().mean()

                    # Calculating validation accuracy for each class
                    for l in range(label.size(0)):
                        class_label = label[l]
                        class_correct[class_label - 1] = (
                            class_correct[class_label - 1] + c[l]
                        )
                        class_total[class_label - 1] = class_total[class_label - 1] + 1

                    # print("Test accuracy on batch: ", testing_accuracy_batch)
                    info = {"loss-Val": loss, "accuracy-test": val_accuracy}
                    conf_matrix_test += confusion_matrix(
                        predictions.cpu(),
                        label.cpu(),
                        labels=np.arange(self.arg.model_args["num_class"]),
                    )
                    np.save(
                        os.path.join(NAME_EXP, f"confusion_test_{epoch}"),
                        conf_matrix_test,
                    )

                score = np.concatenate(score_frag)

                # Added
                loss = np.mean(loss_value)
                score_dict = dict(zip(self.data_loader[ln].dataset.sample_name, score))
                if True:
                    with open(
                        f"{self.arg.work_dir}/epoch{epoch + 1}_{ln}_score.pkl", "wb"
                    ) as f:
                        pickle.dump(score_dict, f)

                self.time_keeper.print_log(
                    f"\tMean {ln} loss of {len(self.data_loader[ln])} batches: {np.mean(loss_value)}."
                )

                if arg.display_recall_precision:
                    precision, recall = self.data_loader[
                        ln
                    ].dataset.calculate_recall_precision(score)
                    for i in range(len(precision)):
                        self.time_keeper.print_log(
                            f"\tClass{i + 1} Precision: {100 * precision[i]:.2f}%, Recall: {100 * recall[i]:.2f}%"
                        )

                for k in self.arg.show_topk:
                    if arg.display_by_category:
                        accuracy = self.data_loader[ln].dataset.top_k_by_category(
                            score, k
                        )
                        for i in range(score.shape[1]):
                            self.time_keeper.print_log(
                                f"\tClass{i + 1} Top{k}: {100 * accuracy[i]:.2f}%"
                            )
                        self.time_keeper.print_log(
                            f"\tTop{k}: {100 * sum(accuracy) / len(accuracy):.2f}%"
                        )
                    else:
                        self.time_keeper.print_log(
                            f"\tTop{k}: {100 * self.data_loader[ln].dataset.top_k(score, k):.2f}%"
                        )

                print("Here are the just predicted labels: ", predictions)
                print("Here are the correct labels: ", label)
                print("Total samples seen so far: ", val_total)

                stats_val = f"Testing: Epoch [{epoch}/{self.arg.num_epoch}], Samples [{val_correct}/{val_total}], Loss: {loss.item()}, Testing Accuracy: {val_accuracy}"

                print(f"\n{stats_val}")

                for i in range(0, self.arg.model_args["num_class"]):
                    if class_total[i] != 0:
                        print(
                            f"Accuracy of {i + 1} : {int(class_correct[i])} / {int(class_total[i])} = {int(100 * class_correct[i] / class_total[i])} %"
                        )

                # Calculates the confusion matrix
                # conf_matrix = confusion_matrix(predictions.cpu(), label.cpu())
                # print("The confusion matrix is: ", conf_matrix)

                # Calculates and plots the confusion matrix
                # df_cm = pd.DataFrame(self.conf_matrix_val, index=[i for i in range(0, 60)],
                # columns=[i for i in range(0, 60)])
                # conf_fig = plt.figure(figsize=(13, 10))
                # plt.title("Confusion Matrix - Validation")
                # sn.heatmap(df_cm, annot=True)

        print(f"\n{stats_val}")

    def val(self, epoch, save_score=False, loader_name=["val"]):
        self.model.eval()
        self.time_keeper.print_log(f"Eval epoch: {epoch + 1}")
        val_correct = 0
        val_total = 0
        class_correct = [0.0] * self.arg.model_args["num_class"]
        class_total = [0.0] * self.arg.model_args["num_class"]

        with torch.no_grad():
            for ln in loader_name:
                loss_value = []
                score_frag = []
                for batch_idx, (data, label, name) in enumerate(self.data_loader[ln]):
                    data = data.float().to(DEVICE)
                    label = label.long().to(DEVICE)

                    output = self.model(data, label, name)
                    loss = self.loss(output, label)
                    score_frag.append(output.data.cpu().numpy())
                    loss_value.append(loss.data.item())

                    _, predictions = torch.max(output, 1)
                    val_total = val_total + label.size(0)
                    val_correct = (
                        val_correct + (predictions == label).double().sum().item()
                    )
                    val_accuracy = (val_correct / val_total) * 100
                    c = (label == predictions.squeeze()).float()
                    (c).float().mean()

                    # Calculating validation accuracy for each class
                    for l in range(0, label.size(0)):
                        class_label = label[l]
                        class_correct[class_label - 1] = (
                            class_correct[class_label - 1] + c[l]
                        )
                        class_total[class_label - 1] = class_total[class_label - 1] + 1

                    info = {"loss-Val": loss, "accuracy-val": val_accuracy}
                # conf_matrix_val += confusion_matrix(predictions.cpu(), label.cpu(), labels=np.arange(self.arg.model_args['num_class']))
                # np.save("./checkpoints/" + NAME_EXP + "/conf_val_" + str(epoch),
                #       conf_matrix_val)

                np.concatenate(score_frag)

                self.time_keeper.print_log(
                    f"\tMean {ln} loss of {len(self.data_loader[ln])} batches: {np.mean(loss_value)}."
                )

                print("Here are the just predicted labels: ", predictions)
                print("Here are the correct labels: ", label)
                print("Total samples seen so far: ", val_total)

                stats_val = f"Validation: Epoch [{epoch}/{self.arg.num_epoch}], Samples [{val_correct}/{val_total}], Loss: {loss.item()}, Validation Accuracy: {val_accuracy}"

                print("\n" + stats_val)

                for i in range(0, self.arg.model_args["num_class"]):
                    if class_total[i] != 0:
                        print(
                            f"Accuracy of {i + 1} : {int(class_correct[i])} / {int(class_total[i])} = {int(100 * class_correct[i] / class_total[i])} %"
                        )

                #
                step = (epoch + 1) * (
                    len(self.data_loader["train"]) / (arg.optimize_every)
                )

                for tag, value in info.items():
                    #     # logger.scalar_summary(tag, value, epoch + 1)
                    writer.add_scalar(tag, value, step)

        print(f"\n{stats_val}")
        return val_accuracy

    def start(self):

        if not self.arg.training:
            self.test(epoch=0, save_score=self.arg.save_score, loader_name=["test"])

        patience = 50
        patient_counter = 0

        pytorch_total_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        layer_params = sum([p.view(-1).shape[0] for p in self.model.parameters()])
        print("Params: ", pytorch_total_params)
        print("Layer params: ", layer_params)

        if self.arg.phase == "train":
            self.time_keeper.print_log(f"Parameters:\n{str(vars(self.arg))}\n")
            for epoch in range(self.arg.start_epoch, self.arg.num_epoch):

                save_model = ((epoch + 1) % self.arg.save_interval == 0) or (
                    epoch + 1 == self.arg.num_epoch
                )
                print(save_model)
                eval_model = ((epoch + 1) % self.arg.eval_interval == 0) or (
                    epoch + 1 == self.arg.num_epoch
                )

                self.train(epoch, save_model=save_model)
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
                else:
                    pass
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
            except:
                state = self.model.state_dict()
                diff = list(set(state.keys()).difference(set(weights.keys())))
                print("Can not find these weights:")
                for d in diff:
                    print("  " + d)
                state.update(weights)
                self.model.load_state_dict(state)
            self.test(epoch=0, save_score=self.arg.save_score, loader_name=["test"])
            self.time_keeper.print_log("Done.\n")

        elif self.arg.phase == "test":
            if self.arg.weights is None:
                raise ValueError("Please appoint --weights.")
            self.arg.time_keeper.print_log = False
            self.time_keeper.print_log(f"Model:   {self.arg.model}.")
            self.time_keeper.print_log(f"Weights: {self.arg.weights}.")

            self.test(epoch=0, save_score=self.arg.save_score, loader_name=["test"])
            self.time_keeper.print_log("Done.\n")


if __name__ == "__main__":

    parser = get_parser(NAME_EXP)
    p = parser.parse_args()
    if p.config is not None:
        with open(p.config, "r") as f:
            default_arg = yaml.load(f)
        key = vars(p).keys()
        for k in default_arg.keys():
            if k not in key:
                print(f"WRONG ARG: {k}")
                assert k in key
        parser.set_defaults(**default_arg)

    arg = parser.parse_args()
    processor = Processor(arg)
    processor.start()
