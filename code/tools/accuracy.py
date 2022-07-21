import torch


def single_label_accuracy(
    outputs: torch.Tensor, labels: torch.Tensor, total: int, total_correct: list
):
    predictions = [torch.max(output, 1)[1] for output in outputs]
    total += labels[0].size(0)
    for ind, predicts in enumerate(predictions):
        total_correct[ind] += (predicts == labels[ind]).sum().item()
    acc = 100 * sum(total_correct) / total
    return acc, predictions, total, total_correct


def multi_label_accuracy(
    outputs: torch.Tensor, labels: torch.Tensor, total: int, total_correct: list
):
    predictions = [torch.round(output) for output in outputs]
    total += labels[0].size(0)
    for ind, predicts in enumerate(predictions):
        total_correct[ind] += (predicts == labels[ind]).sum().item()
    acc = 100 * sum(total_correct) / total
    return acc, predictions, total, total_correct
