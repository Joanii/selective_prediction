import os
import torch.nn.functional as F
import numpy as np
from src.attacks import *
from sklearn import manifold


def get_ckpt_path_from_folder(folder):
    ckpt_folder = os.path.join(folder, 'checkpoints')
    if not os.path.exists(ckpt_folder):
        raise Warning(f'Folder {folder} does not exist!')
    if len(os.listdir(ckpt_folder)) == 0:
        raise Warning(f'No checkpoint in folder {folder}')
    ckpt_file = os.listdir(ckpt_folder)[0]
    ckpt_path = os.path.join(ckpt_folder, ckpt_file)
    return ckpt_path


def get_latest_checkpoint(folder, latest=-1):
    versions = os.listdir(folder)
    numbers = []
    for version in versions:
        numbers.append(int(version.split('_')[1]))
    numbers.sort()
    latest_version = numbers[latest]
    path = os.path.join(folder, f'version_{latest_version}')
    ckpt_path = get_ckpt_path_from_folder(path)
    return ckpt_path


def reconstruction_loss(prediction, targets):
    prediction = prediction.squeeze()
    targets = targets.squeeze()
    if len(prediction.shape) == 2:
        loss = torch.mean((targets - prediction) ** 2, dim=1)
    else:
        loss = torch.mean(torch.mean((targets - prediction) ** 2, dim=1), dim=1)
    return loss


def get_softmax_response(predict):
    softmax = get_softmax(predict)
    softmax_response, _ = softmax.max(1)
    return softmax_response


def get_softmax(predict):
    if predict[0].sum().item() != 1.:  # Applying softmax (in case of cross entropy loss)
        softmax = F.softmax(predict, dim=1)
    else:
        softmax = predict
        print('Softmax is not applied, check if it is included in the model')
    return softmax


def initialize_attack(attack_name, model, epsilon, loss_type='cross_entropy'):
    """
    :param attack_name: name of the attack, e.g. PGD, FGSM, clean, GN
    :param model: target model
    :param epsilon: magnitude of attack
    :param loss_type: type of loss, depending on task, e.g. cross-entropy for classification
    :return: instance of class Attack
    """
    if attack_name == 'FGSM':
        atk = FGSM(model, eps=epsilon, loss_type=loss_type)
    elif attack_name == 'PGD':
        atk = PGD(model, eps=epsilon, alpha=0.1, steps=4, loss_type=loss_type)
    elif attack_name == 'GN':
        atk = GN(model, std=epsilon)
    elif attack_name == 'clean':
        atk = Clean(model)
    else:
        raise Warning('attack_name must be one of FGSM, PGD, GN, clean')
    return atk


def iterate_thresholds(loss, predict, targets, thresholds=None):
    thresholds = thresholds if thresholds is not None else _get_evenly_distr_thresh(loss.copy())
    risk = np.zeros(len(thresholds))
    samples = np.zeros(len(thresholds))
    for t_idx, threshold in enumerate(thresholds):
        cut_off = loss >= threshold
        if sum(cut_off) == len(loss):
            risk[t_idx] = np.nan
            samples[t_idx] = np.nan
        else:
            samples[t_idx] = sum(~cut_off)
            risk[t_idx] = calculate_risk(cut_off, samples[t_idx], predict, targets)
    coverage = samples / loss.shape[0]
    return risk, coverage


def _get_evenly_distr_thresh(selection_values, n=30):
    if isinstance(selection_values, torch.Tensor):
        selection_values, _ = selection_values.sort()
    else:
        selection_values.sort()
    n_th = int(selection_values.shape[0] / n)
    thresholds = selection_values[::n_th]
    return thresholds


def calculate_risk(cut_off, samples, predict, targets):
    _, predicted = predict.squeeze()[~cut_off].max(1)
    correct = predicted.eq(targets[~cut_off]).sum().item()
    accuracy = correct / samples
    risk = 1 - accuracy
    return risk


def get_data_from_loader(dataloader: torch.utils.data.DataLoader):
    """
    Extract all data and targets from a dataloader as single tensor.
    The batch size of the dataloader is assumed to be the size of the data so that only a single batch is formed.
    :param dataloader: torch dataloader
    :return:
    """
    for data, targets in dataloader:
        return data, targets


def tsne_reduce_dimensions(latent_code):
    tsne = manifold.TSNE(n_components=2, random_state=42)
    z_tsne = tsne.fit_transform(latent_code.detach().numpy())
    return z_tsne
