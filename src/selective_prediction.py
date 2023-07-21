import os
import torch

from src.utils import *
from src.plotting import plot_risk_over_coverage


def run_softmax(predict, target):
    softmax_response = get_softmax_response(predict)
    softmax_response = softmax_response.detach().numpy().squeeze()
    risk, coverage = iterate_thresholds(1 - softmax_response, predict, target)
    return risk, coverage


def run_joint_model(predict, second_predict, target):
    _, predicted = predict.max(1)
    _, second_predicted = second_predict.max(1)
    joint_mask = predicted.squeeze() == second_predicted.squeeze()
    cut_off = ~joint_mask
    samples = sum(joint_mask).item()
    risk = calculate_risk(cut_off, samples, predict, target)
    coverage = samples / predicted.shape[0]
    return risk, coverage


def run_joint_softmax(output, second_output, target):
    softmax = get_softmax(output)
    softmax_response, predicted = softmax.max(1)
    second_softmax = get_softmax(second_output)
    second_softmax_response, second_predicted = second_softmax.max(1)
    diff = torch.abs(second_softmax_response - softmax_response)
    diff[predicted != second_predicted] = 1
    risk, coverage = iterate_thresholds(diff.detach().numpy(), output, target)
    return risk, coverage


def run_selective_net(selective_model, sel_data, target, attack_name, epsilon):
    sel_atk = initialize_attack(attack_name, selective_model, epsilon)
    adv_samples_sel = sel_atk(sel_data, target)
    p_pred, s_pred, _ = selective_model(adv_samples_sel, all_outputs=True)
    sel_risk, sel_coverage = iterate_thresholds(1-s_pred.squeeze().detach().numpy(), p_pred, target)
    return sel_risk, sel_coverage


def run_latent_selection(output, targets, mu, recon, samples, vae, ref_dataloader):
    alpha = 0.2
    ref_data, ref_targets = get_data_from_loader(ref_dataloader)
    ref_recon, ref_mu, ref_sigma = vae(ref_data)
    distances = torch.empty(output.shape[0])
    recon_loss = torch.empty(output.shape[0])
    _, predicted_classes = output.squeeze().max(1)
    mu_concat = torch.cat((mu, ref_mu))
    mu_concat_tsne = torch.Tensor(tsne_reduce_dimensions(mu_concat))
    mu_tsne, ref_mu_tsne = mu_concat_tsne[:mu.shape[0], :], mu_concat_tsne[mu.shape[0]:, :]
    for idx, (mu_i, sample, predicted_class) in enumerate(zip(mu_tsne, samples, predicted_classes)):
        # distance to the next datapoint from predicted class is considered
        ref_mu_tsne_class = ref_mu_tsne[ref_targets == predicted_class]
        distances[idx] = torch.linalg.vector_norm(mu_i - ref_mu_tsne_class, dim=1).min()
        recon_loss[idx] = (torch.linalg.vector_norm(recon[idx] - sample)) ** 2 / torch.linalg.vector_norm(sample)
    loss = (1-alpha) * distances + alpha * recon_loss
    risks, coverages = iterate_thresholds(loss.detach().numpy(), output, targets)
    return risks, coverages


def run_selective_prediction(clf, vae, sel, selection_algorithms, datamodule, attack_name, epsilon,
                             folder_path='results'):
    atk = initialize_attack(attack_name, clf, epsilon)
    coverages = {}
    risks = {}
    for data, target in datamodule.test_dataloader():
        sel_data = data.clone()
        adv_samples = atk(data, target)
        output = clf(adv_samples)
        recon, mu, sigma = vae(adv_samples)
        second_prediction = clf(recon)
        recon_loss = reconstruction_loss(recon, adv_samples)
        # VAE Recon Loss
        if selection_algorithms['VAE']:
            risks['VAE'], coverages['VAE'] = iterate_thresholds(recon_loss.detach().numpy(), output, target)
        # Joint Model
        if selection_algorithms['Joint']:
            risks['Joint'], coverages['Joint'] = run_joint_model(output, second_prediction, target)
        # Softmax
        if selection_algorithms['Softmax']:
            risks['Softmax'], coverages['Softmax'] = run_softmax(output, target)
        ref_dataloader = datamodule.validation_dataloader()
        # latent
        if ref_dataloader and selection_algorithms['Latent']:
            risks['Latent'], coverages['Latent'] = run_latent_selection(output, target, mu, recon, adv_samples, vae,
                                                                        ref_dataloader)
        # Joint Softmax selection
        if selection_algorithms['Joint-S']:
            risks['Joint-S'], coverages['Joint-S'] = run_joint_softmax(output, second_prediction, target)
        # Selective Net
        if sel and selection_algorithms['Sel']:
            risks['Sel'], coverages['Sel'] = run_selective_net(sel, sel_data, target, attack_name, epsilon)
        plot_risk_over_coverage(risks, coverages, attack_name, epsilon, folder_path)


if __name__ == '__main__':
    from src.models import Classifier, SelectiveNet, VAE
    from src.dataset import MnistDataModule
    from src.utils import get_latest_checkpoint
    import argparse

    parser = argparse.ArgumentParser(description='Train neural network models')
    parser.add_argument('--attack', required=True, help='Adversarial attack, e.g. pgd, fgsm, vanilla')
    parser.add_argument('--epsilon', required=True, type=float, help='magnitude of attack')
    args = parser.parse_args()
    vae = VAE.load_from_checkpoint(get_latest_checkpoint(os.path.join('models', 'vae', 'lightning_logs'))).eval()
    clf = Classifier.load_from_checkpoint(get_latest_checkpoint(os.path.join('models', 'clf', 'lightning_logs'))).eval()
    sel = SelectiveNet.load_from_checkpoint(get_latest_checkpoint(os.path.join('models', 'sel', 'lightning_logs'))).eval()
    selection_algorithms = {'Softmax': True, 'VAE': True, 'Sel': True, 'Joint': True, 'Joint-S': True, 'Latent': True}
    run_selective_prediction(clf, vae, sel, selection_algorithms, MnistDataModule(), args.attack, args.epsilon)
