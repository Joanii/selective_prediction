import pytorch_lightning as pl
from pytorch_lightning import loggers
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import src.attacks as attacks


class MNISTModule(pl.LightningModule):
    """
    Base class for neural network models for mnist dataset.
    """
    def __init__(self):
        super().__init__()

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        """
        Pytorch lightning training step. Includes model step and logging.
        :param batch: batch data (features and targets)
        :param batch_idx: batch index
        :return: model loss
        """
        loss, accuracy = self._step(batch)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        if accuracy is not None:
            self.log("train/accuracy", accuracy, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        """
        Pytorch lightning validation step. Includes model step and logging.
        :param batch: batch data (features and targets)
        :param batch_idx: batch index
        :return: model loss
        """
        loss, accuracy = self._step(batch)
        self.log("validation/loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        if accuracy is not None:
            self.log("validation/accuracy", accuracy, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        """
        Pytorch lightning test step. Includes model step and logging.
        :param batch: batch data (features and targets)
        :param batch_idx: batch index
        :return: model loss
        """
        loss, accuracy = self._step(batch)
        self.log("test/loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        if accuracy is not None:
            self.log("test/accuracy", accuracy, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def _step(self, batch):
        raise NotImplementedError

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    @staticmethod
    def get_accuracy(predicted, target):
        """
        Compute prediction accuracy of classification model.
        :param predicted: classifier predictions [batch_size, n_classes]
        :param target: target of prediction [batch_size]
        :return:
        """
        accuracy = (torch.max(predicted, 1)[1] == target).sum().item() / float(predicted.shape[0])
        return accuracy


class Classifier(MNISTModule):
    """
    Classification model for MNIST dataset
    """
    def __init__(self, n_classes: int = 10):
        """
        Neural network classifier for MNIST dataset with convolutional and linear layers.
        :param n_classes: number of classes, normally 10 for MNIST
        """
        super().__init__()
        self.save_hyperparameters()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5)
        self.conv2 = nn.Conv2d(16, 22, kernel_size=3)
        self.conv3 = nn.Conv2d(22, 22, kernel_size=3)
        self.fc1 = nn.Linear(88, 50)
        self.fc2 = nn.Linear(50, n_classes)
        self.loss_function = nn.CrossEntropyLoss()

    def forward(self, x: torch.tensor):
        """
        Forward pass through model.
        :param x: input data
        :return: prediction
        """
        batch_size = x.shape[0]
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = F.max_pool2d(F.relu(self.conv3(x)), 2)
        x = x.view(batch_size, -1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x

    def _step(self, batch):
        features, targets = batch
        predicted = self.forward(features)
        loss = self.loss_function(predicted, targets)
        accuracy = self.get_accuracy(predicted, targets)
        return loss, accuracy


class AdvClassifier(Classifier):
    """
    Adversarially trained classifier for MNIST dataset.
    """
    def __init__(self, n_classes: int = 10, attack_name='PGD', eps=0.1):
        """
        Neural network classifier for MNIST dataset trained on adversarially perturbed data.
        :param n_classes: number of classes (usually 10 for MNIST)
        :param attack_name: name of the adversarial attack, e.g. FGSM, PGD
        :param eps: magnitude of the attack (float (0, 1))
        """
        super().__init__(n_classes)
        self.save_hyperparameters()
        self.attack = self._initialize_attack(attack_name, eps)
        self.current_device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    def _initialize_attack(self, attack_name, eps):
        if attack_name == 'PGD':
            atk = attacks.PGD(self, eps=eps)
        elif attack_name == 'FGSM':
            atk = attacks.FGSM(self, eps=eps)
        else:
            raise NotImplementedError
        return atk

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        """
        Modified training step for adversarial training. Includes model step and logging.
        :param batch: input batch (features and targets)
        :param batch_idx: batch index
        :return: model loss
        """
        features, targets = batch
        targets = targets.to(self.current_device)
        features = features.to(self.current_device)

        adv_features = self.attack(features, targets)
        self.conv1 = self.conv1.to(self.current_device)
        self.conv2 = self.conv2.to(self.current_device)
        self.conv3 = self.conv3.to(self.current_device)
        self.fc1 = self.fc1.to(self.current_device)
        self.fc2 = self.fc2.to(self.current_device)
        loss, accuracy = self._step((adv_features.to(self.current_device), targets.to(self.current_device)))
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train/accuracy", accuracy, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss


class SelectiveNet(MNISTModule):
    """
    Implementation of Selective Net by Geifman & El-Yaniv https://arxiv.org/abs/1901.09192
    """
    def __init__(self, coverage: float, alpha: float = 0.5):
        """
        Selective Net for MNIST classification with integrated reject option.
        :param coverage: target coverage of the model [0, 1)
        :param alpha: weighting factor of selective and cross entropy loss
        """
        super().__init__()
        self.save_hyperparameters()
        self.coverage = coverage
        self.alpha = alpha
        self.num_classes = 10
        self.image_size = 32
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5)
        self.conv2 = nn.Conv2d(16, 22, kernel_size=3)
        self.conv3 = nn.Conv2d(22, 22, kernel_size=3)
        self.classifier, self.selector, self.aux_classifier = self._get_layers()
        self.loss_function = nn.CrossEntropyLoss()
        self.selective_loss = SelectiveLoss(self.loss_function, coverage=self.coverage)

    def forward(self, x, all_outputs=False):
        batch_size = x.shape[0]
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = F.max_pool2d(F.relu(self.conv3(x)), 2)
        x = F.dropout(x, training=self.training)
        x = x.view(batch_size, -1)
        prediction_out = self.classifier(x)
        selection_out = self.selector(x)
        auxiliary_out = self.aux_classifier(x)
        if all_outputs:
            return prediction_out, selection_out, auxiliary_out
        else:
            return prediction_out

    def _step(self, batch: torch.Tensor):
        features, target = batch
        out_class, out_select, out_aux = self.forward(features, all_outputs=True)
        selective_loss = self.selective_loss(out_class, out_select, target)
        ce_loss = self.loss_function(out_aux, target)
        loss = self.alpha * selective_loss + (1.0 - self.alpha) * ce_loss
        accuracy = self.get_accuracy(out_class, target)
        return loss, accuracy

    def _get_layers(self):
        dim_features = 88
        classifier = torch.nn.Sequential(nn.Linear(dim_features, 50),
                                         nn.ReLU(True),
                                         nn.Linear(50, self.num_classes),
                                         )

        selector = torch.nn.Sequential(nn.Linear(dim_features,
                                                 50),
                                       nn.ReLU(True),
                                       nn.Linear(50, 1),
                                       nn.Sigmoid())

        aux_classifier = torch.nn.Sequential(nn.Linear(dim_features, 50),
                                             nn.ReLU(True),
                                             nn.Linear(50, self.num_classes),
                                             )
        return classifier, selector, aux_classifier


class AdvSelectiveNet(SelectiveNet):
    """
    Adversarially trained Selective Net for MNIST classification.
    """
    def __init__(self, coverage: float, alpha: float = 0.5, attack_name='PGD', eps=0.1):
        """
        Selective Net for MNIST dataset trained on adversarially perturbed data.
        :param coverage: target coverage (0, 1]
        :param alpha: weighting factor of selective and cross entropy loss
        :param attack_name: name of the adversarial attack, e.g. FGSM, PGD
        :param eps: magnitude of the attack (float (0, 1))
        """
        super().__init__(coverage, alpha)
        self.save_hyperparameters()
        self.attack = self._initialize_attack(attack_name, eps)
        self.current_device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    def _initialize_attack(self, attack_name, eps):
        if attack_name == 'PGD':
            atk = attacks.PGD(self, eps=eps)
        elif attack_name == 'FGSM':
            atk = attacks.FGSM(self, eps=eps)
        else:
            raise NotImplementedError
        return atk

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        features, targets = batch
        targets = targets.to(self.current_device)
        features = features.to(self.current_device)

        adv_features = self.attack(features, targets)
        self.conv1 = self.conv1.to(self.current_device)
        self.conv2 = self.conv2.to(self.current_device)
        self.conv3 = self.conv3.to(self.current_device)
        self.classifier = self.classifier.to(self.current_device)
        self.selector = self.selector.to(self.current_device)
        self.aux_classifier = self.aux_classifier.to(self.current_device)
        loss, accuracy = self._step((adv_features.to(self.current_device), targets.to(self.current_device)))
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train/accuracy", accuracy, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss


class VAE(MNISTModule):
    """
    VAE for reconstruction of MNIST images. Can be used within several selective prediction approaches.
    """
    def __init__(self, bottleneck_dim: int = 2, beta: float = 1.):
        """
        Linear beta VAE for MNIST.
        :param bottleneck_dim: dimension of the bottleneck
        :param beta: weighting factor for KLD and MSE loss.
        """
        super().__init__()
        self.image_size = 32
        self.beta = beta
        self.save_hyperparameters()
        self.bottleneck_dim = bottleneck_dim
        self.encoder = self._get_encoder_layers()
        self.decoder = self._get_decoder_layers()
        self.mu_layer = nn.Linear(256, self.bottleneck_dim)
        self.sigma_layer = nn.Linear(256, self.bottleneck_dim)
        self.loss_function_sum = nn.MSELoss(reduction='sum')
        self.loss_function = nn.MSELoss()

    def _step(self, batch):
        features, _ = batch
        recon, mu, logvar = self.forward(features)
        loss = self._get_loss(recon, features, mu, logvar)
        return loss, None

    def _get_encoder_layers(self):
        layers = [nn.Linear(self.image_size**2, 512),
                  nn.ReLU(),
                  nn.Linear(512, 256),
                  nn.ReLU()]
        layers = nn.Sequential(*layers)
        return layers

    def _get_decoder_layers(self):
        layers = [nn.Linear(self.bottleneck_dim, 256),
                  nn.ReLU(),
                  nn.Linear(256, 512),
                  nn.ReLU(),
                  nn.Linear(512, self.image_size**2)]
        layers = nn.Sequential(*layers)
        return layers

    @staticmethod
    def sampling(mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        b = self.encoder(x.view(-1, self.image_size**2))
        mu, log_var = self.mu_layer(b), self.sigma_layer(b)
        z = self.sampling(mu, log_var)
        return self.decoder(z).view(x.shape), mu, log_var

    def _get_loss(self, recon_x, x, mu, log_var):
        mse = self.loss_function_sum(x, recon_x)
        kld = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        return mse + self.beta * kld


class SelectiveLoss(nn.Module):
    """
    Loss class for Selective Net. Loss considers coverage and prediction result.
    """
    def __init__(self, loss_func, coverage: float, lm: float = 32.0):
        """
        loss_func: base loss function. E.g. nn.CrossEntropyLoss() for classification
        coverage: target coverage
        lm: Lagrange multiplier for coverage constraint
        """
        super().__init__()
        self.loss_func = loss_func
        self.coverage = coverage
        self.lm = lm
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    def forward(self, prediction_out, selection_out, target):
        """
        Compute loss for given prediction and selection output of Selective Net.
        prediction_out: prediction output of Selective Net
        selection_out: selection output of Selective Net
        target: target of prediction
        """
        empirical_coverage = selection_out.mean()

        empirical_risk = (self.loss_func(prediction_out, target)*selection_out.view(-1)).mean()
        empirical_risk = empirical_risk / empirical_coverage

        coverage = torch.tensor([self.coverage], dtype=torch.float32, requires_grad=True).to(self.device)
        penalty = torch.max(coverage-empirical_coverage, torch.tensor([0.0], dtype=torch.float32,
                                                                      requires_grad=True).to(self.device))**2
        penalty *= self.lm

        selective_loss = empirical_risk + penalty
        return selective_loss


if __name__ == '__main__':
    from pytorch_lightning.callbacks import ModelCheckpoint
    import argparse
    from src.dataset import MnistDataModule

    parser = argparse.ArgumentParser(description='Train neural network models')
    parser.add_argument('--model_type', required=True, help='vae, clf, adv_clf, sel or adv_sel')
    parser.add_argument('--epochs', required=True, type=int, help='number of training epochs')
    args = parser.parse_args()
    if args.model_type == 'vae':
        model = VAE(bottleneck_dim=3, beta=0.5)
        args.model_type = 'vae_beta'
    elif args.model_type == 'clf':
        model = Classifier()
    elif args.model_type == 'adv_clf':
        model = AdvClassifier()
    elif args.model_type == 'sel':
        model = SelectiveNet(coverage=0.95, alpha=0.7)
    elif args.model_type == 'adv_sel':
        model = AdvSelectiveNet(coverage=0.95, alpha=0.7)
    else:
        raise NotImplementedError
    logger = loggers.TensorBoardLogger(os.path.join('models', args.model_type), name='lightning_logs')
    dataset = MnistDataModule()
    checkpoint_callback = ModelCheckpoint(monitor="validation/loss")
    trainer = pl.Trainer(max_epochs=args.epochs, callbacks=[checkpoint_callback], logger=logger)
    trainer.fit(model, dataset.train_dataloader(), dataset.validation_dataloader())
