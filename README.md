# Selective Prediction
This repository aims at providing implementations for different selective prediction approaches, namely Softmax, Joint Model, Joint Softmax, Selective Net, VAE Reconstuction Loss and Latent Space Selection.
The evaulation is done examplary on the MNIST dataset. The selection algorithms can be applied on unmodified input data or with adversarial attacks applied, e.g. FGSM and PGD attacks as well as Gaussian noise.
Moreover, the selection approaches can be combined with adversarial training as defense against adversarial attacks. 
The selection approaches are described in the paper: [Defending Against Adversarial Attacks on Time- series with Selective Classification](https://ieeexplore.ieee.org/document/9808576/)

A script for training the required neural network is provided, e.g. run `python -m src.models --model_type vae --epochs 100` to train the VAE.

To conduct the evaluation with an FGSM attack applied run `python -m src.selective_prediction --attack FGSM --epsilon 0.1`.
