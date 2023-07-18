import torch
import torch.nn as nn


class Attack:
    def __init__(self, model, attack):
        self.attack = attack
        self.model = model
        self.device = next(model.parameters()).device

    def forward(self, *inputs):
        raise NotImplementedError

    def __call__(self, *inputs):
        self.model.eval()
        adv_data = self.forward(*inputs)
        return adv_data


class Clean(Attack):
    def __init__(self, model):
        super().__init__(model, 'Clean')

    def forward(self, data, labels):
        adv_data = data.clone().detach().to(self.device)
        return adv_data


class GN(Attack):
    def __init__(self, model, std=0.1):
        super().__init__(model, 'GN')
        self.std = std

    def forward(self, data, labels):
        data = data.clone().detach().to(self.device)
        adv_images = data + self.std*torch.randn_like(data)
        return adv_images.detach()


class FGSM(Attack):
    def __init__(self, model, eps=0.1, is_image=False, loss_type='cross_entropy'):
        super().__init__(model, 'FGSM')
        self.eps = eps
        self.is_image = is_image
        self.loss_type = loss_type
        if self.loss_type == 'mse':
            self.loss = nn.MSELoss()
        elif self.loss_type == 'cross_entropy':
            self.loss = nn.CrossEntropyLoss()
        elif self.loss_type == 'bce_with_logits':
            self.loss = nn.BCEWithLogitsLoss()
        else:
            raise NotImplementedError

    def forward(self, data, labels):
        if isinstance(data, tuple):
            adv_images = self.forward_siamese(data, labels)
        else:
            adv_images = self.forward_standard(data, labels)
        return adv_images

    def forward_standard(self, data, labels):
        data = data.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        data.requires_grad = True
        self.model.to(self.device)
        outputs = self.model(data)
        loss = self.loss(outputs, labels)
        grad = torch.autograd.grad(loss, data,
                                   retain_graph=False, create_graph=False)[0]
        adv_data = data + self.eps * grad.sign()
        if self.is_image:
            adv_data = torch.clamp(adv_data, min=0, max=1)
        adv_data = adv_data.detach()
        return adv_data

    def forward_siamese(self, data, labels):
        data0 = data[0].clone().detach().to(self.device)
        data1 = data[1].clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        data0.requires_grad = True
        data1.requires_grad = True
        self.model.to(self.device)
        outputs = self.model(data0, data1)
        loss = self.loss(outputs, labels)
        grad = torch.autograd.grad(loss, data0,
                                   retain_graph=False, create_graph=False)[0]
        adv_data = data0 + self.eps * grad.sign()
        adv_data = adv_data.detach()
        return adv_data


class PGD(Attack):
    def __init__(self, model, eps=0.3, alpha=0.1, steps=40, random_start=True, is_image=False,
                 loss_type='cross_entopy'):
        super().__init__(model, PGD)
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.random_start = random_start
        self.is_image = is_image
        self.loss_type = loss_type
        if self.loss_type == 'mse':
            self.loss = nn.MSELoss()
        elif self.loss_type == 'cross_entropy':
            self.loss = nn.CrossEntropyLoss()
        elif self.loss_type == 'bce_with_logits':
            self.loss = nn.BCEWithLogitsLoss()
        else:
            raise NotImplementedError

    def forward(self, data, labels):
        if isinstance(data, tuple):
            adv_data = self.forward_siamese(data, labels)
        else:
            adv_data = self.forward_standard(data, labels)
        return adv_data

    def forward_standard(self, data, labels):
        data = data.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        adv_data = data.clone().detach()

        if self.random_start:
            adv_data = adv_data + torch.empty_like(adv_data).uniform_(-self.eps, self.eps)
            if self.is_image:
                adv_data = torch.clamp(adv_data, min=0, max=1).detach()

        for _ in range(self.steps):
            adv_data.requires_grad = True
            self.model.to(self.device)
            outputs = self.model(adv_data)
            loss = self.loss(outputs, labels)

            # Update adversarial images
            grad = torch.autograd.grad(loss, adv_data,
                                       retain_graph=False, create_graph=False)[0]

            adv_data = adv_data.detach() + self.alpha * grad.sign()
            delta = torch.clamp(adv_data - data, min=-self.eps, max=self.eps)
            if self.is_image:
                adv_data = torch.clamp(data + delta, min=0, max=1).detach()
            else:
                adv_data = (data + delta).detach()
        return adv_data

    def forward_siamese(self, data, labels):
        data0 = data[0].clone().detach().to(self.device)
        data1 = data[1].clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        adv_image0 = data0.clone().detach()
        if self.random_start:
            data0 = data0 + torch.empty_like(data0).uniform_(-self.eps, self.eps)
            data0 = data0 + torch.empty_like(data0).uniform_(-self.eps, self.eps)
            if self.is_image:
                data0 = torch.clamp(data0, min=0, max=1).detach()

        for _ in range(self.steps):
            data0.requires_grad = True
            data1.requires_grad = True
            self.model.to(self.device)
            outputs = self.model(data0, data1)
            loss = self.loss(outputs, labels)
            grad = torch.autograd.grad(loss, data0,
                                       retain_graph=False, create_graph=False)[0]

            adv_image0 = adv_image0.detach() + self.alpha * grad.sign()
            delta = torch.clamp(adv_image0 - data0, min=-self.eps, max=self.eps)
            if self.is_image:
                adv_image0 = torch.clamp(data0 + delta, min=0, max=1).detach()
            else:
                adv_image0 = (data0 + delta).detach()
        return adv_image0
