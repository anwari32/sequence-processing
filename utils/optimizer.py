from torch.optim import Adamax, AdamW

def init_optimizer(optim_name, model_parameters, learning_rate, epsilon, beta1, beta2, weight_decay):
    if optim_name == "adamw":
        return AdamW(model_parameters, lr=learning_rate, eps=epsilon, betas=(beta1, beta2), weight_decay=weight_decay)
    elif optim_name == "adamax":
        return Adamax(model_parameters,lr=learning_rate, eps=epsilon, betas=(beta1, beta2), weight_decay=weight_decay)
    else:
        return None
