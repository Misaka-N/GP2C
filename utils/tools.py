import torch
import random
import numpy as np

def set_random(seed, cuda):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed) 
    if cuda:
        torch.cuda.manual_seed(0)
        torch.cuda.manual_seed_all(0)


def cosine_similarity(x1, x2):
    x1_norm = torch.norm(x1, dim=1, keepdim=True)
    x2_norm = torch.norm(x2, dim=1, keepdim=True)
    sim_matrix = torch.mm(x1, x2.t()) / (x1_norm * x2_norm.t() + 1e-8)

    return sim_matrix


def label_smoothing(labels, epsilon, num_classes):
    soft_labels = torch.full((labels.size(0), num_classes), fill_value=epsilon / (num_classes - 1), device=labels.device)
    soft_labels.scatter_(1, labels.unsqueeze(1), (1 - epsilon) + epsilon / (num_classes - 1))
    
    return soft_labels


class EarlyStopping:
    def __init__(self, path1, patience=10, min_delta=0, path2=None):
        """
        :param patience: Can tolerate no improvement within how many epochs
        :param min_delta: The minimum amount of change for improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

        self.path1 = path1
        self.path2 = path2

    def __call__(self, model, val_loss):
        if self.best_score is None:
            self.best_score = val_loss
        elif val_loss >= self.best_score - self.min_delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                print('Early stopping')
                self.early_stop = True
        else:
            self.best_score = val_loss
            self.counter = 0
            if hasattr(model, 'GNN'):
                torch.save(model.GNN.state_dict(), self.path1)
                # torch.save(model.GNN, self.path1)
            else:
                torch.save(model[0], self.path1) 
                if self.path2 is not None:
                    torch.save(model[1], self.path2)
