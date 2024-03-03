import torch

def cosine_similarity(x1, x2):
    x1_norm = x1.norm(p=2, dim=1, keepdim=True)
    x2_norm = x2.norm(p=2, dim=1, keepdim=True)
    sim_matrix = torch.mm(x1, x2.t()) / (x1_norm * x2_norm.t() + 1e-8)
    return sim_matrix


class EarlyStopping:
    def __init__(self, id, datasets, methods, gnn_type, patience=10, min_delta=0):
        """
        :param patience: Can tolerate no improvement within how many epochs
        :param min_delta: The minimum amount of change for improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

        dataset_list = [str(item) for item in datasets]
        methods_list = [str(item) for item in methods]
        self.id = id
        self.datasets = '+'.join(dataset_list)
        self.methods = '+'.join(methods_list)
        self.gnn_type = gnn_type

    def __call__(self, model, val_loss):
        if self.best_score is None:
            self.best_score = val_loss
        elif val_loss > self.best_score - self.min_delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                print('Early stopping')
                self.early_stop = True
        else:
            self.best_score = val_loss
            self.counter = 0
            torch.save(model.GNN.state_dict(), "pretrained_gnn/{}_{}_{}_{}.pth".format(self.datasets, self.methods, self.gnn_type, self.id))
