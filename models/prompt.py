import torch
import torch.nn as nn

class PromptComponent(nn.Module):

    def __init__(self, prompt_num, prompt_dim, layers):
        super(PromptComponent, self).__init__()

        # Initialize
        self.task_cnt = 0
        self.prompt_num = prompt_num
        self.prompt_dim = prompt_dim
        self.layers = layers # -1 is for shallow prompt, other values are for deep prompt (prompt layers = GNN layers)
        self.prompt_pool = []

    def gram_schmidt(prompt):
        basis = []
        for p in prompt:
            w = p - sum(torch.dot(p, b) * b for b in basis)
            if torch.norm(w) > 1e-10:
                basis.append(w / torch.norm(w))
        return torch.stack(basis)

    def new_task_init(self):
        self.task_cnt += 1
        if self.layers != -1:
            new_prompt = nn.ParameterList()
            for _ in self.layers:
                prompt_tmp = nn.Parameter(torch.FloatTensor(self.prompt_num, self.prompt_dim), requires_grad=True)
                new_prompt.append(self.gram_schmidt(prompt_tmp))
            self.prompt_pool.append(new_prompt)
        else:
            new_prompt = nn.ParameterList()
            prompt_tmp = nn.Parameter(torch.FloatTensor(self.prompt_num, self.prompt_dim), requires_grad=True)
            new_prompt.append(self.gram_schmidt(prompt_tmp))
            self.prompt_pool.append(new_prompt)
        
        return self.task_cnt - 1 # Return task id