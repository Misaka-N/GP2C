import torch
import torch.nn as nn
from utils.tools import cosine_similarity

class PromptComponent(nn.Module):

    def __init__(self, prompt_num, prompt_dim, input_dim, layers):
        super(PromptComponent, self).__init__()

        # Initialize
        self.task_cnt = 0
        self.prompt_num = prompt_num
        self.prompt_dim = prompt_dim
        self.input_dim = input_dim

        self.attention = nn.ParameterList()
        self.keys = nn.ParameterList()
        self.prompt = nn.ParameterList()


    def gram_schmidt(self, prompt):
        n, feat_dim = prompt.shape
        basis = torch.zeros_like(prompt)
        for i in range(n):
            w = prompt[i].clone()
            for j in range(i):
                projection = torch.dot(w, basis[j]) * basis[j]
                w = w - projection
            if torch.norm(w) > 1e-10:
                basis[i] = w / torch.norm(w)
            else:
                basis[i] = torch.zeros_like(w)
        return basis
    

    def new_task_init(self):
        self.task_cnt += 1
        self.attention.append(nn.Parameter(torch.FloatTensor(1), requires_grad=True)) # Attention weight

        new_key = nn.Parameter(torch.randn(self.input_dim), requires_grad=True) # Prompt key
        norm = new_key.norm(p=2)
        self.keys.append(new_key / (norm + 1e-8))

        prompt_tmp = nn.Parameter(torch.randn(self.prompt_num, self.prompt_dim), requires_grad=True)
        new_prompt = self.gram_schmidt(prompt_tmp)
        self.prompt.append(new_prompt)


    def freeze_prompt_params(self, train): 
        if train:
            last_id = self.task_cnt - 1 
        else:
            last_id = self.task_cnt
        id = 0
        for attention, key, prompt in zip(self.attention, self.keys, self.prompt):
            if id < last_id:
                attention.requires_grad = False
                key.requires_grad = False
                for param in prompt:
                    if self.layers == -1:
                        param.requires_grad = False
                    else:
                        for p in param:
                            p.requires_grad = False
            id += 1
    

    def forward(self, query, train=False):
        self.freeze_prompt_params(train)

        # Calculating for similarity weight
        weight_prompt = []
        for attention, key, prompt in zip(self.attention, self.keys, self.prompt):
            sim = cosine_similarity(query * attention.unsqueeze(0), key.unsqueeze(0))
            updated_prompt = prompt * sim
            weight_prompt.append(updated_prompt)

        # Get summed prompt
        summed_prompt = nn.Parameter(torch.zeros_like(self.prompt[0]), requires_grad=True)

        for prompt in weight_prompt:
            summed_prompt = torch.add(summed_prompt, prompt)

        return summed_prompt