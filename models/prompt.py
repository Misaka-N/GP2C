import torch
import torch.nn as nn
from utils.tools import cosine_similarity

class PromptComponent(nn.Module):

    def __init__(self, prompt_num, prompt_dim, layers):
        super(PromptComponent, self).__init__()

        # Initialize
        self.task_cnt = 0
        self.prompt_num = prompt_num
        self.prompt_dim = prompt_dim
        self.layers = layers # -1 is for shallow prompt, other values are for deep prompt (prompt layers = GNN layers)
        self.prompt_pool = []
        # self.if_new = False # whether has a new task need to learn


    def gram_schmidt(prompt):
        basis = []
        for p in prompt:
            w = p - sum(torch.dot(p, b) * b for b in basis)
            if torch.norm(w) > 1e-10:
                basis.append(w / torch.norm(w))
        return torch.stack(basis)


    def new_task_init(self):
        self.task_cnt += 1
        prompt_component = {}
        prompt_component['attention'] = nn.Parameter(torch.FloatTensor(1), requires_grad=True) # Attention weight
        prompt_component['key'] = nn.Parameter(torch.FloatTensor(self.prompt_dim), requires_grad=True) # Prompt key
        if self.layers != -1:
            new_prompt = nn.ParameterList()
            for _ in self.layers:
                prompt_tmp = nn.Parameter(torch.FloatTensor(self.prompt_num, self.prompt_dim), requires_grad=True)
                new_prompt.append(self.gram_schmidt(prompt_tmp))
            prompt_component['prompt'] = new_prompt
            self.prompt_pool.append(prompt_component)
        else:
            prompt_tmp = nn.Parameter(torch.FloatTensor(self.prompt_num, self.prompt_dim), requires_grad=True)
            new_prompt = nn.ParameterList((self.gram_schmidt(prompt_tmp)))
            prompt_component['prompt'] = new_prompt
            self.prompt_pool.append(prompt_component)


    def freeze_prompt_params(self, train): 
        if train:
            last_id = self.task_cnt - 1 
        else:
            last_id = self.task_cnt
        for id, prompt_component in enumerate(self.prompt_pool):
            if id < last_id:
                for param in prompt_component.values():
                    if isinstance(param, nn.Parameter):
                        param.requires_grad = False
                    elif isinstance(param, nn.ParameterList):
                        for p in param:
                            p.requires_grad = False


    def activate_prompt_attention(self):
        for _, prompt_component in enumerate(self.prompt_pool):
            prompt_component['attention'].requires_grad = True
    

    def forward(self, query, train=False):
        self.freeze_prompt_params(train)

        # Calculating for similarity weight
        weight_prompt = []
        for prompt_component in self.prompt_pool:
            sim = cosine_similarity(query * prompt_component['attention'], prompt_component['key'])
            if isinstance(prompt_component['prompt'], nn.Parameter):
                prompt_component['prompt'] = prompt_component['prompt'] * sim
            elif isinstance(prompt_component['prompt'], nn.ParameterList):
                for i, _ in enumerate(prompt_component['prompt']):
                    prompt_component['prompt'][i] = prompt_component['prompt'][i] * sim
            weight_prompt.append(prompt_component)

        # Get summed prompt
        summed_prompt = None
        if isinstance(weight_prompt[0]['prompt'], nn.Parameter):
            summed_prompt = nn.Parameter(torch.zeros_like(weight_prompt[0]['prompt']), requires_grad=True)
        elif isinstance(weight_prompt[0]['prompt'], nn.ParameterList):
            summed_prompt = nn.ParameterList([nn.Parameter(torch.zeros_like(p), requires_grad=True) for p in weight_prompt[0]['prompt']])

        for prompt_component in weight_prompt:
            if isinstance(summed_prompt, nn.Parameter):
                summed_prompt = nn.Parameter(summed_prompt + prompt_component['prompt'])
            elif isinstance(summed_prompt, nn.ParameterList):
                for i, p in enumerate(prompt_component['prompt']):
                    summed_prompt[i] = nn.Parameter(summed_prompt[i] + p)

        return summed_prompt