import torch
from typing import Dict, Optional
from architectures.inverted_rep_callback import InvertedRepWrapper
from architectures.inference import inference_with_features


class StretchedInvertedRepWrapper(InvertedRepWrapper):

    def __init__(self, model: torch.nn.Module, seed: torch.Tensor, *args, **kwargs) -> None:
        super().__init__(model, seed, *args, **kwargs) # seed is set in InvertedRepWrapper's __init__

    def normalize(self, vec):
        return torch.div(vec, torch.linalg.norm(vec, ord=2))

    def construct_stretch_vector(self, 
                                 original_reps: torch.Tensor, 
                                 vector_type: str, 
                                 vector_params: Dict,
                                 stretch_factor: float):
        if vector_type == 'class_logit':
            assert 'index' in vector_params, 'Must pass class index'
            final_layer = list(self.model.named_children())[-1][1]
            logit_concept = final_layer.weight[vector_params['index']]
            logit_concept = self.normalize(logit_concept)
            return original_reps + stretch_factor * logit_concept
        elif vector_type == 'random':
            assert 'mean' in vector_params and 'std' in vector_params
            random_concept = torch.add(
                torch.multiply(
                    torch.randn((1, *original_reps.shape[1:]), device=self.device), 
                    vector_params['std']
                ))
            random_concept = self.normalize(random_concept)
            return original_reps + stretch_factor * random_concept
        else:
            raise ValueError(f'{vector_type} not supported')

    def forward(self, x, *args, **kwargs):
        """
        kwargs: needed to form a vector which will be used to stretch the representations
            vector_type (str): how to construct the stretch vector. 
                Options: ['class_logit', 'random']
            stretch_factor (float): amount with which the target_reps will be stretched
            vector_kwargs (dict): extra info about constructing vector
        Final target_reps = target_reps + stretch_factor * vector / || vector ||
        """
        assert hasattr(self, 'vector_kwargs'), \
            'Must pass a callback that initializes vector_kwargs'
        target_rep = inference_with_features(self.model, self.normalizer(x), *args)[1].detach()
        target_rep = self.construct_stretch_vector(target_rep, **self.vector_kwargs)
        # .to(device) is unavoidable here, this way is much faster than putting a tensor of 
        # [batch, channels, wdith, height] on GPU
        seeds = torch.ones((len(x), *self.seed.shape), device=self.device) * \
            self.seed.unsqueeze(0).to(self.device)
        ir, _ = self.attacker(seeds, target_rep, **kwargs)
        return x, ir
