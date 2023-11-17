## Purpose

This repo is intended to be a collection of my PhD projects (inlcuding messy, behind the scenes side of research). Each dir represents a project. Most stuff got published and has accompanied repos that are cleaner versions of code in this repo. 

 * ``human_nn_alignment``: We studied alignment of many DNNs with human perception, including human studies and perceptual measures of similarity. Got published at AAAI (Oral) 2023. [Paper](https://arxiv.org/abs/2111.14726)
 * ``effects_of_finetuning``: We studied how invariances change as we finetune a bunch of vision models on different tasks. Used [STIR](https://github.com/nvedant07/STIR) to analyze shared invariances. Got published at [CoLLAs 2023](https://lifelong-ml.cc/). [Paper](https://arxiv.org/abs/2307.06006). 
 * ``partially_inverted_reps``: Showed that not all neurons are needed to transfer to many downstream tasks -- implying that features are spread throughout the network with a degree of diffused redundancy. Got published at NeurIPS 2023. [Paper](https://arxiv.org/abs/2306.00183)
 * ``equivariance``: Unpublished. The idea was to check if stretching a feature vector in representation space along a concept leads to meaningful changes when inverting the representation.