### Code for initial investigations on LLMs

Some details about how models like chatGPT are trained:

1. Take a large transformer model (untrained model)
2. Pre-trained using language modeling objective on an extremely large corpus (foundation model)
3. Perform supervised finetuning on an instruction dataset (instruction models)
4. Perform RLHF to align instruction models with human preferences

We are currently starting with a bunch of available foundation models (i.e. models that have already been pre-trained in step 2) and are focussing on understanding supervised finetuning (3).



