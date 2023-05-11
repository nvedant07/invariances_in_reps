### Code for initial investigations on LLMs

Some details about how models like chatGPT are trained:

1. Take a large transformer model (untrained model)
2. Pre-trained using language modeling objective on an extremely large corpus (foundation model)
3. Perform supervised finetuning on an instruction dataset (instruction models)
4. Perform RLHF to align instruction models with human preferences

We are currently starting with a bunch of available foundation models (i.e. models that have already been pre-trained in step 2) and are focusing on understanding supervised finetuning (3). We are also interested in analyzing (4).


Some publicly available foundation models include:

 * Pythia suite of models, these come in 8 sizes: 70M, 160M, 410M, 1B, 1.4B, 2.8B, 6.9B, and 12B. All models can be found on huggingface (https://huggingface.co/EleutherAI/pythia-12b) and in general you can find any model by changing. 
 * LLAMA models trained by Meta (https://ai.facebook.com/blog/large-language-model-llama-meta-ai/)


