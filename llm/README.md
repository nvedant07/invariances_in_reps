### Code for initial investigations on LLMs

Some details about how models like chatGPT are trained:

1. Take a large transformer model (untrained model)
2. Pre-trained using language modeling objective on an extremely large corpus (foundation model)
3. Perform supervised finetuning on an instruction dataset (instruction models)
4. Perform RLHF to align instruction models with human preferences

We are currently starting with a bunch of available foundation models (i.e. models that have already been pre-trained in step 2) and are focusing on understanding supervised finetuning (3). We are also interested in analyzing (4).

#### Foundation Models

Some publicly available foundation models include:

 * Pythia suite of models by EleutherAI, these come in 8 sizes: 70M, 160M, 410M, 1B, 1.4B, 2.8B, 6.9B, and 12B. All models can be found on huggingface (https://huggingface.co/EleutherAI/pythia-12b) and in general you can find any model by changing. To avoid duplicating, I'd recommend using the ones I already have on our servers: ``/NS/llm-1/nobackup/vnanda/llm_base_models/pythia-*``.
 * [LLAMA models](https://ai.facebook.com/blog/large-language-model-llama-meta-ai/) trained by Meta. These are not open source, but we have official access to these weights. Since these weights are liscence bound and releasing them can have some legal consequences, I'd request not sharing these with anyone besides people on the project. You can find them on our local shared storage: ``/NS/llm-1/nobackup/vnanda/llm_base_models/llama``.
 * [GPT-J](https://huggingface.co/EleutherAI/gpt-j-6b) was an earlier model by ElueutherAI aimed at replicating GPT-3 and it contains 6B params. You can find it locally here: ``/NS/llm-1/nobackup/vnanda/llm_base_models/gpt-j-6b``
 * [MPT-7b](https://huggingface.co/mosaicml/mpt-7b) is a very new model by MosiacML containing 7B params. It's also available locally: ``/NS/llm-1/nobackup/vnanda/llm_base_models/mpt-7b``


#### Instruction Datasets:

 * [Dolly](https://huggingface.co/datasets/databricks/databricks-dolly-15k)
 * [Alpaca](https://huggingface.co/datasets/tatsu-lab/alpaca) and its [cleaned version](https://huggingface.co/datasets/yahma/alpaca-cleaned)
 * [HC3](https://huggingface.co/datasets/Hello-SimpleAI/HC3)
 * [Evol-instruct](https://huggingface.co/datasets/victor123/evol_instruct_70k)

All these datasets exist at ``/NS/twitter-9/work/vnanda/invariances_in_reps/llm/data`` but since these are small you can also download them to your local working repos.

#### Finetuning foundation models on instruction datasets

In order to finetune I have a script ``finetune_instructions.py``. Some examples of parameters to this script can be found under ``scripts/``. 

 * Usual use on cluster nodes: we use deepspeed launcher for usual jobs since it's simple and does sync across nodes. Examples can be found in ``scripts/pythia-2.8b-ft-1.sh``.
 * SLURM: For launching jobs on SLURM, examples are included in ``scripts/pythia-1.4b-*.sh``. deepspeed is not supported for SLURM runs and hence we use torch.distributed to launch these runs.

To reproduce Out-Of-Memory (OOM) error, you can use the command in ``scripts/pythia-12b-ft-1.sh``.

