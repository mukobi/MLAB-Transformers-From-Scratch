# MLAB Transformers From Scratch

TODO short description

## Introduction

TODO
- This is mostly based off Week 2: Implementing Transformers from Redwood Research's Machine Learning for Alignment Bootcamp (MLAB). The code in this repository is mostly cleaned up code from [the original MLAB repository](https://github.com/redwoodresearch/mlab).

### Prerequisites

## Getting Started

### Installation

1. Install Python 3.6+.
2. Create a virtual environment with venv or Anaconda and activate it if you like to work with virtual environments.
3. [Install PyTorch](https://pytorch.org/get-started/locally/) with the appropriate configuration for your environment.
4. Run `pip install -r requirements.txt` to install the other requirements for this repository.

### Testing

TODO

### Implementing Transformers

TODO

## Week 2: Implementing transformers
Resources:
- https://huggingface.co/course/chapter1
- https://nostalgebraist.tumblr.com/post/185326092369/1-classic-fully-connected-neural-networks-these
- https://nostalgebraist.tumblr.com/post/188730918209/huh-the-alphastar-paper-is-finally-up-linked-in
- GPT-2 paper.
- Implement GPT-2. Check that it matches. Check that it performs similarly if you drop it in as a replacement in a HF training loop.
	- https://github.com/openai/gpt-2/blob/master/src/model.py
- Implement BERT. Check that it does the same thing as HF implementation, including gradients.
	- https://github.com/huggingface/transformers/blob/5b317f7ea4fcd012d537f0a1e3c73aef82f9b70a/examples/research_projects/movement-pruning/emmental/modeling_bert_masked.py
	- Hmm. This is kind of a lot of code. The main part that is hard is BertSelfAttention.
	- Shorter implementation: https://github.com/codertimo/BERT-pytorch
- DeBERTa.
	- maybe we just get them to implement the key smart parts, rather than trying to get all the details of hooking things together correct.
- BYO Tokenizer. The key subword tokenizing function is [here](https://github.com/huggingface/transformers/blob/5b317f7ea4fcd012d537f0a1e3c73aef82f9b70a/src/transformers/models/bert/tokenization_bert.py#L509)
	- I claim that this is worth people's time.
	- It's really easy to test this.
- Sampler.
	- https://towardsdatascience.com/how-to-sample-from-language-models-682bceb97277
	- Batched sampler
	- Sampler for pair of contexts.
	- Sampler for "-- and that was bad"
		- efficiently
- Gwern shit
	- https://www.alignmentforum.org/posts/k7oxdbNaGATZbtEg3/redwood-research-s-current-project?commentId=CMvfybQCk6sxtLFMN


## TODO
- [X] Delete more non-transformers or non-essential stuff
- [X] Delete duplicated BERT solution file
- [X] Organize files to be much simpler (BERT and GPT2 folders)
- [X] Redo requirements by creating a fresh venv
- [ ] Run testing code for BERT
- [ ] Refactor testing code into unittest
- [ ] Create starter file for you with empty stubs
- [ ] Make testing code call starter code and compare to HF BERT and maybe MLAB solution
- [ ] Update BERT readme to be more clear about what to do (e.g. no tokenizer) 
    - [ ] Say it should be about 200 (or 150-300) lines of code
- [ ] Update this main readme
    - [ ] Suggest learning http://einops.rocks/ via https://iclr.cc/virtual/2022/oral/6603 or http://einops.rocks/pytorch-examples.html
    - [ ] Attribution
    - [ ] Known issues
    - [ ] Description of what this is
    - [ ] Prerequisites
    - [ ] Further exploration
    - [ ] Testing
    - [ ] Start implementing with link to each sub-readme
- [ ] Include config or hyperparams or code to load weights
- [ ] Change TODO into a changelist to describe differences from upstream
- [ ] Do BERT
- [ ] Remove commented requirements now that requirements are verified
- [ ] Write BERT readme
- [ ] Clean up GPT-2 folder (might not need to do much)
- [ ] Run testing code for GPT-2
- [ ] Refactor testing code into unittest
- [ ] Create starter file for you with empty stubs
- [ ] Make testing code call starter code and compare to HF GPT-2 and maybe MLAB solution
- [ ] Write GPT-2 readme (can say similar to the BERT folder or use similar content as that)
