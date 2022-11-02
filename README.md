# MLAB Transformers From Scratch

A documented and unit-tested repo to help you learn how to build transformer neural network models from scratch.

<p align="center">
	<img src="./transformer_architecture.png" width=45% alt="The Transformer model architecture from Vaswani et al. 2017"/>
</p>

## Introduction

### What is this?

- It seems useful to gain a deeper understanding of transformers by building some key models "from scratch" in a library like PyTorch, especially if you want to do AI safety research on them.
- Redwood Research runs a [Machine Learning for Alignment Bootcamp (MLAB)](https://www.alignmentforum.org/posts/3ouxBRRzjxarTukMW/apply-to-the-second-iteration-of-the-ml-for-alignment) in which one week consists of building BERT and GPT-2 from scratch, fine tuning them, and exploring some interpretability and training techniques.
- This repository takes the code from [the original MLAB repository](https://github.com/redwoodresearch/mlab) and cleans it up to make it easier for others to do this independently. The key differences between this repo and the original repo are:
	- Removed almost all the content besides the days about building BERT and GPT-2.
	- Created a starter [student](./mlab_tfs/bert/bert_student.py) file that has all the class and function stubs for the parts you'd need to build a transformer but without the implementation.
	- Migrated the original tests into a proper [unittest](https://docs.python.org/3/library/unittest.html) test suite and added several more unit tests for various functionality.
	- Added docstrings to document what each part should do and give hints for the trivial-but-annoying parts.
	- Implemented a new [solution](mlab_tfs/sample_solutions/bert_student_mukobi.py) file with all the new documentation to test that all the tests pass.
	- Various renaming and reorganizing of files to make the repo a bit cleaner.

### Status

- [BERT](./mlab_tfs/bert): Fully operational, tested, documented, and ready for you to build.
- [GPT-2](./mlab_tfs/gpt2): In progress, but I'm currently not prioritizing development on this.

## Getting Started

### Prerequisites

- Python 3.7+.
- Read about and use [the transformer architecture](https://docs.google.com/document/d/1b83_-eo9NEaKDKc9R3P5h5xkLImqMw8ADLmi__rkLo4/edit#heading=h.nlbz88ykqv3r).
- Learn how to use [Numpy](https://www.freecodecamp.org/learn/data-analysis-with-python/) and [PyTorch](https://pytorch.org/tutorials/).
- Recommended: Learn how to use [einops](http://einops.rocks/) and [einsum](https://rockt.github.io/2018/04/30/einsum).

### Installation

1. Create a virtual environment with venv or Anaconda and activate it if you like to work with virtual environments.
2. [Install PyTorch](https://pytorch.org/get-started/locally/) with the appropriate configuration for your environment.
3. Run `pip install -r requirements.txt` to install the other requirements for this repository.
4. Run `pip install -e .` to install the mlab_tfs package in an editable state. 

### Testing

This repo uses the built-in [unittest](https://docs.python.org/3/library/unittest.html) framework for evaluating your code.  You have a few options for running the tests
1. `python -m unittest mlab_tfs.tests.test_bert`
2. `python ./mlab_tfs/tests/test_bert.py`
3. If using an IDE like Visual Studio Code with the Python extension, the unit tests should already be discovered and show up in the Testing pane.

Most of the tests are in the form
- "Randomly initialize the student class and the reference class (from PyTorch or a solution file) with the same seed, pass the same input through it, and see if we get the same output."

but there are also tests for
- "Are the student class' attributes of the correct types?"
- or "Did the student cheat by calling the function from PyTorch that they're supposed to be implementing?"
- and there's one test class at the end that consists of "Instantiate the whole student transformer model and reference transformer model, load in the real BERT/GPT-2 weights from HuggingFace Transformers, pass a sequence of input tokens through, and see if we get the same output logits."

### Implementing Transformers

Now that you know how to test your code, go implement some transformers!
- Go to [the BERT folder](./mlab_tfs/bert).
- Read the instructions in the README file there.
- Reimplement the stubbed BERT classes and functions and pass the tests.

Note: Only the BERT folder is fully tested and documented, but you can also try writing your own `gpt2_student.py` and integrating it into the testing framework (please make a pull request to share this with others!).

### Known issues
- GPT-2 needs a starter file, better documentation (including a readme), and unit tests

## Further Exploration
(Copied from "Week 2: Implementing transformers" of the original MLAB repo)

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
)
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


## TODO for this repository

### General
- [X] Delete more non-transformers or non-essential stuff
- [X] Redo requirements by creating a fresh venv
- [X] Remove old git stuff (prune?) so it's a smaller download
- [X] Update this main readme
    - [X] Suggest learning http://einops.rocks/ via https://iclr.cc/virtual/2022/oral/6603 or http://einops.rocks/pytorch-examples.html
    - [X] Attribution
    - [X] Known issues
    - [X] Description of what this is
    - [X] Prerequisites
    - [X] Further exploration
    - [X] Testing
    - [X] Start implementing with link to each sub-readme
- [X] Set up Pylint with a config
- [X] Remove commented requirements now that requirements are verified

### BERT
- [X] Delete duplicated BERT solution file
- [X] Organize files to be much simpler (BERT and GPT2 folders)
- [X] Run testing code for BERT
- [X] Refactor testing code into unittest
- [X] Rename files to bert_reference, bert_student
- [X] Create starter file for you with empty stubs
- [X] Make testing code call starter code and compare to HF BERT and maybe MLAB solution
- [X] Update BERT readme to be more clear about what to do (e.g. no tokenizer) 
    - [X] Say it should be about 200 (or 150-300) lines of code
- [X] Include config or hyperparams or code to load weights
- [X] Change TODO into a changelist to describe differences from upstream (wrote some descriptions above)
- [X] Do BERT
- [X] Try removing __init__.py and other files if not used
- [ ] Replace existing `# Tensor[...` comments with `TensorType` type hints
- [X] Rewrite/fix the 2 commented out tests
- [ ] Integrate tests from the other archived files
- [X] Investigate [mocking](https://stackoverflow.com/questions/16134281/python-mocking-a-function-from-an-imported-module) to check that the student didn't use methods from torch.nn instead of implementing their own
- [ ] Add type hints to bert_student.py

### GPT
- [ ] Write GPT-2 readme
- [ ] Clean up GPT-2 folder (might not need to do much)
- [ ] Run testing code for GPT-2
- [ ] Refactor testing code into unittest
- [ ] Create starter file for you with empty stubs
- [ ] Make testing code call starter code and compare to HF GPT-2 and maybe MLAB solution
- [ ] Write GPT-2 readme (can say similar to the BERT folder or use similar content as that)
