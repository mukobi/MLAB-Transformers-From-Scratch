import torch as t
import transformers
from . import bert_tao as bert
import torch.nn as nn
import torch.nn.functional as F
from mlab_tfs.utils.mlab_utils import tpeek


def test_bert_mlp(fn):
    reference = bert.bert_mlp
    hidden_size = 768
    intermediate_size = 4 * hidden_size

    token_activations = t.empty(2, 3, hidden_size).uniform_(-1, 1)
    mlp_1 = nn.Linear(hidden_size, intermediate_size)
    mlp_2 = nn.Linear(intermediate_size, hidden_size)
    dropout = t.nn.Dropout(0.1)
    dropout.eval()
    allclose(
        fn(token_activations=token_activations, linear_1=mlp_1, linear_2=mlp_2),
        reference(
            token_activations=token_activations,
            linear_1=mlp_1,
            linear_2=mlp_2,
            dropout=dropout,
        ),
        "bert mlp",
    )


def test_layer_norm(LayerNorm):
    ln1 = LayerNorm(10)
    ln2 = nn.LayerNorm(10)
    tensor = t.randn(20, 10)
    allclose(ln1(tensor), ln2(tensor), "layer norm")

    # TODO maybe incorporate this from tests/nn_functional.py
    # random_weight = t.empty(9).uniform_(0.8, 1.2)
    # random_bias = t.empty(9).uniform_(-0.1, 0.1)
    # random_input = t.empty(8, 9)
    # their_output = reference.layer_norm(random_input, random_weight, random_bias)
    # my_output = fn(random_input, random_weight, random_bias)
    # allclose(my_output, their_output, "layer norm")


# TODO write this
def test_bert(your_module):
    config = {
        "vocab_size": 28996,
        "intermediate_size": 3072,
        "hidden_size": 768,
        "num_layers": 12,
        "num_heads": 12,
        "max_position_embeddings": 512,
        "dropout": 0.1,
        "type_vocab_size": 2,
    }
    t.random.manual_seed(0)
    reference = bert.Bert(config)
    reference.eval()
    t.random.manual_seed(0)
    theirs = your_module(**config)
    theirs.eval()
    tokenizer = transformers.AutoTokenizer.from_pretrained("bert-base-cased")
    input_ids = tokenizer("hello there", return_tensors="pt")["input_ids"]
    allclose(
        theirs(input_ids=input_ids),
        reference(input_ids=input_ids).logits,
        "bert",
    )


def test_bert_classification(your_module):
    config = {
        "vocab_size": 28996,
        "intermediate_size": 3072,
        "hidden_size": 768,
        "num_layers": 12,
        "num_heads": 12,
        "max_position_embeddings": 512,
        "dropout": 0.1,
        "type_vocab_size": 2,
        "num_classes": 2,
    }
    t.random.manual_seed(0)
    reference = bert.Bert(config)
    reference.eval()
    t.random.manual_seed(0)
    theirs = your_module(**config)
    theirs.eval()
    tokenizer = transformers.AutoTokenizer.from_pretrained("bert-base-cased")
    input_ids = tokenizer("hello there", return_tensors="pt")["input_ids"]
    logits, classifs = theirs(input_ids=input_ids)
    allclose(
        logits,
        reference(input_ids=input_ids).logits,
        "bert",
    )

    allclose(
        classifs,
        reference(input_ids=input_ids).classification,
        "bert",
    )


def test_same_output(your_bert, pretrained_bert, tol=1e-4):
    vocab_size = pretrained_bert.embedding.token_embedding.weight.shape[0]
    input_ids = t.randint(0, vocab_size, (10, 20))
    allclose(
        your_bert.eval()(input_ids),
        pretrained_bert.eval()(input_ids).logits,
        "comparing Berts",
        tol=tol,
    )


def test_bert_block(your_module):
    config = {
        "vocab_size": 28996,
        "intermediate_size": 3072,
        "hidden_size": 768,
        "num_layers": 12,
        "num_heads": 12,
        "max_position_embeddings": 512,
        "dropout": 0.1,
        "type_vocab_size": 2,
    }
    t.random.manual_seed(0)
    reference = bert.BertBlock(config)
    reference.eval()
    t.random.manual_seed(0)
    theirs = your_module(
        intermediate_size=config["intermediate_size"],
        hidden_size=config["hidden_size"],
        num_heads=config["num_heads"],
        dropout=config["dropout"],
    )
    theirs.eval()
    input_activations = t.rand((2, 3, 768))
    allclose(
        theirs(input_activations),
        reference(input_activations),
        "bert",
    )
