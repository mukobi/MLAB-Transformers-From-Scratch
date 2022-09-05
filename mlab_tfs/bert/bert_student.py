import math

import torch as t
from torch import einsum
from torch import nn
from torch.nn import functional as F
from einops import rearrange, repeat
from torchtyping import TensorType


class LayerNorm(nn.Module):
    """
    Layer normalization. See https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html

    Parameters:
        normalized_shape (int or list or torch.Size): input shape from an expected input of size
        [*, normalized_shape[0], normalized_shape[1], ..., normalized_shape[-1]]
        If a single integer is used, it is treated as a singleton list, and this module will
        normalize over the last dimension which is expected to be of that specific size.

    Variables:
        LayerNorm.weight: the learnable weights gamma of the module of shape normalized_shape.
        The values are initialized to 1.

        LayerNorm.bias: the learnable bias beta of the module of shape normalized_shape.
        The values are initialized to 0.

    Dependencies:
        None.

    Hints:

    """

    def __init__(self, normalized_shape: int):
        super().__init__()
        self.weight = nn.Parameter(t.ones(normalized_shape))
        self.bias = nn.Parameter(t.zeros(normalized_shape))

    def forward(self, input: TensorType[...]):
        """Applies Layer Normalization over a mini-batch of inputs."""
        all_but_first_dims = tuple(range(1, len(input.shape)))  # (1, 2, ..., n-1)
        var, mean = t.var_mean(input, all_but_first_dims, unbiased=False)
        var = rearrange(var, 'var -> var ()')
        mean = rearrange(mean, 'mean -> mean ()')
        eps = 1e-05
        return (input - mean) / t.sqrt(var + eps) * self.weight + self.bias


class Embedding(nn.Module):
    """
    A simple lookup table storing embeddings of a fixed dictionary and size.
    See https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html

    This module is often used to store word embeddings and retrieve them using indices. The input
    to the module is a list of indices, and the output is the corresponding word embeddings.

    Variables:
        Embedding.weight (Tensor): the learnable weights of the module of shape
        (num_embeddings, embedding_dim) initialized from a normal distribution (mu=0, sigma=1).

    Dependencies:
        None.

    Hints:
        Use torch.randn to create a random tensor sampled by a normal distribution.
    """

    def __init__(self, vocab_size, embed_size):
        super().__init__()
        self.weight = nn.Parameter(t.randn((vocab_size, embed_size)))

    def forward(self, input):
        """Look up the input list of indices in the embedding matrix."""
        return self.weight[input]


class BertEmbedding(nn.Module):
    def __init__(self, vocab_size, hidden_size, max_position_embeddings, type_vocab_size,
                 dropout: float):
        super().__init__()
        raise NotImplementedError

    def forward(self, input_ids, token_type_ids):
        raise NotImplementedError


def raw_attention_pattern(
    token_activations,  # Tensor[batch_size, seq_length, hidden_size(768)],
    num_heads,
    project_query,      # nn.Module, (Tensor[..., 768]) -> Tensor[..., 768],
    project_key,        # nn.Module, (Tensor[..., 768]) -> Tensor[..., 768]
):  # -> Tensor[batch_size, head_num, key_token: seq_length, query_token: seq_length]:
    raise NotImplementedError


def bert_attention(
    token_activations,  # : Tensor[batch_size, seq_length, hidden_size (768)],
    num_heads: int,
    # : Tensor[batch_size,num_heads, seq_length, seq_length],
    attention_pattern,
    project_value,  # : function( (Tensor[..., 768]) -> Tensor[..., 768] ),
    project_output,  # : function( (Tensor[..., 768]) -> Tensor[..., 768] )
):  # -> Tensor[batch_size, seq_length, hidden_size]
    raise NotImplementedError


class MultiHeadedSelfAttention(nn.Module):
    def __init__(self, num_heads, hidden_size):
        super().__init__()
        raise NotImplementedError

    def forward(self, input):  # b n l
        raise NotImplementedError


def bert_mlp(token_activations,  # : torch.Tensor[batch_size,seq_length,768],
             linear_1: nn.Module, linear_2: nn.Module
             ):  # -> torch.Tensor[batch_size, seq_length, 768]
    raise NotImplementedError


class BertMLP(nn.Module):
    def __init__(self, input_size: int, intermediate_size: int):
        super().__init__()
        raise NotImplementedError

    def forward(self, input):
        raise NotImplementedError


class BertBlock(nn.Module):
    def __init__(self, hidden_size, intermediate_size, num_heads, dropout: float):
        super().__init__()
        raise NotImplementedError

    def forward(self, input):
        raise NotImplementedError


class Bert(nn.Module):
    def __init__(self, vocab_size, hidden_size, max_position_embeddings, type_vocab_size,
                 dropout, intermediate_size, num_heads, num_layers):
        super().__init__()
        raise NotImplementedError

    def forward(self, input_ids):
        raise NotImplementedError


class BertWithClassify(nn.Module):
    def __init__(self, vocab_size, hidden_size, max_position_embeddings, type_vocab_size,
                 dropout, intermediate_size, num_heads, num_layers, num_classes):
        super().__init__()
        raise NotImplementedError

    def forward(self, input_ids):
        raise NotImplementedError