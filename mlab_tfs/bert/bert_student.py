import math

import numpy as np
import torch as t
from torch import einsum
from torch import nn
from torch.nn import functional as F
from einops import rearrange, repeat
from torchtyping import TensorType


class LayerNorm(nn.Module):
    """
    Layer normalization. See https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html

    Args:
        normalized_shape (int or list or torch.Size): input shape from an expected input of size
        [*, normalized_shape[0], normalized_shape[1], ..., normalized_shape[-1]]
        If a single integer is used, it is treated as a singleton list, and this module will
        normalize over the last dimension which is expected to be of that specific size.

    Attributes:
        LayerNorm.weight: The learnable weights gamma of the module of shape normalized_shape.
        The values are initialized to 1.

        LayerNorm.bias: The learnable bias beta of the module of shape normalized_shape.
        The values are initialized to 0.

    Dependencies:
        None.

    Hints:
        Norm over the last len(normalized_shape) dimensions, not simply all but the first dimension.
    """

    def __init__(self, normalized_shape: int):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.weight = nn.Parameter(t.ones(normalized_shape))
        self.bias = nn.Parameter(t.zeros(normalized_shape))

    def forward(self, input: TensorType[...]):
        """Apply Layer Normalization over a mini-batch of inputs."""
        num_normed_dims = self.weight.dim()
        normalizing_dims = tuple(range(-num_normed_dims, 0))  # (-(n-1), ..., -2, -1)
        var, mean = t.var_mean(input, normalizing_dims, unbiased=False)
        # Stack up to the dimension of input
        var = rearrange(var, 'x ... -> x ...' + ' ()' * num_normed_dims)
        mean = rearrange(mean, 'x ... -> x ...' + ' ()' * num_normed_dims)
        eps = 1e-05
        return (input - mean) / t.sqrt(var + eps) * self.weight + self.bias


class Embedding(nn.Module):
    """
    A simple lookup table storing embeddings of a fixed dictionary and size.
    See https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html

    This module is often used to store word embeddings and retrieve them using indices. The input
    to the module is a list of indices, and the summed_embeddings is the corresponding word embeddings.

    Args:
        Embedding.weight (Tensor): The learnable weights of the module of shape
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
    """
    BERT embedding layer.

    See §3.4 of Attention Is All You Need https://arxiv.org/pdf/1706.03762.pdf#page=5

    You should create Embedding parameters for tokens, positions, and token types/segments.
    BERT uses learned position embeddings rather than sinusoidal position embeddings.

    The forward pass sums the three embeddings then passes them through LayerNorm and Dropout.

    Note that BERT does not scale the embeddings by sqrt(d_model).

    Args:
        vocab_size, hidden_size, max_position_embeddings, type_vocab_size (int): Embeddings dims.

        dropout: Dropout rate.

    Attributes:
        BertEmbedding.token_embedding (Embedding): Token embeddings.

        BertEmbedding.position_embedding (Embedding): Position embeddings.

        BertEmbedding.token_type_embedding (Embedding): Token type/segment embeddings.

        BertEmbedding.layer_norm (LayerNorm): Layer normalization.

        BertEmbedding.dropout (nn.Dropout): Dropout layer.

    Dependencies:
        Embedding, LayerNorm

    Hints:
        Initialization order matters for our seeded random tests. Initialize token_embedding, then
        position_embedding, then token_type_embedding.

        Use torch.arange to create an ascending integer list to index into your position embeddings.
        You'll have to take care to repeat/expand your tensors to the appropriate sizes so they sum.
    """

    def __init__(self, vocab_size: int, hidden_size: int, max_position_embeddings: int,
                 type_vocab_size: int, dropout: float):
        super().__init__()
        self.token_embedding = Embedding(vocab_size, hidden_size)
        self.position_embedding = Embedding(max_position_embeddings, hidden_size)
        self.token_type_embedding = Embedding(type_vocab_size, hidden_size)
        self.layer_norm = LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids, token_type_ids):
        """Add embeddings and apply layer norm and dropout."""
        num_batches, num_tokens = input_ids.shape
        token_embeddings = self.token_embedding(input_ids)
        token_type_embeddings = self.token_type_embedding(token_type_ids)
        position_embeddings = self.position_embedding(t.arange(num_tokens))
        # Expand position embeddings for each batch.
        position_embeddings = repeat(position_embeddings, 't d -> b t d', b=num_batches)

        embeddings = position_embeddings + token_embeddings + token_type_embeddings
        return self.dropout(self.layer_norm(embeddings))


class GELU(nn.Module):
    """
    Applies the Gaussian Error Linear Units function with no approximation.
    See https://pytorch.org/docs/stable/generated/torch.nn.GELU.html

    GELU(x) = x * Φ(x) where Φ(x) is the Cumulative Distribution Function for Gaussian Distribution.

    Dependencies:
        None.

    Hint:
        The CDF of a normal distribution with mean 0 and variance 1 is
        0.5 * (1 + erf(value / sqrt(2))) where erf is the error function (torch.erf).
    """

    def forward(self, input):
        """Apply the GELU function."""
        return input * 0.5 * (1 + t.erf(input / np.sqrt(2)))


class BertMLP(nn.Module):
    """
    BERT MLP/fully-connected feed forward network layers.
    See 3.3 in Attention Is All You Need https://arxiv.org/pdf/1706.03762.pdf#page=5

    Implement 2 linear layers that go from hidden_size to intermediate_size and back
    with a GELU activation in the middle.

    Args:
        hidden_size (int): Token embedding size.

        intermediate_size (int): Intermediate hidden layer size.

    Attributes:
        BertMLP.lin1 (nn.Linear): First linear layer.

        BertMLP.lin2 (nn.Linear): Second linear layer.

    Dependencies:
        GELU
    """

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.lin1 = nn.Linear(hidden_size, intermediate_size)
        self.gelu = GELU()
        self.lin2 = nn.Linear(intermediate_size, hidden_size)

    def forward(self, input: TensorType['batch_size', 'seq_length', 'hidden_size']
                ) -> TensorType['batch_size', 'seq_length', 'hidden_size']:
        """Apply linear projection, GELU, and another linear projection."""
        return self.lin2(self.gelu(self.lin1(input)))


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
