import unittest
from torch import tensor as t
from torch import nn
from mlab_tfs.bert import bert_tao

# Base test class
# TODO move somewhere better


class MLTest(unittest.TestCase):
    def assertAllClose(self, my_out, their_out, tol=1e-5):
        self.assertTrue(t.allclose(my_out, their_out, rtol=1e-4, atol=tol))


# Utility functions.
# TODO move somewhere better
def get_pretrained_bert():
    pretrained_bert, _ = bert_tao.my_bert_from_hf_weights()
    return pretrained_bert


class TestBertEmbedding(MLTest):
    def test_embedding(self):
        Embedding = None
        input = t.randint(0, 10, (2, 3))
        t.manual_seed(1157)
        emb1 = Embedding(10, 5)
        t.manual_seed(1157)
        emb2 = nn.Embedding(10, 5)
        self.assertAllClose(emb1(input), emb2(input), "embedding")

    def test_bert_embedding_fn(your_fn):
        config = {
            "vocab_size": 28996,
            "hidden_size": 768,
            "max_position_embeddings": 512,
            "type_vocab_size": 2,
            "dropout": 0.1,
        }
        input_ids = t.randint(0, 2900, (2, 3))
        tt_ids = t.randint(0, 2, (2, 3))
        reference = bert_tao.BertEmbedding(config)
        reference.eval()
        self.assertAllClose(
            your_fn(
                input_ids=input_ids,
                token_type_ids=tt_ids,
                token_embedding=reference.token_embedding,
                token_type_embedding=reference.token_type_embedding,
                position_embedding=reference.position_embedding,
                layer_norm=reference.layer_norm,
                dropout=reference.dropout,
            ),
            reference(input_ids=input_ids, token_type_ids=tt_ids),
            "bert embedding",
        )

    def test_bert_embedding(your_module):
        config = {
            "vocab_size": 28996,
            "hidden_size": 768,
            "max_position_embeddings": 512,
            "type_vocab_size": 2,
            "dropout": 0.1,
        }
        input_ids = t.randint(0, 2900, (2, 3))
        tt_ids = t.randint(0, 2, (2, 3))
        t.random.manual_seed(0)
        reference = bert_tao.BertEmbedding(config)
        reference.eval()
        t.random.manual_seed(0)
        yours = your_module(**config)
        yours.eval()
        self.assertAllClose(
            yours(input_ids=input_ids, token_type_ids=tt_ids),
            reference(input_ids=input_ids, token_type_ids=tt_ids),
            "bert embedding",
        )


class TestBertAttention(MLTest):
    def test_attention_fn(fn):
        reference = bert_tao.multi_head_self_attention
        hidden_size = 768
        batch_size = 2
        seq_length = 3
        num_heads = 12
        token_activations = t.empty(
            batch_size, seq_length, hidden_size).uniform_(-1, 1)
        attention_pattern = t.rand(
            batch_size, num_heads, seq_length, seq_length)
        project_value = nn.Linear(hidden_size, hidden_size)
        project_output = nn.Linear(hidden_size, hidden_size)
        dropout = t.nn.Dropout(0.1)
        dropout.eval()
        self.assertAllClose(
            fn(
                token_activations=token_activations,
                num_heads=num_heads,
                attention_pattern=attention_pattern,
                project_value=project_value,
                # project_out=project_output,
                project_output=project_output,
                # dropout=dropout,
            ),
            reference(
                token_activations=token_activations,
                num_heads=num_heads,
                attention_pattern=attention_pattern,
                project_value=project_value,
                project_out=project_output,
                dropout=dropout,
            ),
            "attention",
        )

    def test_attention_pattern_fn(fn):
        reference = bert_tao.raw_attention_pattern
        hidden_size = 768
        token_activations = t.empty(2, 3, hidden_size).uniform_(-1, 1)
        num_heads = 12
        project_query = nn.Linear(hidden_size, hidden_size)
        project_key = nn.Linear(hidden_size, hidden_size)
        self.assertAllClose(
            fn(
                token_activations=token_activations,
                num_heads=num_heads,
                project_query=project_query,
                project_key=project_key,
            ),
            reference(
                token_activations=token_activations,
                num_heads=num_heads,
                project_query=project_query,
                project_key=project_key,
            ),
            "attention pattern raw",
        )

    def test_attention_pattern_single_head(fn):
        reference = bert_tao.raw_attention_pattern
        hidden_size = 768
        token_activations = t.empty(2, 3, hidden_size).uniform_(-1, 1)
        num_heads = 12
        project_query = nn.Linear(hidden_size, hidden_size)
        project_key = nn.Linear(hidden_size, hidden_size)
        head_size = hidden_size // num_heads
        project_query_ub = nn.Linear(hidden_size, head_size)
        project_query_ub.weight = nn.Parameter(
            project_query.weight[:head_size])
        project_query_ub.bias = nn.Parameter(project_query.bias[:head_size])
        project_key_ub = nn.Linear(hidden_size, head_size)
        project_key_ub.weight = nn.Parameter(project_key.weight[:head_size])
        project_key_ub.bias = nn.Parameter(project_key.bias[:head_size])
        self.assertAllClose(
            fn(
                token_activations=token_activations[0],
                project_query=project_query_ub,
                project_key=project_key_ub,
            ),
            reference(
                token_activations=token_activations,
                num_heads=num_heads,
                project_query=project_query,
                project_key=project_key,
            )[0, 0, :, :],
            "attention pattern raw",
        )

    def test_bert_attention(your_module):
        config = {
            "vocab_size": 28996,
            "intermediate_size": 3072,
            "hidden_size": 768,
            "num_layers": 12,
            "num_heads": 12,
            "max_position_embeddings": 512,
            "dropout": 0.0,  # not testing dropout!!
            "type_vocab_size": 2,
        }
        t.random.manual_seed(0)
        reference = bert_tao.SelfAttentionLayer(config)
        reference.eval()
        t.random.manual_seed(0)
        theirs = your_module(
            hidden_size=config["hidden_size"],
            num_heads=config["num_heads"],
            # dropout=config["dropout"],
        )
        theirs.eval()
        input_activations = t.rand((2, 3, 768))
        self.assertAllClose(
            theirs(input_activations),
            reference(input_activations),
            "bert",
        )

    def test_bert_attention_pattern(your_module):
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
        reference = bert_tao.AttentionPattern(config)
        reference.eval()
        t.random.manual_seed(0)
        theirs = your_module(
            hidden_size=config["hidden_size"],
            num_heads=config["num_heads"],
            dropout=config["dropout"],
        )
        theirs.eval()
        input_activations = t.rand((2, 3, 768))
        self.assertAllClose(
            theirs(input_activations),
            reference(input_activations),
            "bert",
        )
