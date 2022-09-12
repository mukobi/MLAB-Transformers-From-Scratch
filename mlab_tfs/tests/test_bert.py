import unittest
from unittest.mock import patch
import re
import copy

import torch as t
from torch import nn
import transformers
from torchtyping import TensorType

from mlab_tfs.bert import bert_student, bert_reference
from mlab_tfs.utils.mlab_utils import itpeek

# Config
BERT_CONFIG_STANDARD = {
    "vocab_size": 28996,
    "intermediate_size": 3072,
    "hidden_size": 768,
    "num_layers": 12,
    "num_heads": 12,
    "max_position_embeddings": 512,
    "dropout": 0.1,
    "type_vocab_size": 2,
}
BERT_CONFIG_NO_DROPOUT = copy.deepcopy(BERT_CONFIG_STANDARD)
BERT_CONFIG_NO_DROPOUT["dropout"] = 0.0

# Base test class
# TODO move somewhere better


class MLTest(unittest.TestCase):
    """Base test case class."""

    def assert_tensors_close(self, student_out: TensorType, reference_out: TensorType, tol=1e-5):
        """Assert that two tensors have the same size and all elements are close."""
        message = f'Mismatched shapes!\nExpected:\n{student_out.shape}\nFound:\n{reference_out.shape}'
        self.assertEqual(reference_out.shape, student_out.shape, message)

        message = f'Not all values are close!\nExpected:\n{itpeek(reference_out)}\nFound:\n{itpeek(student_out)}'
        self.assertTrue(t.allclose(reference_out, student_out, rtol=1e-4, atol=tol), message)


# Utility functions.
# TODO move somewhere better
def get_pretrained_bert():
    """Get just the pretrained reference BERT, discarding the HF BERT."""
    pretrained_bert, _ = bert_reference.my_bert_from_hf_weights()
    return pretrained_bert


def mapkey(key):
    """Map a key from Hugging Face BERT's parameters to our BERT's parameters."""
    key = re.sub(r'^embedding\.', 'embed.', key)
    key = re.sub(r'\.position_embedding\.', '.pos_embedding.', key)
    key = re.sub(r'^lm_head\.mlp\.', 'lin.', key)
    key = re.sub(r'^lm_head\.unembedding\.', 'unembed.', key)
    key = re.sub(r'^lm_head\.layer_norm\.', 'layer_norm.', key)
    key = re.sub(r'^transformer\.([0-9]+)\.layer_norm',
                 'blocks.\\1.layernorm1', key)
    key = re.sub(r'^transformer\.([0-9]+)\.attention\.pattern\.',
                 'blocks.\\1.attention.', key)
    key = re.sub(r'^transformer\.([0-9]+)\.residual\.layer_norm\.',
                 'blocks.\\1.layernorm2.', key)

    key = re.sub(r'^transformer\.', 'blocks.', key)
    key = re.sub(r'\.project_out\.', '.project_output.', key)
    key = re.sub(r'\.residual\.mlp', '.mlp.lin', key)
    return key


class TestBertLayerNorm(MLTest):
    """Test layer normalization functionality."""

    @ patch('torch.nn.functional.layer_norm')
    def test_no_cheating(self, patched_layer_norm):
        """Test that the student doesn't call the PyTorch version."""
        ln1 = bert_student.LayerNorm(2)
        input = t.randn(10, 2)
        ln1(input)
        patched_layer_norm.assert_not_called()

    def test_layer_norm_2d(self):
        """Test a 2D input tensor."""
        ln1 = bert_student.LayerNorm(10)
        ln2 = nn.LayerNorm(10)
        t.random.manual_seed(42)
        input = t.randn(20, 10)
        self.assert_tensors_close(ln1(input), ln2(input))

    def test_layer_norm_3d(self):
        """Test a 3D input tensor."""
        ln1 = bert_student.LayerNorm((10, 5))
        ln2 = nn.LayerNorm((10, 5))
        t.random.manual_seed(42)
        input = t.randn(20, 10, 5)
        self.assert_tensors_close(ln1(input), ln2(input))

    def test_layer_norm_transformer(self):
        """Test a transformer-sized input tensor."""
        ln1 = bert_student.LayerNorm(768)
        ln2 = nn.LayerNorm(768)
        t.random.manual_seed(42)
        input = t.randn(20, 32, 768)
        self.assert_tensors_close(ln1(input), ln2(input))

    def test_layer_norm_variables(self):
        """Modify weight and bias and test for the same output."""
        ln1 = bert_student.LayerNorm(10)
        ln2 = nn.LayerNorm(10)
        t.random.manual_seed(42)
        weight = nn.Parameter(t.randn(10))
        bias = nn.Parameter(t.randn(10))
        ln1.weight.set = weight
        ln2.weight.set = weight
        ln1.bias = bias
        ln2.bias = bias
        input = t.randn(20, 10)
        self.assert_tensors_close(ln1(input), ln2(input))


class TestBertEmbedding(MLTest):
    """Test embedding functionality."""

    @patch('torch.nn.functional.layer_norm')
    @patch('torch.nn.Embedding.forward')
    def test_no_cheating(self, patched_embedding, patched_layer_norm):
        """Test that the student doesn't call the PyTorch version."""
        emb1 = bert_student.Embedding(10, 5)
        random_input = t.randint(0, 10, (2, 3))
        emb1(random_input)
        patched_embedding.assert_not_called()
        patched_layer_norm.assert_not_called()

    def test_attribute_types(self):
        """Test the types of the module's attributes."""
        emb1 = bert_student.BertEmbedding(28996, 768, 512, 2, 0.1)
        self.assertIsInstance(emb1.position_embedding, bert_student.Embedding)
        self.assertIsInstance(emb1.token_embedding, bert_student.Embedding)
        self.assertIsInstance(emb1.token_type_embedding, bert_student.Embedding)
        self.assertIsInstance(emb1.layer_norm, bert_student.LayerNorm)
        self.assertIsInstance(emb1.dropout, t.nn.Dropout)

    def test_embedding(self):
        """Test bert_student.Embedding for parity with nn.Embedding."""
        random_input = t.randint(0, 10, (2, 3))
        t.manual_seed(1157)
        emb1 = bert_student.Embedding(10, 5)
        t.manual_seed(1157)
        emb2 = nn.Embedding(10, 5)
        self.assert_tensors_close(emb1(random_input), emb2(random_input))

    def test_bert_embedding(self):
        """Test bert_student.BertEmbedding for parity with bert_reference.BertEmbedding."""
        config = BERT_CONFIG_STANDARD
        t.random.manual_seed(0)
        input_ids = t.randint(0, 2900, (2, 3))
        tt_ids = t.randint(0, 2, (2, 3))
        t.random.manual_seed(0)
        reference = bert_reference.BertEmbedding(config)
        reference.eval()
        t.random.manual_seed(0)
        yours = bert_student.BertEmbedding(
            config['vocab_size'], config['hidden_size'], config['max_position_embeddings'],
            config['type_vocab_size'], config['dropout'])
        yours.eval()
        self.assert_tensors_close(
            yours(input_ids=input_ids, token_type_ids=tt_ids),
            reference(input_ids=input_ids, token_type_ids=tt_ids)
        )


class TestBertAttention(MLTest):
    """Test multi-headed self-attention functionality."""

    @ patch('torch.nn.MultiheadAttention.forward')
    def test_no_cheating(self, patched_attention):
        """Test that the student doesn't call the PyTorch version."""
        student = bert_student.MultiHeadedSelfAttention(
            hidden_size=768, num_heads=12)
        input_activations = t.rand((2, 3, 768))
        student(input_activations)
        patched_attention.assert_not_called()

    def test_bert_attention_single_head(self):
        """
        Test bert_student.MultiHeadedSelfAttention for parity with
        bert_reference.SelfAttentionLayer (num_heads = 1).
        """
        config = BERT_CONFIG_NO_DROPOUT

        t.random.manual_seed(0)
        reference = bert_reference.SelfAttentionLayer(config)
        reference.eval()
        t.random.manual_seed(0)
        student = bert_student.MultiHeadedSelfAttention(
            hidden_size=config["hidden_size"], num_heads=config["num_heads"])
        student.eval()

        input_activations = t.rand((2, 3, 768))
        self.assert_tensors_close(student(input_activations), reference(input_activations))

    def test_bert_attention_multi_head(self):
        """
        Test bert_student.MultiHeadedSelfAttention for parity with
        bert_reference.SelfAttentionLayer (num_heads = 12).
        """
        config = BERT_CONFIG_NO_DROPOUT

        t.random.manual_seed(0)
        reference = bert_reference.SelfAttentionLayer(config)
        reference.eval()
        t.random.manual_seed(0)
        student = bert_student.MultiHeadedSelfAttention(
            hidden_size=config["hidden_size"], num_heads=config["num_heads"])
        student.eval()

        input_activations = t.rand((2, 3, 768))
        self.assert_tensors_close(student(input_activations), reference(input_activations))


class TestGELU(MLTest):
    """Test GELU functionality."""

    @patch('torch.nn.functional.gelu')
    def test_no_cheating(self, patched_function):
        """Test that the student doesn't call the PyTorch version."""
        gelu = bert_student.GELU()
        random_input = t.randint(0, 10, (2, 3))
        gelu(random_input)
        patched_function.assert_not_called()

    def test_gelu(self):
        """Test bert_student.GELU for parity with bert_reference.gelu and torch.nn.GELU."""
        student = bert_student.GELU()
        reference1 = bert_reference.gelu
        reference2 = t.nn.GELU()
        t.random.manual_seed(0)
        input = t.randn(20, 30)
        self.assert_tensors_close(student(input), reference1(input))
        self.assert_tensors_close(student(input), reference2(input))


class TestBertMLP(MLTest):
    """Test BERT MLP layer functionality."""

    @patch('mlab_tfs.bert.bert_student.GELU.forward')
    def test_calls_user_gelu(self, patched_gelu):
        """Test that the user calls their own code."""
        hidden_size = 768
        intermediate_size = 4 * hidden_size
        student = bert_student.BertMLP(hidden_size, intermediate_size)
        input = t.randn(2, 3, hidden_size)
        patched_gelu.return_value = t.randn(2, 3, intermediate_size)
        student(input)
        patched_gelu.assert_called()

    def test_bert_mlp(self):
        """Test bert_student.BertMLP for parity with bert_reference.BertMLP."""
        hidden_size = 768
        intermediate_size = 4 * hidden_size

        t.random.manual_seed(0)
        student = bert_student.BertMLP(hidden_size, intermediate_size)
        t.random.manual_seed(0)
        reference = bert_reference.BertMLP(hidden_size, intermediate_size)

        t.random.manual_seed(0)
        input = t.randn(2, 3, hidden_size)
        self.assert_tensors_close(student(input), reference(input))


class TestBertBlock(MLTest):
    """Test BERT single block functionality."""

    @patch('torch.nn.functional.layer_norm')
    @ patch('torch.nn.MultiheadAttention.forward')
    def test_no_cheating(self, patched_attention, patched_layer_norm):
        """Test that the student doesn't call the PyTorch version."""
        config = BERT_CONFIG_STANDARD
        bert_block = bert_student.BertBlock(
            intermediate_size=config["intermediate_size"],
            hidden_size=config["hidden_size"],
            num_heads=config["num_heads"],
            dropout=config["dropout"],
        )
        input_activations = t.rand((2, 3, 768))
        bert_block(input_activations)
        patched_attention.assert_not_called()
        patched_layer_norm.assert_not_called()

    def test_attribute_types(self):
        """Test the types of the module's attributes."""
        config = BERT_CONFIG_STANDARD
        bert_block = bert_student.BertBlock(
            intermediate_size=config["intermediate_size"],
            hidden_size=config["hidden_size"],
            num_heads=config["num_heads"],
            dropout=config["dropout"],
        )
        self.assertIsInstance(bert_block.attention, bert_student.MultiHeadedSelfAttention)
        self.assertIsInstance(bert_block.layernorm1, bert_student.LayerNorm)
        self.assertIsInstance(bert_block.mlp, bert_student.BertMLP)
        self.assertIsInstance(bert_block.layernorm2, bert_student.LayerNorm)
        self.assertIsInstance(bert_block.dropout, t.nn.Dropout)

    def test_bert_block_no_dropout(self):
        """Test bert_student.BertBlock for parity with bert_reference.BertBlock in eval mode."""
        config = BERT_CONFIG_STANDARD
        t.random.manual_seed(0)
        reference = bert_reference.BertBlock(config)
        reference.eval()
        t.random.manual_seed(0)
        student = bert_student.BertBlock(
            intermediate_size=config["intermediate_size"],
            hidden_size=config["hidden_size"],
            num_heads=config["num_heads"],
            dropout=config["dropout"],
        )
        student.eval()

        t.random.manual_seed(0)
        input_activations = t.rand((2, 3, 768))

        self.assert_tensors_close(student(input_activations), reference(input_activations))

    @unittest.expectedFailure
    def test_bert_block_with_dropout(self):
        """
        Test bert_student.BertBlock for parity with bert_reference.BertBlock in train mode.

        Note: Dropout makes this weird, so marking as an expectedFailure. Futher refinement needed.
        """
        config = BERT_CONFIG_STANDARD

        t.random.manual_seed(0)
        input_activations = t.rand((2, 3, 768))

        t.random.manual_seed(0)
        reference = bert_reference.BertBlock(config)
        reference.train()
        t.random.manual_seed(0)
        reference_output = reference(input_activations)

        t.random.manual_seed(0)
        input_activations = t.rand((2, 3, 768))

        t.random.manual_seed(0)
        student = bert_student.BertBlock(
            intermediate_size=config["intermediate_size"],
            hidden_size=config["hidden_size"],
            num_heads=config["num_heads"],
            dropout=config["dropout"],
        )
        student.train()
        t.random.manual_seed(0)
        student_output = student(input_activations)

        self.assert_tensors_close(student_output, reference_output)


class TestBertEndToEnd(MLTest):
    """Involves loading pretrained weights."""

    def test_bert_logits(self):
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
        reference = bert_reference.Bert(config)
        reference.eval()
        t.random.manual_seed(0)
        theirs = bert_student.Bert(**config)
        theirs.eval()
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            "bert-base-cased")
        input_ids = tokenizer("hello there", return_tensors="pt")["input_ids"]
        self.assert_tensors_close(
            theirs(input_ids=input_ids),
            reference(input_ids=input_ids).logits
        )

    def test_bert_classification(self):
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
        reference = bert_reference.Bert(config)
        reference.eval()
        t.random.manual_seed(0)
        theirs = bert_student.BertWithClassify(**config)
        theirs.eval()
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            "bert-base-cased")
        input_ids = tokenizer("hello there", return_tensors="pt")["input_ids"]
        logits, classifs = theirs(input_ids=input_ids)
        self.assert_tensors_close(
            logits,
            reference(input_ids=input_ids).logits,
        )

        self.assert_tensors_close(
            classifs,
            reference(input_ids=input_ids).classification,
        )

    def test_same_output_with_pretrained_weights(self):
        my_bert = bert_student.Bert(
            vocab_size=28996, hidden_size=768, max_position_embeddings=512,
            type_vocab_size=2, dropout=0.1, intermediate_size=3072,
            num_heads=12, num_layers=12
        )
        pretrained_bert = get_pretrained_bert()
        mapped_params = {mapkey(k): v for k, v in pretrained_bert.state_dict().items()
                         if not k.startswith('classification_head')}
        my_bert.load_state_dict(mapped_params)
        tol = 1e-4
        vocab_size = pretrained_bert.embedding.token_embedding.weight.shape[0]
        input_ids = t.randint(0, vocab_size, (10, 20))
        self.assert_tensors_close(
            my_bert.eval()(input_ids),
            pretrained_bert.eval()(input_ids).logits,
            tol=tol,
        )


if __name__ == '__main__':
    unittest.main()
