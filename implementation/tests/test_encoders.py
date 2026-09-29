import pytest
import torch

from models.encoder_factory import get_encoder


def test_unknown_model_raises(sample_config):
    with pytest.raises(ValueError):
        get_encoder("resnet", sample_config)


def test_encoder_video_shapes_contract():
    class ContractEncoder:
        def encode_video(self, videos):
            assert videos.dim() in (4, 5)
            return torch.nn.functional.normalize(videos.flatten(1).float(), dim=1)

        def encode_text(self, texts):
            return torch.ones(len(texts), 8)

        def get_embedding_dim(self):
            return 8

    encoder = ContractEncoder()
    single = encoder.encode_video(torch.randn(4, 3, 32, 32))
    batched = encoder.encode_video(torch.randn(2, 4, 3, 32, 32))
    assert single.shape[0] == 1 and batched.shape[0] == 2
    norms = batched.norm(dim=1)
    assert torch.allclose(norms, torch.ones(2), atol=1e-4)
