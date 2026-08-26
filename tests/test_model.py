import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from parking_il_planner.config.model import ModelConfig  # noqa: E402
from parking_il_planner.models.planner_model import APAPlannerImitationModel  # noqa: E402


def test_model_forward_shape() -> None:
    config = ModelConfig(
        resnet_out_channels=64,
        transformer_nhead=4,
        transformer_num_layers=1,
        transformer_dim_feedforward=64,
    )
    model = APAPlannerImitationModel(config).eval()
    with torch.no_grad():
        logits = model(
            torch.zeros(1, 3, 384, 384),
            state_vector=torch.zeros(1, 6),
            action_history=torch.zeros(1, 14),
        )
    assert logits.shape == (1, 7)
