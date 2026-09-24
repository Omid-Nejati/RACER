import pathlib
import sys
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from saliency_qwen import parse_layer_spec


def test_layer_spec():
    assert parse_layer_spec("6,7,8", 20) == [6, 7, 8]
    assert parse_layer_spec("-1", 20) == [19]
    assert parse_layer_spec("all", 3) == [0, 1, 2]


def test_attention_gain_equivalence():
    # Adding log(gamma) to logits is equivalent to multiplying the selected
    # softmax numerator by gamma and renormalizing.
    logits = torch.tensor([0.2, -0.1, 0.5])
    gamma = 1.2
    biased = logits.clone()
    biased[1] += torch.log(torch.tensor(gamma))
    p1 = torch.softmax(biased, dim=-1)

    p = torch.softmax(logits, dim=-1)
    scaled = p.clone()
    scaled[1] *= gamma
    p2 = scaled / scaled.sum()
    assert torch.allclose(p1, p2, atol=1e-6)


if __name__ == "__main__":
    test_layer_spec()
    test_attention_gain_equivalence()
    print("portable logic tests: OK")
