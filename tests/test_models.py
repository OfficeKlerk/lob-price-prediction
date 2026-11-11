import pytest
import torch

from lob.data import N_FEATURES
from lob.models import MODEL_NAMES, N_CLASSES, build_model, fit_normalizer

# Window length used by all tests in this file.
WINDOW = 100


# Test fixture: random input data.
def random_books(*shape: int) -> torch.Tensor:
    """Random positive 'order book' values of the given leading shape."""
    return torch.rand(*shape, N_FEATURES, generator=torch.Generator().manual_seed(0)) + 0.1


# Test fixture: a freshly initialised model with fitted normalisation.
def make_model(name: str) -> torch.nn.Module:
    torch.manual_seed(0)
    mean, std = fit_normalizer(random_books(500))
    return build_model(name, WINDOW, N_FEATURES, mean, std).eval()


# Every model honours the (batch, window, features) -> (batch, 3) contract.
@pytest.mark.parametrize("name", MODEL_NAMES)
def test_models_map_windows_to_class_logits(name):
    logits = make_model(name)(random_books(4, WINDOW))
    assert logits.shape == (4, N_CLASSES)
    assert torch.isfinite(logits).all()


# Mid-relative normalisation makes models invariant to the price level.
@pytest.mark.parametrize("name", MODEL_NAMES)
def test_models_ignore_the_absolute_price_level(name):
    """Shifting every price by a constant must not change the prediction."""
    model = make_model(name)
    x = random_books(4, WINDOW)
    shifted = x.clone()
    shifted[:, :, 0::2] += 5.0  # prices are the even feature indices

    torch.testing.assert_close(model(x), model(shifted), atol=1e-2, rtol=0)


# Unknown model names are an error.
def test_build_model_rejects_unknown_name():
    with pytest.raises(ValueError):
        build_model(
            "transformer", WINDOW, N_FEATURES, torch.zeros(N_FEATURES), torch.ones(N_FEATURES)
        )


# Normalisation statistics have the right shape and are mid-centred.
def test_fit_normalizer_centres_prices_on_the_mid():
    x = random_books(1000)
    mean, std = fit_normalizer(x)

    assert mean.shape == std.shape == (N_FEATURES,)
    assert (std > 0).all()
    # Best ask and best bid are symmetric around the mid, so their means cancel.
    assert torch.isclose(mean[0] + mean[2], torch.tensor(0.0), atol=1e-5)
