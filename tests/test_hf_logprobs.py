import pytest

from post_training_rl.hf_logprobs import compute_hf_completion_logps


def test_hf_completion_logps_is_left_as_practice_checkpoint():
    try:
        compute_hf_completion_logps(None, None, ["1 + 2 ="], [" 3"])
    except NotImplementedError as error:
        pytest.skip(str(error))
