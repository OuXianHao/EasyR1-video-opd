"""Toy checks for the sign and stop-gradient boundaries of Vanilla OPD."""

import math

import pytest
import torch

from verl.trainer.config import PPOConfig
from verl.trainer.opd import compute_opd_loss


def _one_step(student_prob: float, teacher_prob: float):
    logits = torch.nn.Parameter(torch.tensor([math.log(student_prob), math.log1p(-student_prob)]))
    optimizer = torch.optim.SGD([logits], lr=0.1)
    old = torch.log_softmax(logits.detach(), dim=0)[0].view(1, 1)
    teacher = torch.tensor([[math.log(teacher_prob)]])
    before = torch.softmax(logits.detach(), dim=0)[0].item()
    loss, reward, ratio = compute_opd_loss(
        torch.log_softmax(logits, dim=0)[0].view(1, 1), old, teacher, torch.ones(1, 1)
    )
    loss.backward()
    gradient = logits.grad.detach().clone()
    optimizer.step()
    after = torch.softmax(logits.detach(), dim=0)[0].item()
    return before, after, reward.item(), ratio.item(), gradient


def test_teacher_more_likely_increases_student_probability():
    before, after, reward, ratio, _ = _one_step(0.4, 0.8)
    assert reward > 0 and ratio == pytest.approx(1)
    assert after > before


def test_teacher_less_likely_decreases_student_probability():
    before, after, reward, ratio, _ = _one_step(0.6, 0.2)
    assert reward < 0 and ratio == pytest.approx(1)
    assert after < before


def test_equal_policies_have_zero_update():
    before, after, reward, _, gradient = _one_step(0.4, 0.4)
    assert reward == pytest.approx(0, abs=1e-7)
    assert gradient.abs().max().item() < 1e-7
    assert after == pytest.approx(before, abs=1e-7)


def test_masked_token_has_no_loss_or_gradient():
    logits = torch.nn.Parameter(torch.tensor([[0.0, 0.0], [0.0, 0.0]]))
    sampled = torch.log_softmax(logits, dim=-1)[:, 0].view(1, 2)
    old = sampled.detach().clone()
    teacher = torch.log(torch.tensor([[0.9, 0.01]]))
    loss, _, _ = compute_opd_loss(sampled, old, teacher, torch.tensor([[1, 0]]))
    loss.backward()
    assert logits.grad[0].abs().sum().item() > 0
    assert torch.equal(logits.grad[1], torch.zeros(2))


def test_autograd_matches_analytic_and_finite_difference_gradient():
    logits = torch.nn.Parameter(torch.tensor([0.2, -0.3], dtype=torch.float64))
    old = torch.log_softmax(logits.detach(), dim=0)[0].view(1, 1)
    teacher = torch.tensor([[math.log(0.8)]], dtype=torch.float64, requires_grad=True)
    mask = torch.ones(1, 1)
    current = torch.log_softmax(logits, dim=0)[0].view(1, 1)
    loss, reward, _ = compute_opd_loss(current, old, teacher, mask)
    actual = torch.autograd.grad(loss, logits, retain_graph=True)[0]
    expected = -reward.item() * torch.autograd.grad(current, logits)[0]
    assert torch.allclose(actual, expected, atol=1e-12)
    assert not reward.requires_grad

    eps = 1e-5
    finite_difference = []
    for index in range(2):
        shifted = logits.detach().clone()
        shifted[index] += eps
        plus = compute_opd_loss(torch.log_softmax(shifted, dim=0)[0].view(1, 1), old, teacher, mask)[0]
        shifted[index] -= 2 * eps
        minus = compute_opd_loss(torch.log_softmax(shifted, dim=0)[0].view(1, 1), old, teacher, mask)[0]
        finite_difference.append((plus - minus) / (2 * eps))
    assert torch.allclose(actual, torch.stack(finite_difference), atol=1e-10)


def test_opd_config_is_isolated_from_default_grpo():
    config = PPOConfig()
    config.post_init()
    assert not config.algorithm.opd.enabled
    assert not config.worker.actor.opd_enabled
    assert not config.algorithm.disable_kl

    config = PPOConfig()
    config.algorithm.opd.enabled = True
    config.worker.teacher.model.model_path = "/some/local/teacher"
    config.worker.rollout.n = 1
    config.worker.actor.global_batch_size = config.data.rollout_batch_size
    config.post_init()
    assert config.algorithm.disable_kl
    assert not config.algorithm.use_kl_loss
    assert config.worker.actor.opd_enabled
