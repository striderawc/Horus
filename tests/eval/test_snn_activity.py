"""Unit tests for SNN activity metrics."""

from __future__ import annotations

import torch

from horus.eval.snn_activity import summarize_activity, summarize_layer
from horus.models.snn import LAYER_NAMES, LayerActivity, SNNActivityTrace


def _layer(spikes: torch.Tensor, membrane: torch.Tensor | None = None) -> LayerActivity:
    if membrane is None:
        membrane = torch.zeros_like(spikes)
    return LayerActivity(spikes=spikes, membrane=membrane)


def test_summarize_layer_all_silent():
    # (B=2, T=3, C=1, H=2, W=2) — 4 neurons, never fire
    spikes = torch.zeros(2, 3, 1, 2, 2)
    membrane = torch.full_like(spikes, 0.5)
    stats = summarize_layer(_layer(spikes, membrane), hyperactive_threshold=0.9)
    assert stats["mean_membrane"] == 0.5
    assert stats["spike_frac"] == 0.0
    assert stats["spikes_per_neuron_per_timestep"] == 0.0
    assert stats["silent_frac"] == 1.0
    assert stats["hyperactive_frac"] == 0.0


def test_summarize_layer_all_hyperactive():
    # Every neuron spikes every timestep
    spikes = torch.ones(2, 3, 1, 2, 2)
    membrane = torch.ones_like(spikes) * 1.2
    stats = summarize_layer(_layer(spikes, membrane), hyperactive_threshold=0.9)
    assert abs(stats["mean_membrane"] - 1.2) < 1e-6
    assert stats["spike_frac"] == 1.0
    assert stats["spikes_per_neuron_per_timestep"] == 1.0
    assert stats["silent_frac"] == 0.0
    assert stats["hyperactive_frac"] == 1.0


def test_summarize_layer_partial_activity():
    # B=1, T=3, C=1, H=1, W=2 — left silent, right fires once
    spikes = torch.zeros(1, 3, 1, 1, 2)
    spikes[0, 0, 0, 0, 1] = 1.0
    stats = summarize_layer(_layer(spikes), hyperactive_threshold=0.9)
    assert abs(stats["spike_frac"] - 0.5) < 1e-6
    assert abs(stats["silent_frac"] - 0.5) < 1e-6
    assert abs(stats["spikes_per_neuron_per_timestep"] - (1.0 / 6.0)) < 1e-6
    assert stats["hyperactive_frac"] == 0.0


def test_summarize_activity_all_layers():
    zeros = torch.zeros(1, 3, 2, 4, 4)
    ones = torch.ones(1, 3, 2, 4, 4)
    layers = {name: _layer(zeros) for name in LAYER_NAMES}
    layers["lif2"] = _layer(ones)
    trace = SNNActivityTrace(layers=layers)
    stats = summarize_activity(trace)
    assert set(stats) == set(LAYER_NAMES)
    assert stats["lif1"]["silent_frac"] == 1.0
    assert stats["lif2"]["hyperactive_frac"] == 1.0
