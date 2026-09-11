"""Regression tests for training-only candidate retention and full-search seeds."""
import dataclasses

import pytest

from afterimage.runtime import h65_planner as h65
from test_h65_planner import _manifest, _trace


def _params(**overrides):
    return dict(dict(vram_budget_gb=.20, ram_budget_gb=.03, h2d_gbps=1.,
                     decode_slice_elems=1, search_iterations=0,
                     require_causal_trace=False, require_live_validation=False,
                     enable_compressed_ram=False), **overrides)


def _traces(heldout=None):
    return list(map(_trace, [dict(a=.1, b=.2, c=4., d=3.),
                            dict(a=4., b=.2, c=.1, d=3.),
                            heldout or dict(a=.1, b=.2, c=4., d=3.)]))


def _names(plan):
    return {key: option.name for key, option in plan.choices.items()}


def test_retains_safe_candidate_without_restricting_aggregate_search():
    result = h65.optimize_h65_plan(_manifest(), _traces(), **_params())
    report = result.report
    assert report.training_safe_retention_applied
    assert report.aggregate_winner_objective_s < report.candidate_objective_s
    assert min(report.training_improvements + report.validation_improvements) >= 0
    assert not report.fallback_to_control
    assert report.schema_version == 6


@pytest.mark.parametrize('full', [False, True])
def test_holdout_can_reject_but_cannot_select_another_candidate(full):
    params = _params(enable_compressed_ram=full)
    original = h65.optimize_h65_plan(_manifest(), _traces(), **params)
    changed = h65.optimize_h65_plan(
        _manifest(), _traces(dict(a=10., b=100., c=.01, d=.01)), **params)
    assert _names(original.candidate_plan) == _names(changed.candidate_plan)
    assert original.report.training_improvements == changed.report.training_improvements
    assert changed.report.fallback_to_control
    assert min(changed.report.validation_improvements) < 0


def test_full_automatic_seed_never_reads_heldout_during_seed_search(monkeypatch):
    traces = _traces()
    seen = []
    replay = h65._ReplayScorer.replay_samples

    def observe(self, choices):
        if len(self.traces) == 1 and self.traces[0] is traces[-1]:
            seen.append(tuple((key, option.name) for key, option in choices.items()))
        return replay(self, choices)

    monkeypatch.setattr(h65._ReplayScorer, 'replay_samples', observe)
    result = h65.optimize_h65_plan(
        _manifest(), traces, **_params(enable_compressed_ram=True))
    assert result.report.seeded_with_placement_search
    # Exactly one control and one selected candidate, AFTER training search.
    assert len(seen) == 2


@pytest.mark.parametrize('explicit', [False, True])
def test_full_retains_compatible_placement_incumbent(explicit):
    manifest, traces = _manifest(), _traces()
    placement = h65.optimize_h65_plan(manifest, traces, **_params())
    params = _params(enable_compressed_ram=True)
    if explicit:
        params['extra_seed_choices'] = placement.candidate_plan.choices
    full = h65.optimize_h65_plan(manifest, traces, **params)
    assert full.report.candidate_objective_s <= placement.report.candidate_objective_s
    assert min(full.report.training_improvements) >= 0
    assert full.report.placement_seed_objective_s == pytest.approx(
        placement.report.candidate_objective_s)
    assert full.report.seeded_with_external_plan is explicit
    assert full.report.seeded_with_placement_search is not explicit


def test_all_initial_seeds_are_considered_for_training_safe_retention(monkeypatch):
    # A complete supplied seed is safe, but differs by four choices from the
    # aggregate winner. Disk-to-resident one-for-one moves cannot rediscover it.
    disk, vram, ram = 'compressed_disk', 'decoded_vram', 'decoded_ram'
    control = (vram, ram, disk, disk)
    safe = (disk, disk, vram, ram)
    unsafe = (disk, disk, ram, vram)
    samples = {
        control: (10., 20.), safe: (9., 18.), unsafe: (11., 15.),
        (disk,) * 4: (40., 40.),
        (disk, disk, disk, vram): (20., 25.),
        (disk, disk, vram, disk): (28., 28.),
        (disk, disk, disk, ram): (29., 29.),
        (disk, disk, ram, disk): (25., 25.),
    }

    def replay(self, choices):
        key = tuple(choices[k].name for k in 'abcd')
        return samples.get(key, (30., 30.))[:len(self.traces)]

    def control_choices(manifest, tier, options):
        return {key: options[key][name] for key, name in zip('abcd', control)}

    monkeypatch.setattr(h65._ReplayScorer, 'replay_samples', replay)
    monkeypatch.setattr(h65, '_choices_from_tier_plan', control_choices)
    result = h65.optimize_h65_plan(
        _manifest(), _traces(),
        **_params(extra_seed_choices=dict(zip('abcd', safe))))
    assert tuple(_names(result.candidate_plan).values()) == safe
    assert result.report.training_safe_retention_applied
    assert result.report.aggregate_winner_objective_s == 15.
    assert result.report.candidate_objective_s == 18.


@pytest.mark.parametrize('bad', [{}, {'a': 'decoded_vram'},
                                dict(a='unknown', b='decoded_ram',
                                     c='compressed_disk', d='compressed_disk'),
                                dict(a='compressed_ram', b='decoded_ram',
                                     c='compressed_disk', d='compressed_disk')])
def test_invalid_seed_is_rejected_instead_of_silently_becoming_traffic(bad):
    with pytest.raises(ValueError, match='extra seed'):
        h65.optimize_h65_plan(_manifest(), _traces(), **_params(extra_seed_choices=bad))


def test_seed_cannot_forge_memory_costs():
    forged = {key: h65.RepresentationOption(key, 'decoded_vram', vram_bytes=0)
              for key in 'abcd'}
    with pytest.raises(ValueError, match='memory limits'):
        h65.optimize_h65_plan(_manifest(), _traces(),
                             **_params(extra_seed_choices=forged))


def test_retention_does_not_bypass_live_validation():
    result = h65.optimize_h65_plan(
        _manifest(), _traces(), **_params(require_live_validation=True))
    assert result.report.training_safe_retention_applied
    assert result.report.fallback_to_control
    assert 'awaits 2 paired live validation blocks' in result.report.fallback_reason
    # Reporting additions remain serializable for old matrix consumers.
    assert dataclasses.asdict(result.report)['seeded_with_external_plan'] is False


def test_output_head_seed_is_pinned_before_both_fill_orders():
    disk = {
        'lm_head.weight': h65.RepresentationOption(
            'lm_head.weight', 'compressed_disk'),
        'layer': h65.RepresentationOption('layer', 'compressed_disk'),
    }
    head = h65.RepresentationOption(
        'lm_head.weight', 'decoded_vram', vram_bytes=10)
    layer = h65.RepresentationOption('layer', 'decoded_vram', vram_bytes=10)
    options = {
        'lm_head.weight': {
            'compressed_disk': disk['lm_head.weight'],
            'decoded_vram': head,
        },
        'layer': {'compressed_disk': disk['layer'], 'decoded_vram': layer},
    }
    seen = []

    def fill(seed, order):
        seen.append((seed['lm_head.weight'].name, order))
        return dict(seed)

    def feasible(choices):
        return sum(option.vram_bytes for option in choices.values()) <= 10

    seeds = h65._critical_endpoint_seeds(
        options, {'lm_head.weight': disk['lm_head.weight'], 'layer': layer},
        disk, lambda key, name: 0, fill, feasible)
    assert seen == [('decoded_vram', 'vram-first'),
                    ('decoded_vram', 'ram-first')]
    assert all(seed['lm_head.weight'].name == 'decoded_vram' for seed in seeds)
    assert all(seed['layer'].name == 'compressed_disk' for seed in seeds)


def test_output_head_seed_is_omitted_when_it_cannot_fit():
    disk = {'lm_head.weight': h65.RepresentationOption(
        'lm_head.weight', 'compressed_disk')}
    options = {'lm_head.weight': {
        'compressed_disk': disk['lm_head.weight'],
        'decoded_vram': h65.RepresentationOption(
            'lm_head.weight', 'decoded_vram', vram_bytes=10),
    }}
    assert h65._critical_endpoint_seeds(
        options, disk, disk, lambda key, name: 0,
        lambda seed, order: seed, lambda choices: False) == []
