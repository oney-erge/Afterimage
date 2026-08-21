from afterimage.runtime.controllers import (
    KnapsackSemiBandit, LinearProfileBandit, ModelBasedProfileController,
    MPCPrefetchController, PIPrefetchController, PrefetchObservation,
    PageHinkley, SemiBanditItem,
)


def test_pi_prefetch_responds_to_starvation_and_stays_bounded():
    ctl = PIPrefetchController(1, 4, kp=3.0, ki=1.0)
    for _ in range(8):
        ctl.update(PrefetchObservation(False, wait_s=1.0, useful_bytes=10))
    assert 1 < ctl.choose_depth() <= 4


def test_mpc_prefetch_returns_feasible_depth():
    ctl = MPCPrefetchController(1, 6)
    ctl.update(PrefetchObservation(False, wait_s=0.5, useful_bytes=100_000_000,
                                   bandwidth_bytes_s=200_000_000))
    ctl.update_compute(0.1)
    assert 0 <= ctl.choose_depth() <= 6


def test_linear_profile_bandit_learns_contextual_rewards():
    bandit = LinearProfileBandit(["a", "b"], 2, alpha=0.1, baseline_profile="a")
    for _ in range(30):
        bandit.update("a", [1, 0], 1.0)
        bandit.update("b", [1, 0], 3.0)
    # Disable the conservative guard for the direct learning assertion.
    bandit._baseline_rewards.clear()
    assert bandit.choose([1, 0]) == "b"


def test_linear_profile_bandit_calibrates_each_arm_once():
    bandit = LinearProfileBandit(["base", "candidate"], 1,
                                 baseline_profile="base")
    assert bandit.choose([1]) == "base"
    bandit.update("base", [1], 1.0)
    assert bandit.choose([1]) == "candidate"


def test_semi_bandit_selects_high_value_items_under_budget():
    learner = KnapsackSemiBandit([
        SemiBanditItem("a", 10, 1.0, 10),
        SemiBanditItem("b", 10, 3.0, 10),
        SemiBanditItem("c", 20, 2.0, 10),
    ], exploration=0.0, size_quantum=10)
    assert learner.select(10) == ["b"]


def test_model_based_controller_shadow_mode_does_not_apply_recommendation():
    def simulator(state, profile):
        return state + 1, 2.0 if profile == "fast" else 1.0

    ctl = ModelBasedProfileController(["base", "fast"], simulator,
                                      baseline_profile="base", shadow=True)
    assert ctl.choose(0) == "base"
    assert ctl.last_recommendation == "fast"


def test_page_hinkley_detects_large_persistent_shift():
    detector = PageHinkley(threshold=1.0, delta=0.0)
    assert not any(detector.update(0.0) for _ in range(20))
    assert any(detector.update(5.0) for _ in range(20))
