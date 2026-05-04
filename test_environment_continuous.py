"""
Simple smoke tests for the continuous/hybrid Environment.

Expected action format for each UAV:
    actions[u] = [selected_device, move_vector]

where:
    selected_device: int in [0, M]
        0 ~ M-1: select a device
        M: no selection
    move_vector: np.ndarray shape (2,)
        continuous movement vector [dx, dy]
        this test script samples it within the MAX_MOV circle

Usage:
    Put this file in the same folder as Environment.py, then run:
        python test_environment_continuous.py
"""

from __future__ import annotations

import importlib.util
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


# -----------------------------------------------------------------------------
# Load Environment.py dynamically
# -----------------------------------------------------------------------------

def load_environment_module() -> Any:
    """Load Environment module from the same folder as this test file."""
    here = Path(__file__).resolve().parent

    candidates = [
        here / "Environment.py",
        here / "Environment(2).py",
        here / "Environment(1).py",
    ]

    existing = [p for p in candidates if p.exists()]
    env_path = max(existing, key=lambda p: p.stat().st_mtime) if existing else None
    if env_path is None:
        raise FileNotFoundError(
            "Cannot find Environment.py. Put this test file in the same folder "
            "as your Environment.py file."
        )

    spec = importlib.util.spec_from_file_location("environment_under_test", env_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load module from {env_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    print(f"[INFO] Loaded environment module: {env_path.name}")
    return module


# -----------------------------------------------------------------------------
# Minimal config object required by Environment.__init__
# -----------------------------------------------------------------------------

@dataclass
class DummyConfig:
    M: int = 10
    U: int = 2
    nCell: int = 10
    energy_weight: float = 1.0
    penalty: float = 10.0
    prob_of_risk: float = 0.5


# -----------------------------------------------------------------------------
# Action sampler
# -----------------------------------------------------------------------------

def sample_move_in_circle(max_mov: float) -> np.ndarray:
    """
    Uniformly sample a 2D movement vector inside a circle of radius max_mov.
    This assumes movement magnitude is constrained outside the Environment.
    """
    theta = np.random.uniform(0.0, 2.0 * math.pi)
    radius = max_mov * math.sqrt(np.random.uniform(0.0, 1.0))

    return np.array(
        [radius * math.cos(theta), radius * math.sin(theta)],
        dtype=np.float32,
    )


def sample_actions(env: Any) -> list[list[Any]]:
    """
    Generate one hybrid action per UAV.

    Format:
        [selected_device, move_vector]
    """
    actions = []

    for _ in range(env.U):
        selected_device = random.randint(0, env.M)  # M means no selection
        move_vector = sample_move_in_circle(env.MAX_MOV)
        actions.append([selected_device, move_vector])

    return actions


# -----------------------------------------------------------------------------
# Assertions
# -----------------------------------------------------------------------------

def assert_state_valid(env: Any, state: np.ndarray, name: str) -> None:
    state = np.asarray(state)

    assert state.shape == (env.nObservation,), (
        f"{name}: state shape should be {(env.nObservation,)}, got {state.shape}"
    )

    uav_pos = state[: env.U * 2].reshape(env.U, 2)
    aoi = state[env.U * 2 : env.U * 2 + env.M]

    assert np.all(uav_pos >= env.L_map - 1e-6), (
        f"{name}: UAV position below map lower bound: {uav_pos}"
    )
    assert np.all(uav_pos <= env.H_map + 1e-6), (
        f"{name}: UAV position above map upper bound: {uav_pos}"
    )

    assert aoi.shape == (env.M,), f"{name}: AOI shape should be {(env.M,)}, got {aoi.shape}"
    assert np.all(aoi >= 1), f"{name}: AOI should be >= 1, got {aoi}"
    assert np.all(aoi <= env.AOI_max), f"{name}: AOI should be <= AOI_max, got {aoi}"

    assert np.all(np.isfinite(state)), f"{name}: state contains non-finite values: {state}"


def assert_device_coords_valid(env: Any) -> None:
    device_coord = np.asarray(env.device_coord)

    assert device_coord.shape == (env.M, 2), (
        f"device_coord shape should be {(env.M, 2)}, got {device_coord.shape}"
    )
    assert np.all(device_coord >= env.L_map - 1e-6), (
        f"device_coord below map lower bound: {device_coord}"
    )
    assert np.all(device_coord <= env.H_map + 1e-6), (
        f"device_coord above map upper bound: {device_coord}"
    )
    assert np.all(np.isfinite(device_coord)), (
        f"device_coord contains non-finite values: {device_coord}"
    )


def assert_step_outputs_valid(env: Any, next_state: np.ndarray, rewards: list[float], done: np.ndarray) -> None:
    assert_state_valid(env, next_state, "next_state")

    assert len(rewards) == env.U, f"rewards length should be {env.U}, got {len(rewards)}"
    assert np.all(np.isfinite(rewards)), f"rewards contain non-finite values: {rewards}"

    done = np.asarray(done)
    assert done.shape == (env.U,), f"done shape should be {(env.U,)}, got {done.shape}"
    assert np.all(np.logical_or(done == 0, done == 1)), f"done should contain only 0/1, got {done}"


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------

def test_reset(env: Any) -> np.ndarray:
    print("[TEST] reset()")

    state = env.reset()

    assert_state_valid(env, state, "reset_state")
    assert_device_coords_valid(env)

    uav_pos = state[: env.U * 2].reshape(env.U, 2)
    for i, pos in enumerate(uav_pos):
        assert not env.is_coord_risky(pos), f"UAV {i} initialized inside risky region: {pos}"

    print("[PASS] reset()")
    return state


def test_one_step(env: Any, state: np.ndarray) -> np.ndarray:
    print("[TEST] one step with random continuous movement")

    actions = sample_actions(env)
    next_state, rewards, done = env.step(state, actions)

    assert_step_outputs_valid(env, next_state, rewards, done)

    print(f"[INFO] actions = {actions}")
    print(f"[INFO] rewards = {rewards}")
    print(f"[INFO] done = {done}")
    print("[PASS] one step")

    return next_state


def test_rollout_until_done(env: Any) -> None:
    print("[TEST] rollout until done")

    state = env.reset()
    assert_state_valid(env, state, "rollout_initial_state")

    last_rewards = None
    last_done = None

    for t in range(env.max_steps + 5):
        actions = sample_actions(env)
        next_state, rewards, done = env.step(state, actions)

        assert_step_outputs_valid(env, next_state, rewards, done)

        state = next_state
        last_rewards = rewards
        last_done = done

        if env.DONE:
            print(f"[INFO] Environment terminated at rollout step {t + 1}")
            break

    assert env.DONE == 1, "Environment should terminate after max_steps"
    assert np.all(last_done == 1), f"All UAVs should be done, got {last_done}"

    print(f"[INFO] final rewards = {last_rewards}")
    print(f"[INFO] final risk_count = {env.risk_count}")
    print("[PASS] rollout until done")


def test_boundary_clip(env: Any) -> None:
    print("[TEST] boundary clipping")

    # Directly test Update_Location with intentionally oversized movement.
    # This does not test MAX_MOV; it only checks map boundary clipping.
    loc = np.array([9.5, 9.5], dtype=np.float32)
    move = np.array([100.0, 100.0], dtype=np.float32)

    new_loc = env.Update_Location(loc, move)

    assert np.all(new_loc <= env.H_map + 1e-6), f"boundary upper clip failed: {new_loc}"
    assert np.all(new_loc >= env.L_map - 1e-6), f"boundary lower clip failed: {new_loc}"
    assert np.allclose(new_loc, env.H_map), f"expected clipped location {env.H_map}, got {new_loc}"

    print("[PASS] boundary clipping")


def main() -> None:
    np.random.seed(515)
    random.seed(15616)

    env_module = load_environment_module()
    Environment = env_module.Environment
    generate_unique_coords = env_module.generate_unique_coords

    config = DummyConfig(M=10, U=2, nCell=10, energy_weight=1.0, penalty=10.0, prob_of_risk=0.5)

    # Rectangular risky regions: [xmin, ymin, xmax, ymax]
    risky_region = [
        [2.0, 2.0, 3.0, 3.0],
        [6.0, 6.0, 7.0, 7.0],
    ]

    # Initial device_coord is overwritten by env.reset(), but Environment.__init__ still requires it.
    device_coord = generate_unique_coords(N=config.M)

    env = Environment(device_coord=device_coord, risky_region=risky_region, config=config)

    state = test_reset(env)
    state = test_one_step(env, state)
    test_boundary_clip(env)
    test_rollout_until_done(env)

    print("\nAll Environment smoke tests passed.")


if __name__ == "__main__":
    main()
