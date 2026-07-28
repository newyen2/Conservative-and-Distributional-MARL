"""以簡單 greedy 策略檢查 Environment 的兩階段 step。"""

import numpy as np

from Environment import Environment


def greedy_targets(env):
    """依序為 UAV 選擇 AOI 最大、同分時距離最近的未重複裝置。"""
    targets = np.full(env.U, env.M, dtype=np.int64)
    available_devices = list(range(env.M))

    for uav in range(env.U):
        if not available_devices:
            break

        available = np.asarray(available_devices, dtype=np.int64)
        oldest_aoi = np.max(env.aoi[available])
        oldest_devices = available[env.aoi[available] == oldest_aoi]

        distances = np.linalg.norm(
            env.device_coord[oldest_devices] - env.uav_locations[uav],
            axis=1,
        )
        target = int(oldest_devices[np.argmin(distances)])
        targets[uav] = target
        available_devices.remove(target)

    return targets


def greedy_move_actions(env):
    """朝 greedy 目標直線移動，單步距離不超過 max_movement。"""
    targets = greedy_targets(env)
    move_actions = np.zeros((env.U, 2), dtype=np.float32)

    for uav, device in enumerate(targets):
        if device == env.M:
            continue

        direction = env.device_coord[device] - env.uav_locations[uav]
        distance = np.linalg.norm(direction)
        if distance > 0:
            move_actions[uav] = (
                direction / distance * min(distance, env.max_movement)
            )

    return move_actions


def check_transition(
    env,
    old_locations,
    old_aoi,
    old_step_count,
    intermediate_state,
    move_info,
    service_actions,
    next_state,
    terminated,
    service_info,
):
    """檢查一次 move_step 與 service_step 是否符合環境契約。"""
    expected_state_size = env.U * 2 + env.M + env.M * 2

    # 檢查 move_step 產生的位置與中間狀態。
    assert intermediate_state.shape == (expected_state_size,)
    assert np.all(move_info["distance"] <= env.max_movement + 1e-6)
    assert np.all(env.uav_locations >= env.map_low)
    assert np.all(env.uav_locations <= env.map_high)
    assert np.allclose(
        env.uav_locations - old_locations,
        move_info["movement"],
    )

    # service_step 應推進一次時間，AOI 增加後再重置被服務裝置。
    assert old_step_count + 1 == env.step_count
    expected_aoi = np.minimum(old_aoi + 1, env.aoi_max)
    for device in service_actions:
        if 0 <= device < env.M:
            expected_aoi[device] = 1

    assert env.aoi.dtype == np.int32
    assert np.array_equal(env.aoi, expected_aoi)
    assert next_state.shape == (expected_state_size,)
    assert terminated == (env.step_count >= env.max_steps)

    # 有效服務動作的功率須符合 Power_Calc；M 則代表不服務、功率為 0。
    for uav, device in enumerate(service_actions):
        expected_power = (
            env.Power_Calc(int(device), env.uav_locations[uav])
            if 0 <= device < env.M
            else 0.0
        )
        assert np.isclose(service_info["power"][uav], expected_power)


def run_greedy_episode(max_steps=100, seed=7):
    """執行一個 greedy episode，並回傳可讀的環境統計。"""
    np.random.seed(seed)
    env = Environment(max_steps=max_steps)
    env.reset()

    total_power = 0.0
    total_risk = 0
    mean_aoi_history = []
    service_count = np.zeros(env.M, dtype=np.int32)

    while not env.terminated:
        old_locations = env.uav_locations.copy()
        old_aoi = env.aoi.copy()
        old_step_count = env.step_count

        # 第一階段：根據目前狀態朝高 AOI 裝置移動。
        move_actions = greedy_move_actions(env)
        intermediate_state, move_info = env.move_step(move_actions)
        assert env.step_count == old_step_count

        # 第二階段：根據移動後位置重新選擇高 AOI 服務目標。
        service_actions = greedy_targets(env)
        next_state, terminated, service_info = env.service_step(
            service_actions
        )

        check_transition(
            env=env,
            old_locations=old_locations,
            old_aoi=old_aoi,
            old_step_count=old_step_count,
            intermediate_state=intermediate_state,
            move_info=move_info,
            service_actions=service_actions,
            next_state=next_state,
            terminated=terminated,
            service_info=service_info,
        )

        total_power += service_info["total_power"]
        total_risk += int(np.sum(move_info["risk_triggered"]))
        mean_aoi_history.append(service_info["mean_aoi"])

        for device in service_actions:
            if device < env.M:
                service_count[device] += 1

    assert env.step_count == max_steps
    assert np.all(service_count > 0)

    return {
        "steps": env.step_count,
        "terminated": env.terminated,
        "average_aoi": float(np.mean(mean_aoi_history)),
        "final_max_aoi": int(np.max(env.aoi)),
        "total_power": total_power,
        "risk_events": total_risk,
        "service_count": service_count.tolist(),
    }


if __name__ == "__main__":
    result = run_greedy_episode()

    print("Environment greedy test passed.")
    for name, value in result.items():
        print(f"{name}: {value}")
