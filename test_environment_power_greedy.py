"""以最低通訊功率為目標的 greedy Environment 測試。"""

import numpy as np

from Environment import Environment
from test_environment_greedy import check_transition


def power_greedy_targets(env):
    """為每架 UAV 選擇所需發射功率最低的未重複裝置。"""
    targets = np.full(env.U, env.M, dtype=np.int64)
    available_devices = list(range(env.M))

    for uav in range(env.U):
        if not available_devices:
            break

        powers = np.asarray(
            [
                env.Power_Calc(device, env.uav_locations[uav])
                for device in available_devices
            ],
            dtype=np.float64,
        )
        selected_index = int(np.argmin(powers))
        targets[uav] = available_devices.pop(selected_index)

    return targets


def power_greedy_move_actions(env):
    """朝目前功率需求最低的裝置移動。"""
    targets = power_greedy_targets(env)
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


def check_power_greedy_choice(env, service_actions):
    """確認每個服務目標都是依 UAV 順序可選裝置中的最低功率者。"""
    available_devices = list(range(env.M))

    for uav, selected_device in enumerate(service_actions):
        if not available_devices:
            assert selected_device == env.M
            continue

        powers = [
            env.Power_Calc(device, env.uav_locations[uav])
            for device in available_devices
        ]
        expected_device = available_devices[int(np.argmin(powers))]
        assert selected_device == expected_device
        available_devices.remove(int(selected_device))


def run_power_greedy_episode(max_steps=100, seed=7):
    """執行純低功耗 greedy episode，並回傳環境統計。"""
    np.random.seed(seed)
    env = Environment(max_steps=max_steps)
    env.reset()

    total_power = 0.0
    total_distance = 0.0
    total_risk = 0
    mean_aoi_history = []
    service_count = np.zeros(env.M, dtype=np.int32)

    while not env.terminated:
        old_locations = env.uav_locations.copy()
        old_aoi = env.aoi.copy()
        old_step_count = env.step_count

        # 第一階段：朝目前通訊功率最低的裝置移動。
        move_actions = power_greedy_move_actions(env)
        intermediate_state, move_info = env.move_step(move_actions)
        assert env.step_count == old_step_count

        # 第二階段：依移動後位置重新選擇最低功率服務目標。
        service_actions = power_greedy_targets(env)
        check_power_greedy_choice(env, service_actions)
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
        total_distance += float(np.sum(move_info["distance"]))
        total_risk += int(np.sum(move_info["risk_triggered"]))
        mean_aoi_history.append(service_info["mean_aoi"])

        for device in service_actions:
            if device < env.M:
                service_count[device] += 1

    assert env.step_count == max_steps
    assert np.sum(service_count) == max_steps * min(env.U, env.M)

    return {
        "steps": env.step_count,
        "terminated": env.terminated,
        "average_aoi": float(np.mean(mean_aoi_history)),
        "final_max_aoi": int(np.max(env.aoi)),
        "total_power": total_power,
        "total_distance": total_distance,
        "risk_events": total_risk,
        "devices_served": int(np.count_nonzero(service_count)),
        "service_count": service_count.tolist(),
    }


if __name__ == "__main__":
    result = run_power_greedy_episode()

    print("Environment power-greedy test passed.")
    for name, value in result.items():
        print(f"{name}: {value}")
