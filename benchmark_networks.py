"""階層式網路的自包含 toy-game benchmark。

執行方式：
    python benchmark_networks.py
    python benchmark_networks.py --quick
    python benchmark_networks.py --device cuda

本 benchmark 不取代正式 UAV Environment 實驗。它只回答兩個較基礎的問題：
1. High-level DDQN 能否在 GoalChoiceGame 中學會選擇價值最高的目標。
2. MoveActor、SelectActor 與 twin joint Critic 能否在 MoveAndServeGame 中
   學會向指定目標移動並選擇正確服務。

兩個遊戲皆為單步 contextual game，因此不需要 Gym 等額外依賴。
"""

import argparse
import json
import random
import time
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Normal

from networks import DDQN, SAC_Continuous_Actor, SAC_Critic, SAC_Discrete_Actor


@dataclass
class BenchmarkResult:
    device: str
    seed: int
    goal_accuracy: float
    goal_average_regret: float
    move_efficiency: float
    service_accuracy: float
    critic_mse: float
    elapsed_seconds: float
    passed: bool


def set_seed(seed):
    """固定所有隨機來源，讓不同修改版本可重複比較。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_name):
    """解析 CLI device，且在要求 CUDA 但不可用時明確報錯。"""
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is unavailable.")
    return torch.device(device_name)


def benchmark_goal_choice(device, steps, batch_size, hidden_size):
    """以 terminal DQN target 訓練 DDQN 選擇當下分數最高的裝置。"""
    number_of_goals = 4
    network = DDQN(number_of_goals, number_of_goals, hidden_size).to(device)
    optimizer = torch.optim.Adam(network.parameters(), lr=2e-3)

    for step in range(steps):
        states = torch.empty(batch_size, number_of_goals, device=device)
        states.uniform_(-1.0, 1.0)

        with torch.no_grad():
            greedy_actions = network(states).argmax(dim=-1)
            random_actions = torch.randint(
                number_of_goals,
                (batch_size,),
                device=device,
            )
            progress = step / max(steps - 1, 1)
            epsilon = 1.0 + progress * (0.05 - 1.0)
            explore = torch.rand(batch_size, device=device) < epsilon
            actions = torch.where(explore, random_actions, greedy_actions)
            rewards = states.gather(1, actions.unsqueeze(1))

        predicted_q = network(states).gather(1, actions.unsqueeze(1))
        loss = F.smooth_l1_loss(predicted_q, rewards)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        states = torch.empty(4096, number_of_goals, device=device)
        states.uniform_(-1.0, 1.0)
        selected = network(states).argmax(dim=-1)
        optimal = states.argmax(dim=-1)
        selected_value = states.gather(1, selected.unsqueeze(1)).squeeze(1)
        optimal_value = states.max(dim=-1).values

        accuracy = (selected == optimal).float().mean().item()
        average_regret = (optimal_value - selected_value).mean().item()

    return accuracy, average_regret


def make_low_level_batch(batch_size, device, max_move=0.5):
    """產生 MoveAndServeGame 的隨機 transition，供 Critic 學習真實 reward。"""
    device_positions = torch.tensor([-0.75, 0.75], device=device)
    position = torch.empty(batch_size, 1, device=device).uniform_(-1.0, 1.0)
    fixed_devices = device_positions.unsqueeze(0).expand(batch_size, -1)
    states = torch.cat((position, fixed_devices), dim=-1)

    goal_index = torch.randint(2, (batch_size,), device=device)
    goals = F.one_hot(goal_index, num_classes=2).float()
    move_actions = torch.empty(batch_size, 1, device=device)
    move_actions.uniform_(-max_move, max_move)
    service_index = torch.randint(3, (batch_size,), device=device)
    service_onehot = F.one_hot(service_index, num_classes=3).float()

    moved_position = torch.clamp(position + move_actions, -1.0, 1.0)
    moved_states = torch.cat((moved_position, fixed_devices), dim=-1)
    target = device_positions[goal_index].unsqueeze(1)

    # 移動分數位於 [0, 1]；選對服務再獲得 1 分，不服務動作不會成功。
    movement_score = 1.0 - torch.abs(moved_position - target) / 2.0
    service_score = (service_index == goal_index).float().unsqueeze(1)
    rewards = movement_score + service_score

    return {
        "states": states,
        "goals": goals,
        "goal_index": goal_index,
        "move_actions": move_actions,
        "moved_states": moved_states,
        "service_index": service_index,
        "service_onehot": service_onehot,
        "rewards": rewards,
        "target": target,
    }


def sample_continuous_action(move_actor, actor_input, max_move):
    """執行 SAC reparameterization；動作限制與 Network 保持分離。"""
    mu, log_std = move_actor(actor_input)
    log_std = torch.clamp(log_std, -5.0, 2.0)
    distribution = Normal(mu, log_std.exp())
    raw_action = distribution.rsample()
    normalized_action = torch.tanh(raw_action)
    move_action = normalized_action * max_move

    log_prob = distribution.log_prob(raw_action)
    log_prob -= torch.log(1.0 - normalized_action.square() + 1e-6)
    log_prob -= torch.log(torch.as_tensor(max_move, device=actor_input.device))

    return move_action, log_prob.sum(dim=-1, keepdim=True)


def evaluate_critics(critic_1, critic_2, device):
    """在未參與更新的新樣本上計算 twin Critic MSE。"""
    batch = make_low_level_batch(4096, device)
    critic_input = torch.cat(
        (
            batch["states"],
            batch["goals"],
            batch["move_actions"],
            batch["service_onehot"],
        ),
        dim=-1,
    )
    predicted = torch.minimum(critic_1(critic_input), critic_2(critic_input))
    return F.mse_loss(predicted, batch["rewards"]).item()


def benchmark_move_and_serve(device, steps, batch_size, hidden_size):
    """訓練低層 twin Critic 及兩個透過同一 Q-value 更新的 Actor。"""
    state_size = 3
    goal_size = 2
    move_size = 1
    service_size = 3
    max_move = 0.5

    move_actor = SAC_Continuous_Actor(
        state_size + goal_size,
        move_size,
        hidden_size,
    ).to(device)
    select_actor = SAC_Discrete_Actor(
        state_size + goal_size,
        service_size,
        hidden_size,
    ).to(device)
    critic_input_size = state_size + goal_size + move_size + service_size
    critic_1 = SAC_Critic(critic_input_size, hidden_size).to(device)
    critic_2 = SAC_Critic(critic_input_size, hidden_size).to(device)

    actor_optimizer = torch.optim.Adam(
        list(move_actor.parameters()) + list(select_actor.parameters()),
        lr=1e-3,
    )
    critic_1_optimizer = torch.optim.Adam(critic_1.parameters(), lr=2e-3)
    critic_2_optimizer = torch.optim.Adam(critic_2.parameters(), lr=2e-3)

    alpha_continuous = 0.01
    alpha_discrete = 0.01

    for _ in range(steps):
        batch = make_low_level_batch(batch_size, device, max_move=max_move)
        critic_input = torch.cat(
            (
                batch["states"],
                batch["goals"],
                batch["move_actions"],
                batch["service_onehot"],
            ),
            dim=-1,
        )

        q_target = batch["rewards"]
        q_1 = critic_1(critic_input)
        q_2 = critic_2(critic_input)
        critic_1_loss = F.mse_loss(q_1, q_target)
        critic_2_loss = F.mse_loss(q_2, q_target)

        critic_1_optimizer.zero_grad()
        critic_1_loss.backward()
        critic_1_optimizer.step()

        critic_2_optimizer.zero_grad()
        critic_2_loss.backward()
        critic_2_optimizer.step()

        actor_batch = make_low_level_batch(batch_size, device, max_move=max_move)
        actor_input = torch.cat(
            (actor_batch["states"], actor_batch["goals"]),
            dim=-1,
        )
        move_action, move_log_prob = sample_continuous_action(
            move_actor,
            actor_input,
            max_move,
        )

        moved_position = torch.clamp(
            actor_batch["states"][:, :1] + move_action,
            -1.0,
            1.0,
        )
        moved_states = torch.cat(
            (moved_position, actor_batch["states"][:, 1:]),
            dim=-1,
        )
        select_input = torch.cat(
            (moved_states, actor_batch["goals"]),
            dim=-1,
        )
        service_logits = select_actor(select_input)
        service_log_probs = F.log_softmax(service_logits, dim=-1)
        service_probs = service_log_probs.exp()

        # 枚舉所有離散服務動作，使 SelectActor 可透過期望 Q-value 更新。
        expanded_states = actor_batch["states"].unsqueeze(1).expand(
            -1,
            service_size,
            -1,
        )
        expanded_goals = actor_batch["goals"].unsqueeze(1).expand(
            -1,
            service_size,
            -1,
        )
        expanded_moves = move_action.unsqueeze(1).expand(
            -1,
            service_size,
            -1,
        )
        all_services = torch.eye(service_size, device=device).unsqueeze(0)
        all_services = all_services.expand(batch_size, -1, -1)
        all_critic_inputs = torch.cat(
            (
                expanded_states,
                expanded_goals,
                expanded_moves,
                all_services,
            ),
            dim=-1,
        ).reshape(batch_size * service_size, -1)

        for parameter in list(critic_1.parameters()) + list(critic_2.parameters()):
            parameter.requires_grad_(False)

        all_q_1 = critic_1(all_critic_inputs).view(batch_size, service_size)
        all_q_2 = critic_2(all_critic_inputs).view(batch_size, service_size)
        all_min_q = torch.minimum(all_q_1, all_q_2)

        discrete_objective = service_probs * (
            alpha_discrete * service_log_probs - all_min_q
        )
        actor_loss = discrete_objective.sum(dim=-1).mean()
        actor_loss = actor_loss + alpha_continuous * move_log_prob.mean()

        actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_optimizer.step()

        for parameter in list(critic_1.parameters()) + list(critic_2.parameters()):
            parameter.requires_grad_(True)

    with torch.no_grad():
        evaluation = make_low_level_batch(4096, device, max_move=max_move)
        actor_input = torch.cat(
            (evaluation["states"], evaluation["goals"]),
            dim=-1,
        )
        move_mu, _ = move_actor(actor_input)
        move_action = torch.tanh(move_mu) * max_move
        position = evaluation["states"][:, :1]
        moved_position = torch.clamp(position + move_action, -1.0, 1.0)
        moved_states = torch.cat(
            (moved_position, evaluation["states"][:, 1:]),
            dim=-1,
        )
        service_logits = select_actor(
            torch.cat((moved_states, evaluation["goals"]), dim=-1)
        )
        selected_service = service_logits.argmax(dim=-1)

        before_distance = torch.abs(position - evaluation["target"])
        after_distance = torch.abs(moved_position - evaluation["target"])
        optimal_move = torch.clamp(
            evaluation["target"] - position,
            -max_move,
            max_move,
        )
        optimal_after = torch.abs(
            torch.clamp(position + optimal_move, -1.0, 1.0)
            - evaluation["target"]
        )
        actual_gain = (before_distance - after_distance).mean()
        optimal_gain = (before_distance - optimal_after).mean().clamp_min(1e-6)

        move_efficiency = (actual_gain / optimal_gain).item()
        service_accuracy = (
            selected_service == evaluation["goal_index"]
        ).float().mean().item()
        critic_mse = evaluate_critics(critic_1, critic_2, device)

    return move_efficiency, service_accuracy, critic_mse


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark hierarchical networks")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--high-steps", type=int, default=1200)
    parser.add_argument("--low-steps", type=int, default=2500)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run a short smoke benchmark; thresholds remain unchanged.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.quick:
        args.high_steps = min(args.high_steps, 300)
        args.low_steps = min(args.low_steps, 600)

    device = resolve_device(args.device)
    set_seed(args.seed)
    started = time.perf_counter()

    goal_accuracy, goal_regret = benchmark_goal_choice(
        device,
        args.high_steps,
        args.batch_size,
        args.hidden_size,
    )
    move_efficiency, service_accuracy, critic_mse = benchmark_move_and_serve(
        device,
        args.low_steps,
        args.batch_size,
        args.hidden_size,
    )

    # 門檻用於偵測明顯退化，不代表正式 UAV 任務已收斂。
    passed = (
        goal_accuracy >= 0.90
        and goal_regret <= 0.10
        and move_efficiency >= 0.70
        and service_accuracy >= 0.90
        and critic_mse <= 0.10
    )
    result = BenchmarkResult(
        device=str(device),
        seed=args.seed,
        goal_accuracy=goal_accuracy,
        goal_average_regret=goal_regret,
        move_efficiency=move_efficiency,
        service_accuracy=service_accuracy,
        critic_mse=critic_mse,
        elapsed_seconds=time.perf_counter() - started,
        passed=passed,
    )

    print(json.dumps(asdict(result), indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
