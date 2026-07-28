"""networks.py 的基準單元測試。

執行方式：
    python -m unittest -v test_networks.py

本檔只驗證網路本身，不依賴 Environment、Agent 或訓練資料格式。
"""

import unittest

import torch

from networks import (
    DDQN,
    SAC_Continuous_Actor,
    SAC_Critic,
    SAC_Discrete_Actor,
)


class NetworkShapeTests(unittest.TestCase):
    """確認所有網路遵守預期的輸入與輸出契約。"""

    batch_size = 8
    state_size = 34
    goal_size = 10
    move_size = 2
    service_size = 11
    hidden_size = 32

    def setUp(self):
        torch.manual_seed(7)
        self.state = torch.randn(self.batch_size, self.state_size)
        self.goal = torch.randn(self.batch_size, self.goal_size)

    def assert_finite(self, *tensors):
        for tensor in tensors:
            self.assertTrue(torch.isfinite(tensor).all().item())

    def test_high_level_ddqn_output(self):
        network = DDQN(self.state_size, self.goal_size, self.hidden_size)

        q_values = network(self.state)

        self.assertEqual(q_values.shape, (self.batch_size, self.goal_size))
        self.assert_finite(q_values)

    def test_continuous_actor_output(self):
        network = SAC_Continuous_Actor(
            self.state_size + self.goal_size,
            self.move_size,
            self.hidden_size,
        )

        mu, log_std = network(torch.cat((self.state, self.goal), dim=-1))

        expected_shape = (self.batch_size, self.move_size)
        self.assertEqual(mu.shape, expected_shape)
        self.assertEqual(log_std.shape, expected_shape)
        self.assert_finite(mu, log_std)

    def test_discrete_actor_output(self):
        network = SAC_Discrete_Actor(
            self.state_size + self.goal_size,
            self.service_size,
            self.hidden_size,
        )

        logits = network(torch.cat((self.state, self.goal), dim=-1))

        self.assertEqual(logits.shape, (self.batch_size, self.service_size))
        self.assert_finite(logits)

    def test_joint_critic_output(self):
        critic_input_size = (
            self.state_size
            + self.goal_size
            + self.move_size
            + self.service_size
        )
        network = SAC_Critic(critic_input_size, self.hidden_size)
        critic_input = torch.randn(self.batch_size, critic_input_size)

        q_value = network(critic_input)

        self.assertEqual(q_value.shape, (self.batch_size, 1))
        self.assert_finite(q_value)

class NetworkGradientTests(unittest.TestCase):
    """確認階層式網路可反向傳播並由 optimizer 更新。"""

    def test_ddqn_backward_and_update(self):
        torch.manual_seed(9)
        network = DDQN(state_size=5, action_size=3, layer_size=16)
        optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)
        states = torch.randn(12, 5)
        targets = torch.randn(12, 3)
        before = [parameter.detach().clone() for parameter in network.parameters()]

        loss = torch.nn.functional.mse_loss(network(states), targets)
        optimizer.zero_grad()
        loss.backward()

        for parameter in network.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all().item())

        optimizer.step()

        self.assertTrue(
            any(
                not torch.equal(old, new.detach())
                for old, new in zip(before, network.parameters())
            )
        )

    def test_joint_actor_critic_backward_and_update(self):
        torch.manual_seed(11)

        batch_size = 16
        state_size = 6
        goal_size = 3
        move_size = 2
        service_size = 4
        hidden_size = 32

        move_actor = SAC_Continuous_Actor(
            state_size + goal_size,
            move_size,
            hidden_size,
        )
        select_actor = SAC_Discrete_Actor(
            state_size + goal_size,
            service_size,
            hidden_size,
        )
        critic = SAC_Critic(
            state_size + goal_size + move_size + service_size,
            hidden_size,
        )

        network_groups = (
            list(move_actor.parameters()),
            list(select_actor.parameters()),
            list(critic.parameters()),
        )
        parameters = [
            parameter
            for group in network_groups
            for parameter in group
        ]
        optimizer = torch.optim.Adam(parameters, lr=1e-3)

        state = torch.randn(batch_size, state_size)
        goal = torch.nn.functional.one_hot(
            torch.randint(goal_size, (batch_size,)),
            num_classes=goal_size,
        ).float()

        actor_input = torch.cat((state, goal), dim=-1)
        mu, log_std = move_actor(actor_input)
        service_logits = select_actor(actor_input)
        move_action = torch.tanh(mu)
        service_probs = torch.softmax(service_logits, dim=-1)
        critic_input = torch.cat(
            (state, goal, move_action, service_probs),
            dim=-1,
        )
        q_value = critic(critic_input)

        # 正則項確保兩個 Actor 的所有輸出 head 都參與這次測試。
        loss = -q_value.mean()
        loss = loss + 0.01 * mu.square().mean()
        loss = loss + 0.01 * log_std.square().mean()
        loss = loss + 0.01 * service_logits.square().mean()

        before_groups = [
            [parameter.detach().clone() for parameter in group]
            for group in network_groups
        ]
        optimizer.zero_grad()
        loss.backward()

        for parameter in parameters:
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all().item())

        optimizer.step()

        for before, after in zip(before_groups, network_groups):
            self.assertTrue(
                any(
                    not torch.equal(old, new.detach())
                    for old, new in zip(before, after)
                )
            )

    def test_wrong_input_dimension_raises_runtime_error(self):
        network = SAC_Critic(input_size=12, layer_size=16)

        with self.assertRaises(RuntimeError):
            network(torch.randn(4, 11))


if __name__ == "__main__":
    unittest.main(verbosity=2)
