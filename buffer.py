import numpy as np
import random
import torch
from collections import deque, namedtuple

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, batch_size, device):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.device = device
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.stack([e.state for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.stack([e.next_state for e in experiences if e is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)
  
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)


class LowLevelReplayBuffer:
    """保存 low-level SAC 更新需要的完整逐步 transition。"""

    def __init__(self, buffer_size, batch_size, device):
        """建立固定容量的低層記憶體及取樣設定。"""
        self.device = torch.device(device)
        self.batch_size = int(batch_size)
        self.memory = deque(maxlen=int(buffer_size))
        self.experience = namedtuple(
            "LowLevelExperience",
            field_names=[
                "state",
                "goal",
                "move_action",
                "moved_state",
                "service_action",
                "reward",
                "next_state",
                "next_goal",
                "done",
                "action_mask",
                "next_action_mask",
            ],
        )

    def add(
        self,
        state,
        goal,
        move_action,
        moved_state,
        service_action,
        reward,
        next_state,
        next_goal,
        done,
        action_mask=None,
        next_action_mask=None,
    ):
        """複製並保存一筆低層 transition，避免外部陣列後續更新污染資料。"""
        experience = self.experience(
            state=np.asarray(state, dtype=np.float32).copy(),
            goal=int(np.asarray(goal).item()),
            move_action=np.asarray(
                move_action,
                dtype=np.float32,
            ).reshape(-1).copy(),
            moved_state=np.asarray(
                moved_state,
                dtype=np.float32,
            ).copy(),
            service_action=int(np.asarray(service_action).item()),
            reward=float(np.asarray(reward).item()),
            next_state=np.asarray(
                next_state,
                dtype=np.float32,
            ).copy(),
            next_goal=int(np.asarray(next_goal).item()),
            done=bool(np.asarray(done).item()),
            action_mask=self._copy_mask(action_mask),
            next_action_mask=self._copy_mask(next_action_mask),
        )
        self.memory.append(experience)

    def sample(self):
        """隨機取樣並回傳可直接交給 HSAC_Agent.learn() 的 tensor dict。"""
        if len(self.memory) < self.batch_size:
            raise RuntimeError(
                "LowLevelReplayBuffer 的樣本數不足以形成一個 batch。"
            )

        experiences = random.sample(
            self.memory,
            k=self.batch_size,
        )
        batch = {
            "states": self._float_tensor(
                np.stack([item.state for item in experiences])
            ),
            "goals": self._long_tensor(
                [item.goal for item in experiences]
            ),
            "move_actions": self._float_tensor(
                np.stack([item.move_action for item in experiences])
            ),
            "moved_states": self._float_tensor(
                np.stack([item.moved_state for item in experiences])
            ),
            "service_actions": self._long_tensor(
                [item.service_action for item in experiences]
            ),
            "rewards": self._column_float_tensor(
                [item.reward for item in experiences]
            ),
            "next_states": self._float_tensor(
                np.stack([item.next_state for item in experiences])
            ),
            "next_goals": self._long_tensor(
                [item.next_goal for item in experiences]
            ),
            "dones": self._column_float_tensor(
                [item.done for item in experiences]
            ),
        }

        action_masks = self._stack_optional_masks(
            [item.action_mask for item in experiences],
            "action_mask",
        )
        next_action_masks = self._stack_optional_masks(
            [item.next_action_mask for item in experiences],
            "next_action_mask",
        )
        if action_masks is not None:
            batch["action_masks"] = torch.as_tensor(
                action_masks,
                dtype=torch.bool,
                device=self.device,
            )
        if next_action_masks is not None:
            batch["next_action_masks"] = torch.as_tensor(
                next_action_masks,
                dtype=torch.bool,
                device=self.device,
            )

        return batch

    def __len__(self):
        """回傳目前保存的低層 transition 數量。"""
        return len(self.memory)

    @staticmethod
    def _copy_mask(mask):
        """將可選 action mask 複製為一維 bool 陣列。"""
        if mask is None:
            return None
        return np.asarray(mask, dtype=np.bool_).reshape(-1).copy()

    @staticmethod
    def _stack_optional_masks(masks, name):
        """堆疊可選 masks，並阻止同一 batch 混合有／無 mask 的資料。"""
        if all(mask is None for mask in masks):
            return None
        if any(mask is None for mask in masks):
            raise RuntimeError(
                f"同一個 low-level batch 的 {name} 必須全部提供或全部省略。"
            )
        return np.stack(masks)

    def _float_tensor(self, values):
        """建立位於 buffer device 的 float32 tensor。"""
        return torch.as_tensor(
            values,
            dtype=torch.float32,
            device=self.device,
        )

    def _long_tensor(self, values):
        """建立位於 buffer device 的 int64 tensor。"""
        return torch.as_tensor(
            values,
            dtype=torch.long,
            device=self.device,
        )

    def _column_float_tensor(self, values):
        """建立 shape 為 (batch, 1) 的 float32 tensor。"""
        return self._float_tensor(values).reshape(-1, 1)


class HighLevelReplayBuffer:
    """保存 high-level DQN 的區段 transition。"""

    def __init__(self, buffer_size, batch_size, device):
        """建立固定容量的高層記憶體及取樣設定。"""
        self.device = torch.device(device)
        self.batch_size = int(batch_size)
        self.memory = deque(maxlen=int(buffer_size))
        self.experience = namedtuple(
            "HighLevelExperience",
            field_names=[
                "state",
                "goal",
                "cumulative_reward",
                "next_state",
                "done",
                "duration",
            ],
        )

    def add(self, state, goal, cumulative_reward, next_state, done, duration):
        """複製並保存一筆包含實際持續時間的高層 transition。"""
        experience = self.experience(
            state=np.asarray(state, dtype=np.float32).copy(),
            goal=int(np.asarray(goal).item()),
            cumulative_reward=float(
                np.asarray(cumulative_reward).item()
            ),
            next_state=np.asarray(
                next_state,
                dtype=np.float32,
            ).copy(),
            done=bool(np.asarray(done).item()),
            duration=int(np.asarray(duration).item()),
        )
        self.memory.append(experience)

    def sample(self):
        """隨機取樣並回傳可直接交給 HSAC_Agent.learn() 的 tensor dict。"""
        if len(self.memory) < self.batch_size:
            raise RuntimeError(
                "HighLevelReplayBuffer 的樣本數不足以形成一個 batch。"
            )

        experiences = random.sample(
            self.memory,
            k=self.batch_size,
        )
        return {
            "states": self._float_tensor(
                np.stack([item.state for item in experiences])
            ),
            "goals": self._long_tensor(
                [item.goal for item in experiences]
            ),
            "cumulative_rewards": self._column_float_tensor(
                [item.cumulative_reward for item in experiences]
            ),
            "next_states": self._float_tensor(
                np.stack([item.next_state for item in experiences])
            ),
            "dones": self._column_float_tensor(
                [item.done for item in experiences]
            ),
            "durations": self._column_float_tensor(
                [item.duration for item in experiences]
            ),
        }

    def __len__(self):
        """回傳目前保存的高層 transition 數量。"""
        return len(self.memory)

    def _float_tensor(self, values):
        """建立位於 buffer device 的 float32 tensor。"""
        return torch.as_tensor(
            values,
            dtype=torch.float32,
            device=self.device,
        )

    def _long_tensor(self, values):
        """建立位於 buffer device 的 int64 tensor。"""
        return torch.as_tensor(
            values,
            dtype=torch.long,
            device=self.device,
        )

    def _column_float_tensor(self, values):
        """建立 shape 為 (batch, 1) 的 float32 tensor。"""
        return self._float_tensor(values).reshape(-1, 1)
