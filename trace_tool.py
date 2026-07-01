import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from matplotlib.patches import Rectangle


# 讀取Episode
def load_episode_df(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"找不到 episode parquet：{path}")

    df = pd.read_parquet(path, engine="pyarrow")

    required = ["episode", "init_UAV_pos", "init_device_pos", "risk_region"]
    for col in required:
        if col not in df.columns:
            raise KeyError(f"episode parquet 缺少必要欄位：{col}")

    return df

# 讀取Step
def load_step_df(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"找不到 step parquet：{path}")

    df = pd.read_parquet(path, engine="pyarrow")

    required = ["episode", "step", "UAV_pos", "device_pos"]
    for col in required:
        if col not in df.columns:
            raise KeyError(f"step parquet 缺少必要欄位：{col}")

    return df

def get_episode_init_row(episode_df, episode_id):
    matched = episode_df[episode_df["episode"] == episode_id]
    if len(matched) == 0:
        raise ValueError(f"在 episode parquet 中找不到 episode = {episode_id}")
    return matched.iloc[0]


def get_episode_step_df(step_df, episode_id):
    matched = step_df[step_df["episode"] == episode_id].sort_values("step").reset_index(drop=True)
    if len(matched) == 0:
        raise ValueError(f"在 step parquet 中找不到 episode = {episode_id}")
    matched["frame_idx"] = np.arange(len(matched))
    return matched

def parse_episode_data(episode_data):
    init_UAV_pos = episode_data["init_UAV_pos"]
    init_UAV_pos = np.array(init_UAV_pos).reshape(-1, 2).tolist()

    return {
        "episode": episode_data["episode"],
        "init_UAV_pos": init_UAV_pos,
        "init_device_pos": episode_data["init_device_pos"],
        "risk_region": episode_data["risk_region"]
    }

def parse_step_data(step_data):
    keys = ["step", "UAV_pos", "device_pos", "device_AOI", "UAV_action_select", "UAV_target_point", "reward"]
    step_all = {key: [] for key in keys}

    for _, step_row in step_data.iterrows():
        step = step_row["step"]
        step = step % 100 if step % 100 else 100

        UAV_pos = step_row["UAV_pos"]
        UAV_pos = np.array(UAV_pos).reshape(-1, 2).tolist()

        device_pos = [arr.tolist() for arr in step_row["device_pos"]]

        UAV_target_point = [arr.tolist() for arr in step_row["UAV_target_point"]]
        

        step_all["step"].append(step)
        step_all["UAV_pos"].append(UAV_pos)
        step_all["device_pos"].append(device_pos)
        step_all["device_AOI"].append(step_row["device_AOI"].tolist())
        step_all["UAV_action_select"].append(step_row["UAV_action_select"].tolist())
        step_all["UAV_target_point"].append(UAV_target_point)
        step_all["reward"].append(step_row["reward"].tolist())

    return step_all

def draw_common_scene(
    ax,
    UAV_pos,
    device_pos,
    risk_region,
    title,
    device_AOI,
):
    ax.clear()

    rect = Rectangle(
        (0, 0),
        10,
        10,
        fill = False,
        alpha = 0.5,
        linewidth = 2,
        color = "black"
    )
    ax.add_patch(rect)

    for risk_box in risk_region:
        x1, y1, x2, y2 = risk_box
        rect = Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            fill = True,
            alpha = 0.15,
            linewidth = 2,
            label = "Risk Region",
            color = "red"
        )
        ax.add_patch(rect)

    UAV_x = [p[0] for p in UAV_pos]
    UAV_y = [p[1] for p in UAV_pos]
    ax.scatter(UAV_x, UAV_y, s=50, marker="s", label="UAV")
    for idx, (x, y) in enumerate(UAV_pos, start = 1):
        ax.text(float(x), float(y)-0.15, fr"$\text{{UAV}}_{{{idx}}}$", fontsize=10, va="top", ha="center")

    device_x = [p[0] for p in device_pos]
    device_y = [p[1] for p in device_pos]
    sizes = [25 + 175 * (AOI / 100) for AOI in device_AOI]
    ax.scatter(device_x, device_y, s=sizes, marker="o", alpha=1, label="Device")
    for idx, (x, y) in enumerate(device_pos, start = 1):
        ax.text(float(x), float(y)-0.15, fr"$\text{{D}}_{{{idx}}}$", fontsize=10, va="top", ha="center")
        ax.text(float(x), float(y), fr"{int(device_AOI[idx - 1])}", fontsize=6, va="center", ha="center")

    ax.plot([0], [0], linewidth=4, alpha=0.95, color="red", label = f"Latest Move", linestyle='-')
    ax.plot([0], [0], linewidth=2, alpha=0.35, color="black", label = f"Service Link", linestyle='--')

    ax.set_title(title)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)

class EpisodeStepViewer:
    def __init__(
        self,
        episode,
        episode_df,
        step_df,
        interval,
    ):
        self.episode = episode
        self.episode_df = episode_df
        self.step_df = step_df
        self.interval = interval

        episode_data = self.episode_df[self.episode_df["episode"] == self.episode]
        if len(episode_data) == 0:
            raise ValueError(f"在 episode parquet 中找不到 episode = {self.episode}")
        self.episode_data = parse_episode_data(episode_data.iloc[0])

        self.step = 0
        self.playing = False

        step_data = self.step_df[self.step_df["episode"] == self.episode].sort_values("step")
        self.step_data = parse_step_data(step_data)

        self.fig, self.ax = plt.subplots(figsize=(9, 6))
        plt.subplots_adjust(bottom=0.18)

        self.ax_prev = self.fig.add_axes([0.20, 0.04, 0.15, 0.07])
        self.ax_play = self.fig.add_axes([0.42, 0.04, 0.15, 0.07])
        self.ax_next = self.fig.add_axes([0.64, 0.04, 0.15, 0.07])
        
        self.btn_prev = Button(self.ax_prev, "<")
        self.btn_play = Button(self.ax_play, "Play")
        self.btn_next = Button(self.ax_next, ">")

        self.btn_prev.on_clicked(self.on_prev)
        self.btn_play.on_clicked(self.on_play_pause)
        self.btn_next.on_clicked(self.on_next)

        self.timer = self.fig.canvas.new_timer(interval=self.interval)
        self.timer.add_callback(self.on_timer)

        self.draw_init_frame()

    def draw_init_frame(self):
        episode = self.episode_data["episode"]
        init_UAV_pos = self.episode_data["init_UAV_pos"]
        init_device_pos = self.episode_data["init_device_pos"]
        risk_region = self.episode_data["risk_region"]

        title = f"Episode {episode} | Step {self.step}"

        draw_common_scene(
            ax = self.ax,
            UAV_pos = init_UAV_pos,
            device_pos = init_device_pos,
            risk_region = risk_region,
            title = title,
            device_AOI = [0] * 10
        )

        self.ax.set_xlim(-1, 11)
        self.ax.set_ylim(-1, 11)

        self.ax.legend(loc = "upper right", bbox_to_anchor=(1.4, 1))
        self.fig.canvas.draw_idle()

    def draw_frame(self):
        episode = self.episode_data["episode"]
        risk_region = self.episode_data["risk_region"]

        assert self.step >= 1
        UAV_pos = self.step_data["UAV_pos"][self.step - 1]
        device_pos = self.step_data["device_pos"][self.step - 1]
        device_AOI = self.step_data["device_AOI"][self.step - 1]
        UAV_action_select = self.step_data["UAV_action_select"][self.step - 1]
        UAV_target_point = self.step_data["UAV_target_point"][self.step - 1]
        reward = self.step_data["reward"][self.step - 1]

        title = f"Episode {episode} | Step {self.step}"

        draw_common_scene(
            ax = self.ax,
            UAV_pos = UAV_pos,
            device_pos = device_pos,
            risk_region = risk_region,
            title = title,
            device_AOI = device_AOI
        )

        traj_start = max(0, self.step - 10)
        traj_UAV_pos = []
        if self.step <= 10:
            traj_UAV_pos = [self.episode_data["init_UAV_pos"]]
        else:
            traj_UAV_pos = []
        traj_UAV_pos.extend(self.step_data["UAV_pos"][traj_start : self.step])
        self.draw_target(UAV_target_pos = UAV_target_point)
        self.draw_trajectory(traj_UAV_pos = traj_UAV_pos)

        self.draw_service(
            UAV_pos = UAV_pos,
            device_pos = device_pos,
            UAV_action_select = UAV_action_select
        )        

        self.draw_reward(
            UAV_pos = UAV_pos,
            reward = reward
        )

        self.ax.set_xlim(-1, 11)
        self.ax.set_ylim(-1, 11)
        self.ax.legend(loc="upper right", bbox_to_anchor=(1.4, 1))
        self.fig.canvas.draw_idle()

    def draw_target(self, UAV_target_pos):
        UAV_x = [p[0] for p in UAV_target_pos]
        UAV_y = [p[1] for p in UAV_target_pos]
        self.ax.scatter(UAV_x, UAV_y, s=200, marker="1", label="Target")

    def draw_trajectory(self, traj_UAV_pos):
        for u in range(len(traj_UAV_pos[0])):
            UAV_x = [p[u][0] for p in traj_UAV_pos]
            UAV_y = [p[u][1] for p in traj_UAV_pos]

            self.ax.plot(UAV_x[:-1], UAV_y[:-1], linewidth=2, alpha=0.35, linestyle='-')
            self.ax.plot(UAV_x[-2:], UAV_y[-2:], linewidth=4, alpha=0.95, color="red", linestyle='-')

    def draw_service(self, UAV_pos, device_pos, UAV_action_select):
        for u in range(len(UAV_pos[0])):
            action_select = UAV_action_select[u]
            if action_select == len(device_pos): 
                continue
            
            service_x = [UAV_pos[u][0], device_pos[action_select][0]]
            service_y = [UAV_pos[u][1], device_pos[action_select][1]]

            self.ax.plot(service_x, service_y, linewidth=2, alpha=0.35, linestyle='--', color="black")
                
    def draw_reward(self, UAV_pos, reward):
        for idx, (x, y) in enumerate(UAV_pos, start = 1):
            self.ax.text(float(x), float(y)-0.6, fr"{reward[idx - 1]:.1f}", fontsize=8, va="top", ha="center")

    def on_prev(self, event):
        self.playing = False
        self.step = max(0, self.step - 1)
        if self.step == 0:
            self.draw_init_frame()
        else:
            self.draw_frame()

    def on_next(self, event):
        self.playing = False
        self.step = min(100, self.step + 1)
        self.draw_frame()

    def on_play_pause(self, event):
        self.playing = not self.playing

        if self.playing:
            if self.step == 100:
                self.step = 0
            self.timer.start()
        else:
            self.timer.stop()
        
        self.btn_play.label.set_text("Pause" if self.playing else "Play")

    def on_timer(self):
        if not self.playing:
            return

        if self.step < 100:
            self.step += 1
            self.draw_frame()
            self.timer.start()
        else:
            self.playing = False
            self.timer.stop()
            self.btn_play.label.set_text("Replay")
            self.draw_frame()
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode_parquet", type=str, default="episode.parquet")
    parser.add_argument("--step_parquet", type=str, default="step.parquet")
    parser.add_argument("--episode", type=int, required=True)
    parser.add_argument("--interval", type=int, default=10, help="自動播放時每幀間隔，單位 ms，預設 500")
    parser.add_argument(
        "--display",
        type=str,
        default="all",
        choices=["all", "move-only", "action-only", "static"],
        help="all: 完全顯示, move-only: 包含移動軌跡, action-only: 包含服務動作, static: 僅狀態",
    )
    parser.add_argument("--trajectory", type=int, default=10, help="UAV軌跡保留最近幾個step，預設 10")

    args = parser.parse_args()

    episode_df = load_episode_df(args.episode_parquet)
    step_df = load_step_df(args.step_parquet)

    viewer = EpisodeStepViewer(
        episode = args.episode,
        episode_df = episode_df,
        step_df = step_df,
        interval = args.interval
    )
    plt.show()