# visual_experiment_bundle_buttons.py
# 整合 episode.parquet 與 step.parquet 的互動式可視化工具
#
# 功能：
# 1. 使用 matplotlib Button 建立互動按鈕：
#       [Prev] [Play/Pause] [Next]
# 2. 顯示某個 episode 的 step-by-step 視覺化
# 3. 使用 UAV_action_select，以半透明虛線顯示 UAV 在該 step 服務的 device
# 4. UAV 軌跡只保留最近 traj_window 個 step，預設 10
# 5. 不顯示當次移動箭頭
#
# 安裝需求：
#   pip install pandas pyarrow matplotlib numpy
#
# 使用範例：
#   python visual_experiment_bundle_buttons.py --episode_parquet episode.parquet --step_parquet step.parquet --episode 1
#   python visual_experiment_bundle_buttons.py --episode_parquet episode.parquet --step_parquet step.parquet --episode 1 --interval 500
#   python visual_experiment_bundle_buttons.py --episode_parquet episode.parquet --step_parquet step.parquet --episode 1 --traj_window 10
#   python visual_experiment_bundle_buttons.py --episode_parquet episode.parquet --step_parquet step.parquet --episode 1 --select_base auto

import argparse
import ast
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.widgets import Button


# ============================================================
# 基礎工具
# ============================================================

def is_scalar_number(x):
    return isinstance(x, (int, float, np.integer, np.floating))


def extract_numbers_from_string(s):
    pattern = r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?"
    return [float(x) for x in re.findall(pattern, s)]


def to_python_obj(obj):
    if obj is None:
        return None

    if isinstance(obj, np.ndarray):
        return obj.tolist()

    if isinstance(obj, np.generic):
        return obj.item()

    if isinstance(obj, tuple):
        return [to_python_obj(x) for x in obj]

    if isinstance(obj, list):
        return [to_python_obj(x) for x in obj]

    if isinstance(obj, str):
        try:
            parsed = ast.literal_eval(obj)
            return to_python_obj(parsed)
        except Exception:
            nums = extract_numbers_from_string(obj)
            if nums:
                return nums
            return obj

    return obj


def flatten_numbers(obj):
    obj = to_python_obj(obj)

    if obj is None:
        return []

    if is_scalar_number(obj):
        return [float(obj)]

    if isinstance(obj, str):
        return extract_numbers_from_string(obj)

    if isinstance(obj, list):
        out = []
        for item in obj:
            out.extend(flatten_numbers(item))
        return out

    raise ValueError(f"無法攤平成數字：{obj}, type={type(obj)}")


# ============================================================
# 座標解析
# ============================================================

def flat_to_points(coords, name="coords"):
    """
    支援：
    1. [x1, y1, x2, y2, ...]
    2. [[x1, y1], [x2, y2], ...]
    3. [array([x1, y1]), array([x2, y2]), ...]
    4. 混合格式
    """
    coords = to_python_obj(coords)

    if coords is None:
        return []

    if not isinstance(coords, list):
        raise ValueError(f"{name} 無法解析為 list, type={type(coords)}, value={coords}")

    points = []
    pending = None

    for item in coords:
        item = to_python_obj(item)

        if (
            isinstance(item, list)
            and len(item) == 2
            and is_scalar_number(item[0])
            and is_scalar_number(item[1])
        ):
            points.append((float(item[0]), float(item[1])))
            continue

        if isinstance(item, list):
            sub_points = flat_to_points(item, name=name)
            points.extend(sub_points)
            continue

        if is_scalar_number(item):
            if pending is None:
                pending = float(item)
            else:
                points.append((pending, float(item)))
                pending = None
            continue

        if isinstance(item, str):
            nums = extract_numbers_from_string(item)
            if len(nums) == 2:
                points.append((nums[0], nums[1]))
                continue
            elif len(nums) > 2 and len(nums) % 2 == 0:
                for i in range(0, len(nums), 2):
                    points.append((nums[i], nums[i + 1]))
                continue

        raise ValueError(f"{name} 中出現無法解析的座標格式：{item}, type={type(item)}")

    if pending is not None:
        raise ValueError(f"{name} 座標數量不成對，剩下一個值：{pending}")

    return points


def parse_risk_region(region):
    """
    支援：
    [x1, y1, x2, y2]
    或 [[x1, y1], [x2, y2]]
    """
    region = to_python_obj(region)

    if region is None:
        return None

    if (
        isinstance(region, list)
        and len(region) == 2
        and all(isinstance(v, list) for v in region)
        and all(len(v) == 2 for v in region)
    ):
        nums = [region[0][0], region[0][1], region[1][0], region[1][1]]
    else:
        nums = flatten_numbers(region)

    if len(nums) != 4:
        raise ValueError(f"risk_region 格式錯誤，應為 [x1, y1, x2, y2]，目前為：{region}")

    x1, y1, x2, y2 = [float(v) for v in nums]
    left = min(x1, x2)
    bottom = min(y1, y2)
    width = abs(x2 - x1)
    height = abs(y2 - y1)
    return left, bottom, width, height


def parse_action_select(select_obj):
    """
    解析 UAV_action_select
    預期格式：
      [a0, a1, ...]
    """
    values = flatten_numbers(select_obj)
    return [int(round(v)) for v in values]


def resolve_device_index(action_value, num_devices, select_base="auto"):
    """
    將 select_action 轉成 device index

    select_base:
      - auto:
          優先假設 0 表示 idle，1~N 表示 device 1~N
          若不符合，再退回 0~N-1
      - one:
          0 視為 idle，1~N -> 0~N-1
      - zero:
          0~N-1 直接當 device index
    """
    a = int(action_value)

    if select_base == "one":
        if a == 0:
            return None
        if 1 <= a <= num_devices:
            return a - 1
        return None

    if select_base == "zero":
        if 0 <= a < num_devices:
            return a
        return None

    # auto
    if a == 0:
        return None
    if 1 <= a <= num_devices:
        return a - 1
    if 0 <= a < num_devices:
        return a
    return None


# ============================================================
# DataFrame 讀取與整理
# ============================================================

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


def load_step_df(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"找不到 step parquet：{path}")

    df = pd.read_parquet(path, engine="pyarrow")

    required = ["episode", "step", "UAV_pos", "device_pos"]
    for col in required:
        if col not in df.columns:
            raise KeyError(f"step parquet 缺少必要欄位：{col}")

    df = df.sort_values(["episode", "step"]).reset_index(drop=True)
    df["frame_idx"] = df.groupby("episode").cumcount()

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


# ============================================================
# 繪圖工具
# ============================================================

def compute_axes_limits(uav_points, device_points, risk_box=None, margin=1.0):
    all_x = [float(x) for x, _ in uav_points] + [float(x) for x, _ in device_points]
    all_y = [float(y) for _, y in uav_points] + [float(y) for _, y in device_points]

    if risk_box is not None:
        left, bottom, width, height = risk_box
        all_x += [left, left + width]
        all_y += [bottom, bottom + height]

    if not all_x or not all_y:
        return None

    return (
        min(all_x) - margin,
        max(all_x) + margin,
        min(all_y) - margin,
        max(all_y) + margin,
    )


def draw_common_scene(ax, uav_points, device_points, risk_box=None, show_labels=True, title=None):
    ax.clear()

    # risk region
    if risk_box is not None:
        left, bottom, width, height = risk_box
        rect = Rectangle(
            (left, bottom),
            width,
            height,
            fill=True,
            alpha=0.2,
            linewidth=2,
            label="Risk Region",
        )
        ax.add_patch(rect)

    # UAV
    if uav_points:
        uav_x = [p[0] for p in uav_points]
        uav_y = [p[1] for p in uav_points]
        ax.scatter(uav_x, uav_y, s=140, marker="^", label="UAV")

        if show_labels:
            for idx, (x, y) in enumerate(uav_points, start=1):
                ax.text(float(x), float(y), f" UAV_{idx}", fontsize=10, va="bottom")

    # Device
    if device_points:
        dev_x = [p[0] for p in device_points]
        dev_y = [p[1] for p in device_points]
        ax.scatter(dev_x, dev_y, s=70, marker="o", label="Device")

        if show_labels:
            for idx, (x, y) in enumerate(device_points, start=1):
                ax.text(float(x), float(y), f" D_{idx}", fontsize=8, va="bottom")

    if title is not None:
        ax.set_title(title)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)


# ============================================================
# 互動播放器
# ============================================================

class EpisodeStepViewer:
    def __init__(
        self,
        episode_row,
        step_episode_df,
        interval_ms=500,
        show_labels=True,
        show_traj=True,
        show_service=True,
        traj_window=10,
        select_base="auto",
    ):
        self.episode_row = episode_row
        self.step_episode_df = step_episode_df
        self.interval_ms = interval_ms
        self.show_labels = show_labels
        self.show_traj = show_traj
        self.show_service = show_service
        self.traj_window = max(1, int(traj_window))
        self.select_base = select_base

        self.episode_id = episode_row["episode"]
        self.risk_box = parse_risk_region(episode_row["risk_region"])
        self.current_frame = 0
        self.playing = False

        self.init_uav_points = flat_to_points(episode_row["init_UAV_pos"], name="init_UAV_pos")
        self.init_device_points = flat_to_points(episode_row["init_device_pos"], name="init_device_pos")

        self.all_uav_points = []
        self.all_device_points = []
        for _, row in step_episode_df.iterrows():
            try:
                self.all_uav_points.extend(flat_to_points(row["UAV_pos"], name="UAV_pos"))
            except Exception:
                pass
            try:
                self.all_device_points.extend(flat_to_points(row["device_pos"], name="device_pos"))
            except Exception:
                pass

        if not self.all_uav_points:
            self.all_uav_points = self.init_uav_points
        if not self.all_device_points:
            self.all_device_points = self.init_device_points

        self.limits = compute_axes_limits(
            self.all_uav_points,
            self.all_device_points,
            risk_box=self.risk_box,
        )

        self._setup_figure()
        self._setup_timer()
        self.redraw()

    def _setup_figure(self):
        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        # 預留底部空間放按鈕與狀態文字
        plt.subplots_adjust(bottom=0.22)

        # 按鈕區
        self.ax_prev = self.fig.add_axes([0.18, 0.06, 0.16, 0.08])
        self.ax_play = self.fig.add_axes([0.42, 0.06, 0.16, 0.08])
        self.ax_next = self.fig.add_axes([0.66, 0.06, 0.16, 0.08])

        self.btn_prev = Button(self.ax_prev, "Prev")
        self.btn_play = Button(self.ax_play, "Play")
        self.btn_next = Button(self.ax_next, "Next")

        self.btn_prev.on_clicked(self.on_prev)
        self.btn_play.on_clicked(self.on_play_pause)
        self.btn_next.on_clicked(self.on_next)

        # 鍵盤快捷鍵
        self.fig.canvas.mpl_connect("key_press_event", self.on_key_press)

    def _setup_timer(self):
        self.timer = self.fig.canvas.new_timer(interval=self.interval_ms)
        self.timer.add_callback(self.on_timer)

    def on_timer(self):
        if not self.playing:
            return

        if self.current_frame < len(self.step_episode_df) - 1:
            self.current_frame += 1
            self.redraw()
        else:
            self.playing = False
            self.btn_play.label.set_text("Play")
            self.timer.stop()
            self.redraw()

    def on_prev(self, event=None):
        if self.current_frame > 0:
            self.current_frame -= 1
            self.redraw()

    def on_next(self, event=None):
        if self.current_frame < len(self.step_episode_df) - 1:
            self.current_frame += 1
            self.redraw()

    def on_play_pause(self, event=None):
        self.playing = not self.playing

        if self.playing:
            self.btn_play.label.set_text("Pause")
            self.timer.start()
        else:
            self.btn_play.label.set_text("Play")
            self.timer.stop()

        self.redraw()

    def on_key_press(self, event):
        if event.key == "left":
            if self.playing:
                self.playing = False
                self.btn_play.label.set_text("Play")
                self.timer.stop()
            self.on_prev()

        elif event.key == "right":
            if self.playing:
                self.playing = False
                self.btn_play.label.set_text("Play")
                self.timer.stop()
            self.on_next()

        elif event.key == " ":
            self.on_play_pause()

        elif event.key == "home":
            if self.playing:
                self.playing = False
                self.btn_play.label.set_text("Play")
                self.timer.stop()
            self.current_frame = 0
            self.redraw()

        elif event.key == "end":
            if self.playing:
                self.playing = False
                self.btn_play.label.set_text("Play")
                self.timer.stop()
            self.current_frame = len(self.step_episode_df) - 1
            self.redraw()

    def redraw(self):
        row = self.step_episode_df.iloc[self.current_frame]

        uav_points = flat_to_points(row["UAV_pos"], name="UAV_pos")
        device_points = flat_to_points(row["device_pos"], name="device_pos")

        local_idx = int(row["frame_idx"])
        global_step = row["step"]

        title = f"Episode {self.episode_id} | Frame {local_idx} | Global Step {global_step}"
        draw_common_scene(
            self.ax,
            uav_points=uav_points,
            device_points=device_points,
            risk_box=self.risk_box,
            show_labels=self.show_labels,
            title=title,
        )

        # 畫 UAV 軌跡：只保留最近 traj_window step
        if self.show_traj:
            hist_start = max(0, self.current_frame - self.traj_window + 1)
            hist_df = self.step_episode_df.iloc[hist_start:self.current_frame + 1]
            hist_points = [flat_to_points(v, name="UAV_pos") for v in hist_df["UAV_pos"]]

            if hist_points:
                num_uav = len(hist_points[0])

                for u in range(num_uav):
                    traj = [pts[u] for pts in hist_points if len(pts) > u]
                    if len(traj) >= 2:
                        traj_x = [p[0] for p in traj]
                        traj_y = [p[1] for p in traj]
                        self.ax.plot(traj_x, traj_y, linewidth=2, alpha=0.8)

        # 畫 service line：半透明虛線連到被服務的 device
        if self.show_service and "UAV_action_select" in row.index:
            try:
                select_actions = parse_action_select(row["UAV_action_select"])
                first_service_label = True

                for i, (ux, uy) in enumerate(uav_points):
                    if i >= len(select_actions):
                        continue

                    dev_idx = resolve_device_index(
                        select_actions[i],
                        num_devices=len(device_points),
                        select_base=self.select_base,
                    )

                    if dev_idx is None:
                        continue

                    if 0 <= dev_idx < len(device_points):
                        dx, dy = device_points[dev_idx]
                        line_label = "Service Link" if first_service_label else None
                        first_service_label = False

                        self.ax.plot(
                            [ux, dx],
                            [uy, dy],
                            linestyle="--",
                            linewidth=2,
                            alpha=0.35,
                            label=line_label,
                        )
            except Exception:
                pass

        # 顯示資訊
        info_lines = [
            f"Frame: {self.current_frame + 1}/{len(self.step_episode_df)}",
            f"Playing: {'Yes' if self.playing else 'No'}",
        ]

        if "reward" in row.index:
            info_lines.append(f"reward = {to_python_obj(row['reward'])}")
        if "UAV_action_select" in row.index:
            info_lines.append(f"select = {to_python_obj(row['UAV_action_select'])}")

        self.ax.text(
            0.02,
            0.98,
            "\n".join(info_lines),
            transform=self.ax.transAxes,
            va="top",
            fontsize=9,
            bbox=dict(alpha=0.2),
        )

        if self.limits is not None:
            xmin, xmax, ymin, ymax = self.limits
            self.ax.set_xlim(xmin, xmax)
            self.ax.set_ylim(ymin, ymax)

        self.ax.legend(loc="upper right")
        self.fig.canvas.draw_idle()

    def show(self):
        plt.show()


# ============================================================
# 主程式
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--episode_parquet", type=str, default="episode.parquet")
    parser.add_argument("--step_parquet", type=str, default="step.parquet")
    parser.add_argument("--episode", type=int, required=True)
    parser.add_argument("--interval", type=int, default=500, help="播放間隔，單位 ms，預設 500")
    parser.add_argument("--no_label", action="store_true", help="不顯示 UAV / Device 文字標籤")
    parser.add_argument("--no_traj", action="store_true", help="不顯示 UAV 軌跡線")
    parser.add_argument("--no_service", action="store_true", help="不顯示 UAV 到被服務 device 的虛線")
    parser.add_argument("--traj_window", type=int, default=10, help="UAV 軌跡保留最近幾個 step，預設 10")
    parser.add_argument(
        "--select_base",
        type=str,
        default="auto",
        choices=["auto", "one", "zero"],
        help="UAV_action_select 的 device 編號方式：auto / one / zero",
    )

    args = parser.parse_args()

    episode_df = load_episode_df(args.episode_parquet)
    step_df = load_step_df(args.step_parquet)

    episode_row = get_episode_init_row(episode_df, args.episode)
    step_episode_df = get_episode_step_df(step_df, args.episode)

    viewer = EpisodeStepViewer(
        episode_row=episode_row,
        step_episode_df=step_episode_df,
        interval_ms=args.interval,
        show_labels=not args.no_label,
        show_traj=not args.no_traj,
        show_service=not args.no_service,
        traj_window=args.traj_window,
        select_base=args.select_base,
    )
    viewer.show()


if __name__ == "__main__":
    main()
