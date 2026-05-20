# visual_experiment_bundle_v2.py
# 整合 episode.parquet 與 step.parquet 的可視化工具
#
# 新增功能：
# 1. 依據 UAV_action_select，在目前 step 用半透明虛線顯示 UAV 正在服務的 device
# 2. UAV 軌跡只保留最近 traj_window 個 step，預設 10
#
# 安裝需求：
#   pip install pandas pyarrow matplotlib numpy pillow
#
# 使用範例：
#   python visual_experiment_bundle_v2.py --episode_parquet episode.parquet --step_parquet step.parquet --episode 1 --mode animate
#   python visual_experiment_bundle_v2.py --episode_parquet episode.parquet --step_parquet step.parquet --episode 1 --mode animate --interval 500 --traj_window 10
#   python visual_experiment_bundle_v2.py --episode_parquet episode.parquet --step_parquet step.parquet --episode 1 --mode animate --select_base auto
#   python visual_experiment_bundle_v2.py --episode_parquet episode.parquet --step_parquet step.parquet --episode 1 --mode animate --save outputs/ep1.gif

import argparse
import ast
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.animation import FuncAnimation


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


def parse_action_move(move_obj):
    """
    解析 UAV_action_move
    預期格式：
      [[dx1, dy1], [dx2, dy2], ...]
    若為 flat list 也可處理
    """
    return flat_to_points(move_obj, name="UAV_action_move")


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


def plot_episode_init(episode_row, save_path=None, show_labels=True):
    episode_id = episode_row["episode"]
    uav_points = flat_to_points(episode_row["init_UAV_pos"], name="init_UAV_pos")
    device_points = flat_to_points(episode_row["init_device_pos"], name="init_device_pos")
    risk_box = parse_risk_region(episode_row["risk_region"])

    fig, ax = plt.subplots(figsize=(8, 8))
    draw_common_scene(
        ax,
        uav_points=uav_points,
        device_points=device_points,
        risk_box=risk_box,
        show_labels=show_labels,
        title=f"Episode {episode_id} - Initial Scene",
    )

    limits = compute_axes_limits(uav_points, device_points, risk_box=risk_box)
    if limits is not None:
        xmin, xmax, ymin, ymax = limits
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)

    ax.legend()
    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200)
        print(f"已儲存：{save_path}")
        plt.close(fig)
    else:
        plt.show()


def animate_episode(
    episode_row,
    step_episode_df,
    interval_ms=500,
    save_path=None,
    show_labels=True,
    show_traj=True,
    show_arrow=True,
    show_service=True,
    traj_window=10,
    select_base="auto",
):
    episode_id = episode_row["episode"]
    risk_box = parse_risk_region(episode_row["risk_region"])

    init_uav_points = flat_to_points(episode_row["init_UAV_pos"], name="init_UAV_pos")
    init_device_points = flat_to_points(episode_row["init_device_pos"], name="init_device_pos")

    # 先掃過全部 frame，取得固定座標範圍
    all_uav_points = []
    all_device_points = []
    for _, row in step_episode_df.iterrows():
        try:
            all_uav_points.extend(flat_to_points(row["UAV_pos"], name="UAV_pos"))
        except Exception:
            pass

        try:
            all_device_points.extend(flat_to_points(row["device_pos"], name="device_pos"))
        except Exception:
            pass

    if not all_uav_points:
        all_uav_points = init_uav_points

    if not all_device_points:
        all_device_points = init_device_points

    limits = compute_axes_limits(all_uav_points, all_device_points, risk_box=risk_box)

    fig, ax = plt.subplots(figsize=(8, 8))

    def update(frame_idx):
        row = step_episode_df.iloc[frame_idx]

        uav_points = flat_to_points(row["UAV_pos"], name="UAV_pos")
        device_points = flat_to_points(row["device_pos"], name="device_pos")

        local_idx = int(row["frame_idx"])
        global_step = row["step"]

        title = f"Episode {episode_id} | Frame {local_idx} | Global Step {global_step}"
        draw_common_scene(
            ax,
            uav_points=uav_points,
            device_points=device_points,
            risk_box=risk_box,
            show_labels=show_labels,
            title=title,
        )

        # 畫 UAV 軌跡：只保留最近 traj_window step
        if show_traj:
            hist_start = max(0, frame_idx - traj_window + 1)
            hist_df = step_episode_df.iloc[hist_start:frame_idx + 1]
            hist_points = [flat_to_points(v, name="UAV_pos") for v in hist_df["UAV_pos"]]

            if hist_points:
                num_uav = len(hist_points[0])

                for u in range(num_uav):
                    traj = [pts[u] for pts in hist_points if len(pts) > u]
                    if len(traj) >= 2:
                        traj_x = [p[0] for p in traj]
                        traj_y = [p[1] for p in traj]
                        ax.plot(traj_x, traj_y, linewidth=2, alpha=0.8)

        # 畫 action move 箭頭
        if show_arrow and "UAV_action_move" in row.index:
            try:
                move_points = parse_action_move(row["UAV_action_move"])
                for i, (ux, uy) in enumerate(uav_points):
                    if i < len(move_points):
                        dx, dy = move_points[i]
                        ax.arrow(
                            ux,
                            uy,
                            dx,
                            dy,
                            length_includes_head=True,
                            head_width=0.15,
                            alpha=0.7,
                        )
            except Exception:
                pass

        # 畫 service line：半透明虛線連到被服務的 device
        if show_service and "UAV_action_select" in row.index:
            try:
                select_actions = parse_action_select(row["UAV_action_select"])
                first_service_label = True

                for i, (ux, uy) in enumerate(uav_points):
                    if i >= len(select_actions):
                        continue

                    dev_idx = resolve_device_index(
                        select_actions[i],
                        num_devices=len(device_points),
                        select_base=select_base,
                    )

                    if dev_idx is None:
                        continue

                    if 0 <= dev_idx < len(device_points):
                        dx, dy = device_points[dev_idx]
                        line_label = "Service Link" if first_service_label else None
                        first_service_label = False

                        ax.plot(
                            [ux, dx],
                            [uy, dy],
                            linestyle="--",
                            linewidth=2,
                            alpha=0.35,
                            label=line_label,
                        )
            except Exception:
                pass

        # 顯示簡單文字資訊
        info_lines = []
        if "reward" in row.index:
            info_lines.append(f"reward = {to_python_obj(row['reward'])}")
        if "UAV_action_select" in row.index:
            info_lines.append(f"select = {to_python_obj(row['UAV_action_select'])}")

        if info_lines:
            ax.text(
                0.02,
                0.98,
                "\n".join(info_lines),
                transform=ax.transAxes,
                va="top",
                fontsize=9,
                bbox=dict(alpha=0.2),
            )

        if limits is not None:
            xmin, xmax, ymin, ymax = limits
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)

        ax.legend(loc="upper right")

    ani = FuncAnimation(
        fig,
        update,
        frames=len(step_episode_df),
        interval=interval_ms,
        repeat=False,
    )

    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        suffix = save_path.suffix.lower()
        if suffix == ".gif":
            ani.save(save_path, writer="pillow")
        elif suffix == ".mp4":
            ani.save(save_path, writer="ffmpeg")
        else:
            raise ValueError("save_path 若指定，副檔名請使用 .gif 或 .mp4")

        print(f"已儲存動畫：{save_path}")
        plt.close(fig)
    else:
        plt.show()


# ============================================================
# 主程式
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--episode_parquet", type=str, default="episode.parquet")
    parser.add_argument("--step_parquet", type=str, default="step.parquet")
    parser.add_argument("--episode", type=int, required=True)
    parser.add_argument(
        "--mode",
        type=str,
        default="both",
        choices=["init", "animate", "both"],
        help="init: 顯示初始化圖, animate: 播放動畫, both: 先顯示初始化圖再播放動畫",
    )
    parser.add_argument("--interval", type=int, default=500, help="動畫每幀間隔，單位 ms，預設 500")
    parser.add_argument("--save", type=str, default=None, help="若 mode=animate，可指定輸出 gif/mp4 路徑")
    parser.add_argument("--no_label", action="store_true", help="不顯示 UAV / Device 文字標籤")
    parser.add_argument("--no_traj", action="store_true", help="不顯示 UAV 軌跡線")
    parser.add_argument("--no_arrow", action="store_true", help="不顯示 action move 箭頭")
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

    show_labels = not args.no_label
    show_traj = not args.no_traj
    show_arrow = not args.no_arrow
    show_service = not args.no_service

    if args.mode in ["init", "both"]:
        plot_episode_init(
            episode_row,
            save_path=None,
            show_labels=show_labels,
        )

    if args.mode in ["animate", "both"]:
        animate_episode(
            episode_row,
            step_episode_df,
            interval_ms=args.interval,
            save_path=args.save,
            show_labels=show_labels,
            show_traj=show_traj,
            show_arrow=show_arrow,
            show_service=show_service,
            traj_window=args.traj_window,
            select_base=args.select_base,
        )


if __name__ == "__main__":
    main()
