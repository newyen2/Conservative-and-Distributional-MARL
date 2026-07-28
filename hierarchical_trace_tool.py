"""階層式 HSAC 軌跡的互動回放工具。

本工具直接讀取 HSAC_Online.py 輸出的 hierarchical_step_logs.json，
並顯示 UAV 軌跡、High-level goal、服務連線、AOI、功耗與風險事件。
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from matplotlib.widgets import Button, Slider


DEFAULT_STEP_LOG = "Stored_Datas/hierarchical_step_logs.json"
DEFAULT_MAP_BOUNDARY = (0.0, 0.0, 10.0, 10.0)
DEFAULT_RISK_REGIONS = ((3.0, 2.0, 7.0, 7.0),)

REQUIRED_STEP_FIELDS = (
    "episode",
    "episode_step",
    "state",
    "goals",
    "move_actions",
    "service_actions",
    "rewards",
    "uav_locations",
    "device_aoi",
    "power",
    "mean_aoi",
    "risk_triggered",
    "movement_distance",
)


def load_step_logs(path):
    """讀取階層式 step log，並檢查回放所需欄位。"""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"找不到階層式 step log：{path}")

    with path.open("r", encoding="utf-8") as log_file:
        payload = json.load(log_file)

    # 同時支援純 list，以及未來可能包在 step_logs 欄位中的格式。
    step_logs = (
        payload.get("step_logs")
        if isinstance(payload, dict)
        else payload
    )
    if not isinstance(step_logs, list) or not step_logs:
        raise ValueError("step log 必須是非空的 JSON list。")

    missing = [
        field
        for field in REQUIRED_STEP_FIELDS
        if field not in step_logs[0]
    ]
    if missing:
        raise KeyError(
            "step log 缺少回放所需欄位："
            + ", ".join(missing)
        )

    return step_logs


def available_episodes(step_logs):
    """取得 log 中所有可回放的 episode 編號。"""
    return sorted({int(record["episode"]) for record in step_logs})


def get_episode_steps(step_logs, episode):
    """擷取指定 episode，並依 episode_step 排序。"""
    records = [
        record
        for record in step_logs
        if int(record["episode"]) == int(episode)
    ]
    if not records:
        raise ValueError(f"step log 中找不到 episode = {episode}")

    records.sort(key=lambda record: int(record["episode_step"]))
    return records


def parse_initial_state(first_record):
    """從第一筆 transition 的 state 還原 episode 初始場景。"""
    num_uavs = len(first_record["uav_locations"])
    num_devices = len(first_record["device_aoi"])
    state = np.asarray(first_record["state"], dtype=np.float32)

    uav_end = num_uavs * 2
    aoi_end = uav_end + num_devices
    device_end = aoi_end + num_devices * 2
    if state.size < device_end:
        raise ValueError(
            "state 維度不足，無法還原 UAV、AOI 與裝置座標。"
        )

    return {
        "uav_locations": state[:uav_end].reshape(num_uavs, 2),
        "device_aoi": state[uav_end:aoi_end].astype(np.int32),
        "device_coord": state[aoi_end:device_end].reshape(
            num_devices,
            2,
        ),
    }


class HierarchicalEpisodeViewer:
    """以 Matplotlib 按鈕、滑桿與計時器回放單一 episode。"""

    def __init__(
        self,
        episode,
        step_logs,
        interval=500,
        display="all",
        trajectory=10,
        map_boundary=DEFAULT_MAP_BOUNDARY,
        risk_regions=DEFAULT_RISK_REGIONS,
    ):
        self.episode = int(episode)
        self.records = get_episode_steps(step_logs, self.episode)
        self.initial = parse_initial_state(self.records[0])
        self.num_frames = len(self.records)
        self.num_uavs = len(self.records[0]["uav_locations"])
        self.num_devices = len(self.records[0]["device_aoi"])

        self.interval = max(1, int(interval))
        self.display = str(display)
        self.trajectory = max(0, int(trajectory))
        self.map_boundary = np.asarray(
            map_boundary,
            dtype=np.float32,
        ).reshape(4)
        self.risk_regions = np.asarray(
            risk_regions,
            dtype=np.float32,
        ).reshape(-1, 4)

        self.frame = 0
        self.playing = False
        self._updating_slider = False
        self.team_rewards = np.asarray(
            [
                float(np.mean(record["rewards"]))
                for record in self.records
            ],
            dtype=np.float64,
        )
        self.cumulative_returns = np.cumsum(self.team_rewards)

        # 主地圖置左，右側保留當前 step 的階層決策與環境資訊。
        self.fig, self.ax = plt.subplots(figsize=(11, 7))
        self.fig.subplots_adjust(
            left=0.08,
            right=0.75,
            bottom=0.24,
            top=0.93,
        )
        self.info_ax = self.fig.add_axes([0.78, 0.18, 0.20, 0.49])
        self.info_ax.axis("off")
        self.figure_legend = None

        self.prev_ax = self.fig.add_axes([0.20, 0.05, 0.12, 0.06])
        self.play_ax = self.fig.add_axes([0.36, 0.05, 0.12, 0.06])
        self.next_ax = self.fig.add_axes([0.52, 0.05, 0.12, 0.06])
        self.prev_button = Button(self.prev_ax, "< Previous")
        self.play_button = Button(self.play_ax, "Play")
        self.next_button = Button(self.next_ax, "Next >")

        self.slider_ax = self.fig.add_axes([0.16, 0.14, 0.52, 0.03])
        self.step_slider = Slider(
            self.slider_ax,
            "Step",
            0,
            self.num_frames,
            valinit=0,
            valstep=1,
        )

        self.prev_button.on_clicked(self.on_previous)
        self.play_button.on_clicked(self.on_play_pause)
        self.next_button.on_clicked(self.on_next)
        self.step_slider.on_changed(self.on_slider_changed)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key_press)
        self.fig.canvas.mpl_connect("close_event", self.on_close)

        self.timer = self.fig.canvas.new_timer(interval=self.interval)
        self.timer.add_callback(self.on_timer)
        self.draw_frame()

    def draw_frame(self):
        """依目前 frame 重畫地圖、階層動作與資訊欄。"""
        if self.frame == 0:
            uav_locations = self.initial["uav_locations"]
            device_aoi = self.initial["device_aoi"]
            record = None
        else:
            record = self.records[self.frame - 1]
            uav_locations = np.asarray(
                record["uav_locations"],
                dtype=np.float32,
            )
            device_aoi = np.asarray(
                record["device_aoi"],
                dtype=np.int32,
            )

        self.draw_common_scene(
            uav_locations=uav_locations,
            device_aoi=device_aoi,
        )

        if record is not None:
            if self.display in ("all", "move-only"):
                self.draw_trajectory()
                self.draw_high_goals(
                    uav_locations,
                    record["goals"],
                )
            elif self.display == "action-only":
                self.draw_high_goals(
                    uav_locations,
                    record["goals"],
                )

            if self.display in ("all", "action-only"):
                self.draw_service_links(
                    uav_locations,
                    record["service_actions"],
                )

            self.draw_risk_events(
                uav_locations,
                record["risk_triggered"],
            )

        title = (
            f"Episode {self.episode} | "
            f"Step {self.frame}/{self.num_frames}"
        )
        self.ax.set_title(title)
        self.draw_info_panel(record)
        self._sync_slider()
        self.fig.canvas.draw_idle()

    def draw_common_scene(self, uav_locations, device_aoi):
        """繪製地圖邊界、風險區、裝置及 UAV 當前位置。"""
        self.ax.clear()
        xmin, ymin, xmax, ymax = self.map_boundary
        self.ax.add_patch(
            Rectangle(
                (xmin, ymin),
                xmax - xmin,
                ymax - ymin,
                fill=False,
                alpha=0.65,
                linewidth=2,
                color="black",
                label="Map boundary",
            )
        )

        for index, (x1, y1, x2, y2) in enumerate(
            self.risk_regions
        ):
            self.ax.add_patch(
                Rectangle(
                    (x1, y1),
                    x2 - x1,
                    y2 - y1,
                    fill=True,
                    alpha=0.15,
                    linewidth=2,
                    color="red",
                    label="Risk region" if index == 0 else None,
                )
            )

        device_coord = self.initial["device_coord"]
        clipped_aoi = np.clip(device_aoi, 0, 100)
        device_sizes = 30.0 + 170.0 * clipped_aoi / 100.0
        self.ax.scatter(
            device_coord[:, 0],
            device_coord[:, 1],
            s=device_sizes,
            marker="o",
            alpha=0.9,
            color="tab:green",
            label="Device (size = AOI)",
            zorder=3,
        )
        for device, ((x, y), aoi) in enumerate(
            zip(device_coord, device_aoi)
        ):
            self.ax.text(
                float(x),
                float(y) - 0.18,
                f"D{device}",
                fontsize=9,
                ha="center",
                va="top",
                zorder=6,
            )
            self.ax.text(
                float(x),
                float(y),
                str(int(aoi)),
                fontsize=7,
                ha="center",
                va="center",
                color="white",
                zorder=6,
            )

        for uav, (x, y) in enumerate(uav_locations):
            color = f"C{uav % 10}"
            self.ax.scatter(
                [x],
                [y],
                s=75,
                marker="s",
                color=color,
                label=f"UAV {uav}",
                zorder=7,
            )
            self.ax.text(
                float(x),
                float(y) - 0.22,
                f"UAV{uav}",
                fontsize=9,
                ha="center",
                va="top",
                zorder=8,
            )

        padding = 0.8
        self.ax.set_xlim(xmin - padding, xmax + padding)
        self.ax.set_ylim(ymin - padding, ymax + padding)
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_aspect("equal", adjustable="box")
        self.ax.grid(True, alpha=0.25)

    def draw_trajectory(self):
        """顯示最近 N steps 的實際 UAV 軌跡，並加粗最新位移。"""
        all_locations = [self.initial["uav_locations"]]
        all_locations.extend(
            np.asarray(
                record["uav_locations"],
                dtype=np.float32,
            )
            for record in self.records[: self.frame]
        )

        if self.trajectory > 0:
            all_locations = all_locations[
                -(self.trajectory + 1):
            ]
        locations = np.asarray(all_locations, dtype=np.float32)

        for uav in range(self.num_uavs):
            color = f"C{uav % 10}"
            points = locations[:, uav, :]
            if len(points) >= 2:
                self.ax.plot(
                    points[:, 0],
                    points[:, 1],
                    linewidth=2,
                    alpha=0.35,
                    color=color,
                    label=f"UAV {uav} trajectory",
                    zorder=2,
                )
                self.ax.plot(
                    points[-2:, 0],
                    points[-2:, 1],
                    linewidth=4,
                    alpha=0.9,
                    color=color,
                    zorder=4,
                )

    def draw_high_goals(self, uav_locations, goals):
        """以點線連接 UAV 與本 segment 的 High-level goal。"""
        device_coord = self.initial["device_coord"]
        for uav, goal in enumerate(goals):
            goal = int(goal)
            if not 0 <= goal < self.num_devices:
                continue

            color = f"C{uav % 10}"
            target = device_coord[goal]
            self.ax.plot(
                [uav_locations[uav, 0], target[0]],
                [uav_locations[uav, 1], target[1]],
                linewidth=1.5,
                linestyle=":",
                alpha=0.75,
                color=color,
                label=(
                    "High-level goal"
                    if uav == 0
                    else None
                ),
                zorder=1,
            )
            self.ax.scatter(
                [target[0]],
                [target[1]],
                s=250,
                marker="o",
                facecolors="none",
                edgecolors=color,
                linewidths=2,
                zorder=5,
            )

    def draw_service_links(self, uav_locations, service_actions):
        """以虛線顯示本 step 實際服務的裝置；M 代表 idle。"""
        device_coord = self.initial["device_coord"]
        service_label_added = False
        for uav, action in enumerate(service_actions):
            action = int(action)
            if not 0 <= action < self.num_devices:
                continue

            target = device_coord[action]
            self.ax.plot(
                [uav_locations[uav, 0], target[0]],
                [uav_locations[uav, 1], target[1]],
                linewidth=2,
                linestyle="--",
                alpha=0.55,
                color="black",
                label=(
                    "Service link"
                    if not service_label_added
                    else None
                ),
                zorder=1,
            )
            service_label_added = True

    def draw_risk_events(self, uav_locations, risk_triggered):
        """在本 step 觸發風險的 UAV 外加上紅色警示圈。"""
        for uav, triggered in enumerate(risk_triggered):
            if not int(triggered):
                continue
            x, y = uav_locations[uav]
            self.ax.scatter(
                [x],
                [y],
                s=260,
                marker="o",
                facecolors="none",
                edgecolors="red",
                linewidths=2.5,
                label="Risk triggered",
                zorder=9,
            )

    def draw_info_panel(self, record):
        """更新右側的 reward、goal、action、power 與移動資訊。"""
        self.info_ax.clear()
        self.info_ax.axis("off")
        if record is None:
            lines = [
                f"Episode: {self.episode}",
                "Step: 0 (initial state)",
                "",
                f"Mean AOI: {np.mean(self.initial['device_aoi']):.2f}",
                f"UAVs: {self.num_uavs}",
                f"Devices: {self.num_devices}",
            ]
        else:
            rewards = np.asarray(record["rewards"], dtype=np.float64)
            powers = np.asarray(record["power"], dtype=np.float64)
            move_actions = np.asarray(
                record["move_actions"],
                dtype=np.float64,
            )
            move_norms = np.linalg.norm(move_actions, axis=1)
            movement = np.asarray(
                record["movement_distance"],
                dtype=np.float64,
            )
            goals = np.asarray(record["goals"], dtype=np.int64)
            services = np.asarray(
                record["service_actions"],
                dtype=np.int64,
            )
            risks = np.asarray(
                record["risk_triggered"],
                dtype=np.int64,
            )
            refresh = np.asarray(
                record.get(
                    "goal_refresh_mask",
                    [False] * self.num_uavs,
                ),
                dtype=np.bool_,
            )

            lines = [
                f"Episode: {self.episode}",
                f"Step: {self.frame}/{self.num_frames}",
                f"Global step: {record.get('global_step', '-')}",
                "",
                f"Team reward: {np.mean(rewards):.3f}",
                f"Cumulative: {self.cumulative_returns[self.frame - 1]:.3f}",
                f"Mean AOI: {float(record['mean_aoi']):.2f}",
                f"Total power: {float(np.sum(powers)):.5f}",
                "",
            ]
            for uav in range(self.num_uavs):
                service_text = (
                    "idle"
                    if services[uav] == self.num_devices
                    else f"D{services[uav]}"
                )
                goal_mark = " *refresh" if refresh[uav] else ""
                lines.extend(
                    [
                        f"UAV {uav}{goal_mark}",
                        f"  goal: D{goals[uav]}",
                        f"  service: {service_text}",
                        f"  reward: {rewards[uav]:.3f}",
                        f"  power: {powers[uav]:.5f}",
                        (
                            "  move: "
                            f"{move_norms[uav]:.3f}"
                            f" -> {movement[uav]:.3f}"
                        ),
                        f"  risk: {risks[uav]}",
                        "",
                    ]
                )

        self.info_ax.text(
            0.0,
            1.0,
            "\n".join(lines),
            ha="left",
            va="top",
            fontsize=10,
            family="monospace",
            transform=self.info_ax.transAxes,
        )

        handles, labels = self.ax.get_legend_handles_labels()
        unique = {}
        for handle, label in zip(handles, labels):
            if label and label not in unique:
                unique[label] = handle
        if self.figure_legend is not None:
            self.figure_legend.remove()
        self.figure_legend = self.fig.legend(
            unique.values(),
            unique.keys(),
            loc="upper left",
            bbox_to_anchor=(0.77, 0.93),
            fontsize=8,
        )

    def set_frame(self, frame, pause=True):
        """切換至指定 frame，必要時停止自動播放。"""
        if pause:
            self.stop_playback()
        self.frame = int(np.clip(frame, 0, self.num_frames))
        self.draw_frame()

    def stop_playback(self):
        """停止 timer 並恢復 Play 按鈕文字。"""
        self.playing = False
        self.timer.stop()
        self.play_button.label.set_text("Play")

    def on_previous(self, _event):
        self.set_frame(self.frame - 1)

    def on_next(self, _event):
        self.set_frame(self.frame + 1)

    def on_play_pause(self, _event):
        if self.playing:
            self.stop_playback()
            return

        if self.frame >= self.num_frames:
            self.frame = 0
            self.draw_frame()
        self.playing = True
        self.play_button.label.set_text("Pause")
        self.timer.start()

    def on_timer(self):
        if not self.playing:
            return False
        if self.frame >= self.num_frames:
            self.stop_playback()
            self.play_button.label.set_text("Replay")
            return False

        self.frame += 1
        self.draw_frame()
        return True

    def on_slider_changed(self, value):
        if self._updating_slider:
            return
        self.set_frame(int(value))

    def _sync_slider(self):
        if int(self.step_slider.val) == self.frame:
            return
        self._updating_slider = True
        self.step_slider.set_val(self.frame)
        self._updating_slider = False

    def on_key_press(self, event):
        """支援鍵盤方向鍵、空白鍵、Home 與 End。"""
        if event.key == "left":
            self.set_frame(self.frame - 1)
        elif event.key == "right":
            self.set_frame(self.frame + 1)
        elif event.key == "home":
            self.set_frame(0)
        elif event.key == "end":
            self.set_frame(self.num_frames)
        elif event.key == " ":
            self.on_play_pause(event)

    def on_close(self, _event):
        self.stop_playback()

    def show(self):
        """開啟 Matplotlib 互動視窗。"""
        plt.show()


def build_parser():
    """建立與舊 trace_tool.py 相近的命令列介面。"""
    parser = argparse.ArgumentParser(
        description="回放階層式 HSAC 的 UAV 軌跡與決策。",
    )
    parser.add_argument(
        "--step-log",
        type=str,
        default=DEFAULT_STEP_LOG,
        help=(
            "hierarchical_step_logs.json 路徑，預設為 "
            f"{DEFAULT_STEP_LOG}"
        ),
    )
    parser.add_argument(
        "--episode",
        type=int,
        default=None,
        help="要回放的 episode；未提供時使用 log 中最後一個 episode。",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=500,
        help="自動播放每幀間隔，單位 ms，預設 500。",
    )
    parser.add_argument(
        "--display",
        type=str,
        default="all",
        choices=("all", "move-only", "action-only", "static"),
        help=(
            "all: 全部；move-only: 軌跡與 goal；"
            "action-only: goal 與服務；static: 僅狀態。"
        ),
    )
    parser.add_argument(
        "--trajectory",
        type=int,
        default=10,
        help="保留最近幾個 step 的軌跡；0 代表顯示完整軌跡。",
    )
    parser.add_argument(
        "--map-boundary",
        type=float,
        nargs=4,
        default=DEFAULT_MAP_BOUNDARY,
        metavar=("XMIN", "YMIN", "XMAX", "YMAX"),
        help="地圖邊界，預設 0 0 10 10。",
    )
    parser.add_argument(
        "--risk-region",
        type=float,
        nargs=4,
        action="append",
        metavar=("XMIN", "YMIN", "XMAX", "YMAX"),
        help=(
            "風險區，可重複提供多次；"
            "未提供時使用目前 MVP 的 3 2 7 7。"
        ),
    )
    return parser


def main(args=None):
    """解析參數、載入 episode 並啟動回放視窗。"""
    parser = build_parser()
    options = parser.parse_args(args)
    step_logs = load_step_logs(options.step_log)
    episodes = available_episodes(step_logs)
    episode = episodes[-1] if options.episode is None else options.episode

    risk_regions = (
        DEFAULT_RISK_REGIONS
        if options.risk_region is None
        else options.risk_region
    )
    viewer = HierarchicalEpisodeViewer(
        episode=episode,
        step_logs=step_logs,
        interval=options.interval,
        display=options.display,
        trajectory=options.trajectory,
        map_boundary=options.map_boundary,
        risk_regions=risk_regions,
    )
    viewer.show()
    return viewer


if __name__ == "__main__":
    main()
