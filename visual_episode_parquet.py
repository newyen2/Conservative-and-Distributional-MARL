# visual_episode_parquet.py
# 用途：
#   讀取 test.parquet，將每個 episode 的 init_UAV_pos、init_device_pos、risk_region 畫出來
#
# 支援座標格式：
#   1. [x1, y1, x2, y2, ...]
#   2. [[x1, y1], [x2, y2], ...]
#   3. [array([x1, y1]), array([x2, y2]), ...]
#   4. [x1, y1, array([x2, y2]), ..., np.float64(xn), np.float64(yn)]
#
# 安裝需求：
#   pip install pandas pyarrow matplotlib numpy
#
# 使用範例：
#   python visual_episode_parquet.py --parquet test.parquet
#   python visual_episode_parquet.py --parquet test.parquet --episode 5
#   python visual_episode_parquet.py --parquet test.parquet --row 10
#   python visual_episode_parquet.py --parquet test.parquet --all --save_dir output_figs

import argparse
import ast
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


# ============================================================
# 基礎轉換工具
# ============================================================

def is_scalar_number(x):
    """
    判斷 x 是否為純量數值
    支援 Python int/float 與 numpy integer/floating
    """
    return isinstance(x, (int, float, np.integer, np.floating))


def extract_numbers_from_string(s):
    """
    從字串中提取所有數值
    可處理：
      "[1, 2, 3, 4]"
      "array([3.2, 1.4])"
      "np.float64(3.0)"
    """
    pattern = r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?"
    return [float(x) for x in re.findall(pattern, s)]


def to_python_obj(obj):
    """
    將常見物件轉成 Python 原生型別
    目標是避免 pandas / pyarrow / numpy 讀出來後含有 ndarray 或 numpy scalar
    """
    if obj is None:
        return None

    # numpy array -> list
    if isinstance(obj, np.ndarray):
        return obj.tolist()

    # numpy scalar -> Python scalar
    if isinstance(obj, np.generic):
        return obj.item()

    # tuple -> list
    if isinstance(obj, tuple):
        return [to_python_obj(x) for x in obj]

    # list -> recursive list
    if isinstance(obj, list):
        return [to_python_obj(x) for x in obj]

    # string -> 優先 literal_eval，失敗則嘗試提取數字
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


# ============================================================
# 座標解析
# ============================================================

def flatten_one_level_if_needed(coords):
    """
    有些 parquet 讀回後，可能會多包一層：
      [[[1, 2], [3, 4]]]
    這種情況可以拆成：
      [[1, 2], [3, 4]]

    但只會在明顯只有一層包裝時處理，避免誤拆正常資料
    """
    coords = to_python_obj(coords)

    if (
        isinstance(coords, list)
        and len(coords) == 1
        and isinstance(coords[0], list)
        and not (
            len(coords[0]) == 2
            and all(is_scalar_number(v) for v in coords[0])
        )
    ):
        return coords[0]

    return coords


def flat_to_points(coords, name="coords"):
    """
    將各種可能格式轉為：
      [(x1, y1), (x2, y2), ...]

    支援：
      [x1, y1, x2, y2, ...]
      [[x1, y1], [x2, y2], ...]
      [np.array([x1, y1]), np.array([x2, y2]), ...]
      [x1, y1, np.array([x2, y2]), ..., xN, yN]
    """
    coords = flatten_one_level_if_needed(coords)

    if coords is None:
        return []

    if not isinstance(coords, list):
        coords = to_python_obj(coords)

    if not isinstance(coords, list):
        raise ValueError(f"{name} 無法解析為 list，目前 type={type(coords)}, value={coords}")

    points = []
    pending_scalar = None

    for item in coords:
        item = to_python_obj(item)

        # 情況 A：item 是 [x, y]
        if (
            isinstance(item, list)
            and len(item) == 2
            and is_scalar_number(item[0])
            and is_scalar_number(item[1])
        ):
            points.append((float(item[0]), float(item[1])))
            continue

        # 情況 B：item 是 np.array/list，但不是 [x, y]
        # 例如 item 是 [x1, y1, x2, y2]，則遞迴解析後加入
        if isinstance(item, list):
            sub_points = flat_to_points(item, name=name)
            points.extend(sub_points)
            continue

        # 情況 C：item 是純量，需要和下一個純量配對
        if is_scalar_number(item):
            if pending_scalar is None:
                pending_scalar = float(item)
            else:
                points.append((pending_scalar, float(item)))
                pending_scalar = None
            continue

        # 情況 D：item 是字串，嘗試抽數字
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

    if pending_scalar is not None:
        raise ValueError(f"{name} 座標數量不成對，最後剩下一個 x 沒有 y：{pending_scalar}")

    return points


def parse_risk_region(region):
    """
    將 risk_region 轉成矩形資訊：
      [x1, y1, x2, y2] -> (left, bottom, width, height)

    若 risk_region 被存成 [[x1, y1], [x2, y2]] 也可以處理
    """
    region = to_python_obj(region)

    if region is None:
        raise ValueError("risk_region 是 None")

    # 若是 [[x1, y1], [x2, y2]]
    if (
        isinstance(region, list)
        and len(region) == 2
        and all(isinstance(v, list) for v in region)
        and all(len(v) == 2 for v in region)
    ):
        nums = [region[0][0], region[0][1], region[1][0], region[1][1]]
    else:
        nums = to_python_obj(region)

        # 若是字串或奇怪格式，直接抽數字
        if isinstance(nums, str):
            nums = extract_numbers_from_string(nums)

        # 若還有巢狀，攤平成數字
        nums = flatten_numbers(nums)

    if len(nums) != 4:
        raise ValueError(f"risk_region 格式錯誤，應為 [x1, y1, x2, y2]，目前解析結果為：{nums}")

    x1, y1, x2, y2 = [float(v) for v in nums]

    left = min(x1, x2)
    bottom = min(y1, y2)
    width = abs(x2 - x1)
    height = abs(y2 - y1)

    return left, bottom, width, height


def flatten_numbers(obj):
    """
    將任意 list / ndarray / scalar / string 攤平成數字 list
    """
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
# 繪圖
# ============================================================

def plot_one_episode(row, row_idx=None, save_path=None, show_id=True):
    """
    繪製單一 episode
    """
    episode_value = row["episode"] if "episode" in row.index else row_idx

    uav_points = flat_to_points(row["init_UAV_pos"], name="init_UAV_pos")
    device_points = flat_to_points(row["init_device_pos"], name="init_device_pos")
    left, bottom, width, height = parse_risk_region(row["risk_region"])

    fig, ax = plt.subplots(figsize=(8, 8))

    # Risk region
    rect = Rectangle(
        (left, bottom),
        width,
        height,
        fill=True,
        alpha=0.25,
        edgecolor="red",
        facecolor="red",
        linewidth=2,
        label="Risk Region",
    )
    ax.add_patch(rect)

    # UAV points
    if uav_points:
        uav_x = [float(p[0]) for p in uav_points]
        uav_y = [float(p[1]) for p in uav_points]
        ax.scatter(uav_x, uav_y, s=140, marker="^", label="UAV")

        if show_id:
            for idx, (x, y) in enumerate(uav_points, start=1):
                ax.text(float(x), float(y), f" UAV_{idx}", fontsize=10, va="bottom")

    # Device points
    if device_points:
        dev_x = [float(p[0]) for p in device_points]
        dev_y = [float(p[1]) for p in device_points]
        ax.scatter(dev_x, dev_y, s=70, marker="o", label="Device")

        if show_id:
            for idx, (x, y) in enumerate(device_points, start=1):
                ax.text(float(x), float(y), f" D_{idx}", fontsize=9, va="bottom")

    ax.set_title(f"Episode {episode_value}")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # 自動調整顯示範圍
    all_x = (
        [float(p[0]) for p in uav_points]
        + [float(p[0]) for p in device_points]
        + [left, left + width]
    )
    all_y = (
        [float(p[1]) for p in uav_points]
        + [float(p[1]) for p in device_points]
        + [bottom, bottom + height]
    )

    if all_x and all_y:
        margin = 1.0
        ax.set_xlim(0 - margin, 10 + margin)
        ax.set_ylim(0 - margin, 10 + margin)

    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200)
        print(f"已儲存圖片：{save_path}")
        plt.close(fig)
    else:
        plt.show()


def preview_row(row):
    """
    印出該 row 解析後的座標，方便 debug
    """
    print("========== Preview Parsed Row ==========")

    if "episode" in row.index:
        print(f"episode: {row['episode']}")

    print("\ninit_UAV_pos raw:")
    print(row["init_UAV_pos"])
    print("init_UAV_pos parsed:")
    print(flat_to_points(row["init_UAV_pos"], name="init_UAV_pos"))

    print("\ninit_device_pos raw:")
    print(row["init_device_pos"])
    print("init_device_pos parsed:")
    print(flat_to_points(row["init_device_pos"], name="init_device_pos"))

    print("\nrisk_region raw:")
    print(row["risk_region"])
    print("risk_region parsed:")
    print(parse_risk_region(row["risk_region"]))

    print("========================================")


# ============================================================
# 主程式
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--parquet",
        type=str,
        default="test.parquet",
        help="Parquet 檔案路徑，預設為 test.parquet",
    )

    parser.add_argument(
        "--episode",
        type=int,
        default=None,
        help="指定要顯示哪個 episode 值",
    )

    parser.add_argument(
        "--row",
        type=int,
        default=None,
        help="指定要顯示 DataFrame 的第幾列，從 0 開始",
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help="輸出全部 episodes",
    )

    parser.add_argument(
        "--save_dir",
        type=str,
        default=None,
        help="若指定，圖片會儲存到該資料夾；若不指定則直接顯示視窗",
    )

    parser.add_argument(
        "--no_id",
        action="store_true",
        help="不顯示 UAV / Device 的文字編號",
    )

    parser.add_argument(
        "--preview",
        action="store_true",
        help="只印出解析結果，不畫圖",
    )

    args = parser.parse_args()

    parquet_path = Path(args.parquet)

    if not parquet_path.exists():
        raise FileNotFoundError(f"找不到 parquet 檔案：{parquet_path}")

    df = pd.read_parquet(parquet_path, engine="pyarrow")

    required_cols = ["init_UAV_pos", "init_device_pos", "risk_region"]
    for col in required_cols:
        if col not in df.columns:
            raise KeyError(f"缺少必要欄位：{col}")

    print(f"成功讀取：{parquet_path}")
    print(f"資料筆數：{len(df)}")
    print(f"欄位：{list(df.columns)}")

    show_id = not args.no_id

    # 輸出全部 episodes
    if args.all:
        for idx, row in df.iterrows():
            episode_value = row["episode"] if "episode" in df.columns else idx

            if args.preview:
                preview_row(row)
                continue

            save_path = None
            if args.save_dir is not None:
                save_path = Path(args.save_dir) / f"episode_{episode_value}.png"

            plot_one_episode(
                row,
                row_idx=idx,
                save_path=save_path,
                show_id=show_id,
            )

        return

    # 根據 episode 欄位選擇
    if args.episode is not None:
        if "episode" not in df.columns:
            raise KeyError("DataFrame 中沒有 episode 欄位，無法使用 --episode")

        matched = df[df["episode"] == args.episode]

        if len(matched) == 0:
            raise ValueError(f"找不到 episode = {args.episode}")

        row = matched.iloc[0]

        if args.preview:
            preview_row(row)
            return

        save_path = None
        if args.save_dir is not None:
            save_path = Path(args.save_dir) / f"episode_{args.episode}.png"

        plot_one_episode(row, save_path=save_path, show_id=show_id)
        return

    # 根據 row index 選擇
    if args.row is not None:
        if args.row < 0 or args.row >= len(df):
            raise IndexError(f"row 超出範圍，合法範圍為 0 ~ {len(df) - 1}")

        row = df.iloc[args.row]

        if args.preview:
            preview_row(row)
            return

        save_path = None
        if args.save_dir is not None:
            episode_value = row["episode"] if "episode" in df.columns else args.row
            save_path = Path(args.save_dir) / f"episode_{episode_value}.png"

        plot_one_episode(row, row_idx=args.row, save_path=save_path, show_id=show_id)
        return

    # 預設畫第 0 列
    row = df.iloc[1]

    if args.preview:
        preview_row(row)
        return

    save_path = None
    if args.save_dir is not None:
        episode_value = row["episode"] if "episode" in df.columns else 0
        save_path = Path(args.save_dir) / f"episode_{episode_value}.png"

    plot_one_episode(row, row_idx=0, save_path=save_path, show_id=show_id)


if __name__ == "__main__":
    main()
