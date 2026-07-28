"""HSAC MVP 的端到端 smoke test。"""

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np
import torch

from HSAC_Online import HSACOnlineTrainer, Train_HSAC_Online


def build_smoke_config(output_dir):
    """建立能快速涵蓋 warm-up、訓練、評估與存檔的設定。"""
    return {
        "M": 10,
        "U": 2,
        "episodes": 2,
        "max_steps": 3,
        "max_mov": 2.0,
        "aoi_max": 100,
        "energy_weight": 500.0,
        "risk_penalty": 300.0,
        "risk_probability": 0.1,
        "high_interval": 2,
        "warmup_steps": 4,
        "epsilon_start": 1.0,
        "epsilon_end": 0.1,
        "epsilon_decay_steps": 10,
        "low_buffer_size": 64,
        "high_buffer_size": 64,
        "low_batch_size": 2,
        "high_batch_size": 1,
        "low_update_frequency": 1,
        "high_update_frequency": 1,
        "updates_per_step": 1,
        "hidden_size": 32,
        "eval_period": 1,
        "eval_episodes": 2,
        "train_seed": 101,
        "eval_seed": 202,
        "output_dir": str(output_dir),
        "quiet": True,
    }


def run_smoke_test(device):
    """執行完整 MVP 並驗證 checkpoint 可重現 evaluation。"""
    with tempfile.TemporaryDirectory(
        prefix="hsac_mvp_"
    ) as temp_directory:
        output_dir = Path(temp_directory)
        config = build_smoke_config(output_dir)

        trainer, result = Train_HSAC_Online(
            config=config,
            device=device,
        )
        output_files = {
            name: Path(path)
            for name, path in result["files"].items()
        }
        missing_files = [
            name
            for name, path in output_files.items()
            if not path.is_file() or path.stat().st_size == 0
        ]
        if missing_files:
            raise RuntimeError(
                f"MVP 輸出檔案缺失：{missing_files}"
            )

        # Checkpoint 保存後的第一次 evaluation，應可由載入後 RNG 完整重現。
        expected_eval = trainer.evaluate(2)
        expected_parameters = [
            next(agent.high_network.parameters())
            .detach()
            .cpu()
            .clone()
            for agent in trainer.agents
        ]

        loaded_trainer = HSACOnlineTrainer(
            config=config,
            device=device,
        )
        load_info = loaded_trainer.load_checkpoint(
            output_files["checkpoint"]
        )
        loaded_eval = loaded_trainer.evaluate(2)

        for expected, loaded_agent in zip(
            expected_parameters,
            loaded_trainer.agents,
        ):
            loaded_parameter = (
                next(loaded_agent.high_network.parameters())
                .detach()
                .cpu()
            )
            if not torch.equal(expected, loaded_parameter):
                raise RuntimeError(
                    "Checkpoint 載入後的網路參數不一致。"
                )

        if not np.array_equal(
            np.asarray(expected_eval["episode_returns"]),
            np.asarray(loaded_eval["episode_returns"]),
        ):
            raise RuntimeError(
                "Checkpoint 載入前後的 evaluation 不一致。"
            )
        if load_info["global_step"] != trainer.global_step:
            raise RuntimeError(
                "Checkpoint 未正確恢復 global_step。"
            )

        summary = {
            "passed": True,
            "device": str(trainer.device),
            "global_step": trainer.global_step,
            "episode_logs": len(trainer.episode_logs),
            "step_logs": len(trainer.step_logs),
            "eval_logs": len(trainer.eval_logs),
            "low_buffer_sizes": [
                len(buffer) for buffer in trainer.low_buffers
            ],
            "high_buffer_sizes": [
                len(buffer) for buffer in trainer.high_buffers
            ],
            "evaluation_return": expected_eval[
                "mean_return"
            ],
            "checkpoint_reload_match": True,
            "output_files": sorted(output_files),
        }
        print(
            json.dumps(
                summary,
                ensure_ascii=False,
                indent=2,
            )
        )
        return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the HSAC MVP smoke test."
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=("cpu", "cuda"),
    )
    smoke_args = parser.parse_args()
    run_smoke_test(smoke_args.device)
