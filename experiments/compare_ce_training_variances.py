"""Paired comparison of plug-in and tangent training in the conditional CE pilot."""
from __future__ import annotations

if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "tmp/mpl_ce_conditional"))
from experiments import run_ce_conditional_fit_pilot as pilot
import numpy as np
import pandas as pd
import scipy
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plugin-dir", type=Path, default=ROOT / "output/ce_conditional_fit_pilot")
    parser.add_argument("--tangent-dir", type=Path, default=ROOT / "output/ce_conditional_tangent_fit")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "output/neural_training_comparison")
    args = parser.parse_args()
    sources = {"plugin":args.plugin_dir, "tangent":args.tangent_dir}
    settings = {name:json.loads((folder / "settings.json").read_text(encoding="utf-8"))
                for name, folder in sources.items()}
    common_fields = ("replicates", "train_paths", "epochs", "training_steps", "substeps",
                     "horizon", "pilot_paths", "cdf_paths", "reference_size", "path_batch", "seed",
                     "truth_sigma", "teacher", "fit_width", "learning_rate", "weight_decay", "score",
                     "jax_version", "backend", "python")
    for field in common_fields:
        if settings["plugin"][field] != settings["tangent"][field]:
            raise ValueError(f"Unmatched comparison setting: {field}")
    if settings["plugin"]["train_variance"] != "plugin" or settings["tangent"]["train_variance"] != "tangent":
        raise ValueError("Expected the two specified training variance arms.")
    oracle_sources = {name:(folder / settings[name].get("shared_oracle_dir", ".")).resolve()
                      for name, folder in sources.items()}
    for name, folder in oracle_sources.items():
        if folder != sources[name].resolve():
            shared_settings = json.loads((folder / "settings.json").read_text(encoding="utf-8"))
            for field in common_fields:
                if settings[name][field] != shared_settings[field]:
                    raise ValueError(f"Unmatched shared oracle setting: {name}, {field}")
    shared_oracle = oracle_sources["plugin"] == oracle_sources["tangent"]
    specification = settings["plugin"]
    reps, counts = specification["replicates"], specification["train_paths"]
    expected_each = reps*(1+len(counts))
    frames = {name:pd.read_csv(folder / "results.csv") for name, folder in sources.items()}
    for name, frame in frames.items():
        if len(frame) != expected_each or settings[name].get("completed_comparisons") != expected_each:
            raise ValueError(f"Incomplete {name} run.")
        if frame.duplicated(["replicate", "training_paths"]).any():
            raise ValueError(f"Duplicate {name} records.")
    expected_keys = {(rep, n) for rep in range(reps) for n in [0]+counts}
    for name, frame in frames.items():
        if set(zip(frame.replicate, frame.training_paths)) != expected_keys:
            raise ValueError(f"Missing condition in {name}.")

    # Each arm's recorded oracle estimates must agree, including when files are shared.
    oracle_columns = [column for column in frames["plugin"].columns if column != "fitting_seconds"]
    pd.testing.assert_frame_equal(
        frames["plugin"].loc[frames["plugin"].training_paths == 0, oracle_columns].sort_values("replicate").reset_index(drop=True),
        frames["tangent"].loc[frames["tangent"].training_paths == 0, oracle_columns].sort_values("replicate").reset_index(drop=True),
        check_exact=True,
    )
    # Separate oracle files are checked for equality; shared files are counted once.
    max_prefix_error = max_normalization_error = max_oracle_error = 0.0
    for rep in range(reps):
        for n in [0]+counts:
            model_sources = oracle_sources if n == 0 else sources
            with np.load(model_sources["plugin"] / f"rep{rep:02d}_n{n}_model.npz") as left, np.load(
                    model_sources["tangent"] / f"rep{rep:02d}_n{n}_model.npz") as right:
                for field in ("prefix", "input_mean", "input_scale"):
                    err = float(np.max(np.abs(left[field]-right[field])))
                    np.testing.assert_array_equal(left[field], right[field])
                    if field == "prefix":
                        max_prefix_error = max(max_prefix_error, err)
                    else:
                        max_normalization_error = max(max_normalization_error, err)
        with np.load(oracle_sources["plugin"] / f"rep{rep:02d}_n0_scores.npz") as left, np.load(
                oracle_sources["tangent"] / f"rep{rep:02d}_n0_scores.npz") as right:
            for field in left.files:
                np.testing.assert_array_equal(left[field], right[field])
                max_oracle_error = max(max_oracle_error, float(np.max(np.abs(left[field]-right[field]))))
    with np.load(oracle_sources["plugin"] / "teacher.npz") as left, np.load(oracle_sources["tangent"] / "teacher.npz") as right:
        for field in left.files:
            np.testing.assert_array_equal(left[field], right[field])

    # Recompute one joint 95% family, rather than combine two separate 95% families.
    total_comparisons = reps*(1+2*len(counts))
    rows = []
    for name, frame in frames.items():
        for raw in frame.to_dict("records"):
            n, rep = int(raw["training_paths"]), int(raw["replicate"])
            if n == 0 and name == "tangent":
                continue
            score_source = oracle_sources[name] if n == 0 else sources[name]
            with np.load(score_source / f"rep{rep:02d}_n{n}_scores.npz") as values:
                estimate = pilot.compare_score_laws(values["held"], values["reference"],
                                                    specification["reference_size"], total_comparisons)
                if abs(estimate["conditional_rejection"]-raw["conditional_rejection"]) > 1e-12:
                    raise AssertionError("Point estimate changed during interval recalculation.")
                row = {**raw, **estimate, "train_variance":"oracle" if n == 0 else name,
                       "held_energy":float(values["held_features"][:, 2].mean()),
                       "reference_energy":float(values["reference_features"][:, 2].mean())}
            rows.append(row)
    results = pd.DataFrame(rows)
    target = float(results.nominal_mc_target.iloc[0])
    summary_rows = []
    for (n, variance), group in results.groupby(["training_paths", "train_variance"], sort=True):
        summary_rows.append({"training_paths":int(n), "train_variance":variance, "analyses":len(group),
            "gap_median":group.cdf_gap.median(), "gap_min":group.cdf_gap.min(), "gap_max":group.cdf_gap.max(),
            "mean_conditional_rejection":group.conditional_rejection.mean(),
            "rejection_min":group.conditional_rejection.min(), "rejection_max":group.conditional_rejection.max(),
            "mean_rejection_low":group.conditional_rejection_low.mean(),
            "mean_rejection_high":group.conditional_rejection_high.mean(),
            "above_target":int((group.conditional_rejection_low > target).sum()),
            "below_target":int((group.conditional_rejection_high < target).sum()),
            "mean_sigma":group.fitted_sigma.mean(),
            "mean_held_energy":group.held_energy.mean(), "mean_reference_energy":group.reference_energy.mean()})
    summary = pd.DataFrame(summary_rows)
    plugin = results.loc[results.train_variance == "plugin"].set_index(["replicate", "training_paths"])
    tangent = results.loc[results.train_variance == "tangent"].set_index(["replicate", "training_paths"])
    paired = pd.DataFrame(index=plugin.index)
    for metric, lower, upper in (("cdf_gap", "cdf_gap_low", "cdf_gap_high"),
                                 ("conditional_rejection", "conditional_rejection_low", "conditional_rejection_high")):
        paired[f"{metric}_plugin"] = plugin[metric]
        paired[f"{metric}_tangent"] = tangent[metric]
        paired[f"{metric}_change"] = tangent[metric]-plugin[metric]
        paired[f"{metric}_change_low"] = tangent[lower]-plugin[upper]
        paired[f"{metric}_change_high"] = tangent[upper]-plugin[lower]
    paired["absolute_size_error_change"] = abs(tangent.conditional_rejection-target)-abs(plugin.conditional_rejection-target)
    paired["sigma_plugin"], paired["sigma_tangent"] = plugin.fitted_sigma, tangent.fitted_sigma
    paired = paired.reset_index()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.out_dir / "combined_results.csv", index=False)
    summary.to_csv(args.out_dir / "summary.csv", index=False)
    paired.to_csv(args.out_dir / "paired_differences.csv", index=False)
    checks = {"common_settings":{field:specification[field] for field in common_fields},
              "max_prefix_difference":max_prefix_error, "max_normalization_difference":max_normalization_error,
              "max_oracle_array_difference":max_oracle_error, "teacher_parameters_equal":True,
              "oracle_arrays_shared":shared_oracle,
              "unique_conditional_comparisons":total_comparisons,
              "simultaneous_confidence":0.95,
              "library_versions":{"numpy":np.__version__, "scipy":scipy.__version__,
                                  "pandas":pd.__version__, "optax":pilot.optax.__version__,
                                  "matplotlib":matplotlib.__version__},
              "scope":"CDF and rejection-probability Monte Carlo error conditional on these realized analysis states"}
    (args.out_dir / "comparison_checks.json").write_text(json.dumps(checks, indent=2)+"\n", encoding="utf-8")

    plt.rcParams.update({"font.size":10, "axes.spines.top":False, "axes.spines.right":False})
    fig, axes = plt.subplots(2, 3, figsize=(12.8, 7.7), layout="constrained")
    metric_specs = (("cdf_gap", "cdf_gap_low", "cdf_gap_high", "Conditional score CDF gap"),
                    ("conditional_rejection", "conditional_rejection_low", "conditional_rejection_high", "Conditional rejection probability"),
                    ("fitted_sigma", None, None, "Fitted diffusion scale"))
    for i, n in enumerate(counts):
        p = plugin.xs(n, level="training_paths").sort_index()
        t = tangent.xs(n, level="training_paths").sort_index()
        jitter = np.linspace(-0.10, 0.10, reps)
        for ax, (metric, low, high, title) in zip(axes[i], metric_specs):
            for rep in range(reps):
                ax.plot([jitter[rep], 1+jitter[rep]], [p.loc[rep,metric], t.loc[rep,metric]],
                        color="#aeb8bd", alpha=.6, lw=.8, zorder=1)
            for x, group, color in ((0,p,"#80634a"),(1,t,"#176481")):
                yy = group[metric].to_numpy()
                if low:
                    ax.errorbar(x+jitter, yy, yerr=np.vstack([yy-group[low],group[high]-yy]),
                                fmt="none", ecolor=color, alpha=.35, lw=.7)
                ax.scatter(x+jitter, yy, s=22, color=color, zorder=3)
                ax.plot([x-.17,x+.17],[np.median(yy)]*2,color=color,lw=2.6)
            ax.set_xticks([0,1],["Plug-in training","Tangent training"], fontsize=9)
            ax.set_xlim(-.3,1.3)
            ax.set_title(f"{n} training paths: {title}",loc="left",fontsize=10)
            ax.grid(axis="y",alpha=.18)
            if metric == "conditional_rejection":
                ax.axhline(target,color="#a53e35",ls="--",lw=1,label="Nominal 5%")
                ax.legend(frameon=False,loc="upper right",fontsize=8)
                ax.set_ylim(bottom=0)
            elif metric == "cdf_gap":
                ax.set_ylim(bottom=0)
            else:
                ax.axhline(specification["truth_sigma"],color="#a53e35",ls="--",lw=1,label="True scale = 0.42")
                ax.legend(frameon=False,loc="upper left",fontsize=8)
    fig.suptitle("Changing only the training variance objective: paired final CE diagnostics",fontsize=13)
    fig.savefig(args.out_dir / "training_variance_comparison.png",dpi=190)
    fig.savefig(args.out_dir / "training_variance_comparison.svg")
    plt.close(fig)

    table = ["|训练路径数|训练方差|CDF差距中位数|平均条件拒绝率|逐组拒绝率范围|平均 σ|",
             "|---:|---|---:|---:|---:|---:|"]
    for row in summary.itertuples():
        count = "真参数" if row.training_paths == 0 else str(row.training_paths)
        table.append(f"|{count}|{row.train_variance}|{row.gap_median:.3f}|{100*row.mean_conditional_rejection:.2f}%|"
                     f"{100*row.rejection_min:.1f}%–{100*row.rejection_max:.1f}%|{row.mean_sigma:.4f}|")
    pairing_lines = []
    for n, group in paired.groupby("training_paths"):
        pairing_lines.append(f"- {int(n)} 条训练路径：{int((group.cdf_gap_change < 0).sum())}/{reps} 组经验CDF差距减小；"
                             f"{int((group.conditional_rejection_change < 0).sum())}/{reps} 组条件拒绝率估计下降；"
                             f"{int((group.absolute_size_error_change < 0).sum())}/{reps} 组拒绝率点估计更接近名义水平。"
                             f"CDF差距下降的同时区间完全低于0者为 {int((group.cdf_gap_change_high < 0).sum())}/{reps} 组，"
                             f"拒绝率下降对应为 {int((group.conditional_rejection_change_high < 0).sum())}/{reps} 组。")
    remaining_lines = []
    for row in summary.loc[summary.train_variance == "tangent"].itertuples():
        remaining_lines.append(f"- {row.training_paths} 条训练路径，tangent训练：{row.above_target}/{reps} 个条件拒绝率同时下界高于名义水平，"
                               f"{row.below_target}/{reps} 个同时上界低于名义水平；"
                               f"真实续接平均能量 {row.mean_held_energy:.3f}，参考平均能量 {row.mean_reference_energy:.3f}。")
    radius = float((results.dkw_cdf_radius_p+results.dkw_cdf_radius_q).iloc[0])
    plugin_link = os.path.relpath(args.plugin_dir, args.out_dir).replace("\\", "/")
    tangent_link = os.path.relpath(args.tangent_dir, args.out_dir).replace("\\", "/")
    report = f"""# 训练方差的配对对照：plug-in 与 tangent

在上一轮完全相同的 12 组训练数据、输入标准化、初始化种子和条件前缀上重新拟合。唯一科学设置变化是训练损失的方差项由 plug-in 改为 tangent；两边的后续诊断始终使用 tangent 残差及最终六分量 S_CE。

{chr(10).join(table)}

名义水平为5%，有限 M=2500 的精确目标为 {target:.8f}。表中平均值只对本轮12个固定分析状态的条件拒绝率估计取平均。

![训练方差配对结果](training_variance_comparison.png)

每条连线连接同一组数据的两种训练结果；点为条件分析，粗横线为中位数。CDF和拒绝率的竖线为对全部 {total_comparisons} 个唯一比较同时有效的95%内层Monte Carlo区间。

## 配对变化

{chr(10).join(pairing_lines)}

{chr(10).join(remaining_lines)}

拒绝率下降与达到5%校准是不同判断。区间包含名义水平不能证明校准；本次CDF差距同时区间半宽约 {radius:.4f}，而且只有12组训练数据，不能据此保证一般拟合状态的条件校准。

## 保持一致的内容

- 同一个已知真生成器、二维宽度16模型类、真实扩散 σ=0.42，48/192条各112步训练路径，10个Euler子步，3,000次full-batch Adam更新、学习率0.003和正则系数0.0001。
- 两种训练目标都有相同的零噪声中心与漂移残差平方项。tangent方案对协方差完整求导，包括其对漂移参数的依赖。
- 每个拟合的诊断窗口仍为60步。独立pilot有2,500条路径；真生成器和拟合模型各生成8,192条路径来估计分数分布。控制沿每条路径重算。
- 所有随机流、分数特征、尾部规则和参考库大小均保持一致。由于拟合改变，pilot数值和参考路径自然随该拟合重新计算。
- 已逐组核对保存的输入均值/尺度和前缀完全相同；两轮的真参数对照全部分数、特征和pilot数组也完全相同。见 [comparison_checks.json](comparison_checks.json)。

这测量的是训练方差选择对整个拟合—诊断程序的影响，不是固定同一模型后仅更换残差权重的收益，也没有比较备择下的检验力。仍然使用二维合成训练器，没有复现真实SIR的参数化、裁剪、signature控制、Huber损失和AdamW训练日程。

## 数值计算与不确定性

为减少tangent训练的运行时间，使用等价的批量tanh Jacobian及2×2协方差递推，把固定控制的仿射项只计算一次。已将两种方差目标的损失和梯度分别与原函数核对，误差在机器精度量级；相同种子下100次更新后的参数也一致。见 [training_calculation_checks.json](training_calculation_checks.json)。训练数据生成、续接模拟和分数计算仍调用原来的实现。

旧、新单独运行的区间各自覆盖36个比较。本报告从保存的原始分数重新计算覆盖两种方法的共同区间：两轮完全相同的oracle只计一次，故共12+24+24={total_comparisons}个唯一比较。每个条件分析的P/Q经验CDF分别用DKW界控制，再取并集界；配对差区间用两端点相减。原理见 [Reeve (2024)](https://arxiv.org/html/2403.16651v1)。没有重校准或修改分数。

条件拒绝率以Binomial尾概率积分计入每次新参考库的随机性。所有置信区间描述这批固定分析状态下的内层Monte Carlo误差；不把8,192条续接路径当作独立拟合次数，也不把它们解释为未来拟合总体均值的置信区间。

## 文件与复现

逐组重新计算结果：[combined_results.csv](combined_results.csv)。配对差：[paired_differences.csv](paired_differences.csv)。摘要：[summary.csv](summary.csv)。

原始设置：[plug-in]({plugin_link}/settings.json) 和 [tangent]({tangent_link}/settings.json)。这两组原始结果属于配套材料，源码包仅含汇总 CSV。若设置了 `shared_oracle_dir`，teacher、真参数对照的 `rep*_n0_model.npz`／`rep*_n0_scores.npz` 及公共实现检查从该相对目录读取；两种训练仍保留各自的拟合结果。共享真参数对照只计一次。以下命令从 release 根目录运行，并假定 plug-in 原始结果已补齐；Linux/macOS 使用 `.venv/bin/python` 和正斜杠路径。

```powershell
.\\.venv\\Scripts\\python.exe experiments\\check_ce_training_calculation.py --out-dir output\\ce_tangent_training_comparison_reproduction
.\\.venv\\Scripts\\python.exe experiments\\run_ce_conditional_fit_pilot.py --out-dir output\\ce_conditional_tangent_fit_reproduction --train-variance tangent --training-implementation batched
.\\.venv\\Scripts\\python.exe experiments\\compare_ce_training_variances.py --tangent-dir output\\ce_conditional_tangent_fit_reproduction --out-dir output\\ce_tangent_training_comparison_reproduction
```
"""
    (args.out_dir / "README.md").write_text(report, encoding="utf-8")
    print(summary.to_string(index=False))
    print("\n".join(pairing_lines))


if __name__ == "__main__":
    main()
