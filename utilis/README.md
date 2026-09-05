ASV19 测试与评分入口位于项目根目录：

- `test_asv19.py` 生成 `tested_results/asv19la_<step>.csv` 和对应 metrics JSON。
- `score_asv19.py` 扫描上述 CSV，生成 CM score txt、分攻击类型结果和 `score.csv`。

`get_asv19_scores.py` 只保留 EER/t-DCF 的底层计算函数，不再承担命令行流程。

固定长度音频数据集可通过 `build_fixed_audio_dataset.py` 创建：

```bash
python utilis/build_fixed_audio_dataset.py --preset asv19
python utilis/build_fixed_audio_dataset.py --preset asv21-la
python utilis/build_fixed_audio_dataset.py --preset asv21-df
```

三个 preset 默认生成 16 kHz、4 秒的 `input_values`。ASV21-DF preset
内置五个 TorchCodec 异常文件的 ffmpeg fallback。输出目录已存在时脚本会拒绝覆盖；
可通过 `--output-dir` 指定新目录，或先使用 `--dry-run` 查看解析结果。

Preset 对应的 Hugging Face audiofolder 输入和 `save_to_disk` 输出为：

| Preset | 输入 | 输出 |
| --- | --- | --- |
| `asv19` | `../datasets/ASVspoof2019_hf/data` | `../datasets/ASVspoof2019_16k_4s_fixed` |
| `asv21-la` | `../datasets/ASVspoof2021_LA_hf/data` | `../datasets/ASVspoof2021_LA_16k_4s_fixed` |
| `asv21-df` | `../datasets/ASVspoof2021_DF_hf/data` | `../datasets/ASVspoof2021_DF_16k_4s_fixed` |

ASV19 输出包含 `train`、`validation`、`test`；ASV21 LA/DF 输出均包含
`test`。输出可直接通过 `datasets.load_from_disk` 加载。

训练时可通过 `configs/train_common.yaml` 中的 `rawboost_algo` 对固定 4 秒
波形在线应用 RawBoost。`train.py` 只把增强挂到 `train` 和可选的
`validation` split；测试数据不会在训练入口加载或增强。当前顺序是
“已定长波形 -> RawBoost”，因此与原始波形先增强再定长的流程不完全等价。
如需稳定的验证集指标和早停判断，可设置：

```yaml
rawboost_algo: 4
rawboost_apply_to_validation: false
```

任意兼容 Hugging Face `audiofolder` 的目录也可以显式处理：

```bash
python utilis/build_fixed_audio_dataset.py \
  --data-dir ../datasets/custom/data \
  --output-dir ../datasets/custom_16k_4s_fixed
```
