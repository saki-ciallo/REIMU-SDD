## ASVspoof2019

`asv19_convert_hf_meta.py` 可以把官方下载目录直接整理为 Hugging Face
`audiofolder` 结构，并在每个 split 中生成 `metadata.csv`：

```text
ASVspoof2019_hf/
└── data/
    ├── train/
    │   ├── *.flac
    │   └── metadata.csv
    ├── validation/
    │   ├── *.flac
    │   └── metadata.csv
    └── test/
        ├── *.flac
        └── metadata.csv
```

默认输入为 `../datasets/ASVspoof2019_T`，输出为同级目录
`../datasets/ASVspoof2019_hf`。默认使用硬链接，不复制音频数据，也不会改变
原下载目录：

```bash
python datasets_process/asv19_convert_hf_meta.py --dry-run
python datasets_process/asv19_convert_hf_meta.py
```

需要真正移动三个 `flac` 目录时使用：

```bash
python datasets_process/asv19_convert_hf_meta.py --mode move
```

源目录和输出目录不在同一文件系统时，使用完整复制模式：

```bash
python datasets_process/asv19_convert_hf_meta.py \
  --source-dir ../datasets/ASVspoof2019_T \
  --output-dir ../datasets/ASVspoof2019_hf \
  --mode copy
```

脚本只处理 LA；PA 目录会被忽略。官方 `dev` split 会转换为
`validation`。音频目录中未被 CM protocol 引用的文件会保留，但不会写入
`metadata.csv`。

## ASVspoof2021

`asv21_convert_hf_meta.py` 默认使用官方评测包中的协议：

```text
official_scores/2021/eval-package/keys/DF/CM/trial_metadata.txt
official_scores/2021/eval-package/keys/LA/CM/trial_metadata.txt
```

脚本将以下两个音频目录转换为 Hugging Face `audiofolder` 数据集：

```text
../datasets/ASVspoof2021_DF_eval/flac
../datasets/ASVspoof2021_LA_eval/flac
```

默认输出结构为：

```text
../datasets/
├── ASVspoof2021_DF_hf/data/test/
│   ├── *.flac
│   └── metadata.csv
└── ASVspoof2021_LA_hf/data/test/
    ├── *.flac
    └── metadata.csv
```

默认使用硬链接，不复制音频内容，也不会改变原始 eval 目录：

```bash
python datasets_process/asv21_convert_hf_meta.py --dry-run
python datasets_process/asv21_convert_hf_meta.py
```

也可以单独处理一个数据集，或真正移动音频目录：

```bash
python datasets_process/asv21_convert_hf_meta.py --subset DF
python datasets_process/asv21_convert_hf_meta.py --subset LA --mode move
```

官方评测包位于其他位置时，可以显式传入：

```bash
python datasets_process/asv21_convert_hf_meta.py \
  --keys-dir /path/to/eval-package/keys
```

脚本保留 `trial_metadata.txt` 的全部记录，不按第八列的 subset 字段过滤。
生成的 CSV 字段为 `file_name`、`utterance_id`、`system_id` 和 `labels`。

DF_E_2416160、DF_E_2101080、DF_E_4830191、DF_E_4459808、
DF_E_4887195 五个音频存在解码异常。
