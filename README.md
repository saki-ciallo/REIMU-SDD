#  REIMU Speech Deepfake Detection Official Implementation

This repository contains official implementation of the papaer: [REIMU: Efficient Heterogeneous Hierarchical Reasoning for SSL-Based Speech Deepfake Detection (arxiv:2608.00857)](https://arxiv.org/abs/2608.00857)



Also, this repository provides two implementations of the same method: a paper-aligned legacy implementation and a modular implementation intended for further experimentation and adaptation.

## Legacy vs. Improved

| Implementation | Location | Best for |
| --- | --- | --- |
| Paper-aligned (Legacy) | Root, `src/`, `utilis/` | Reproducing paper results and loading original checkpoints |
| Modular (Improved) | `code_optimize/` | Custom architectures, ablations, and adapting to new tasks |


* Legacy: The original implementation used for the paper's experiments and public models.
* Modular: A refactored, better modular version designed for flexibility and new experiments.

**Notice**: Due to the structural changes, `state_dict` parameter names differ between versions.


## Getting Started

### Environment

The dependency baseline is declared in
[`code_optimize/pyproject.toml`](code_optimize/pyproject.toml). We recommend use `conda` or `uv` to create the enviroment.

### Datasets
* [ASVspoof 2019 LA](https://datashare.ed.ac.uk/handle/10283/3336), ASVspoof 2021 [LA](https://zenodo.org/record/4835108)/[DF](https://zenodo.org/record/4837263)

### SSL models
We employ the HuBERT, wav2vec 2.0, and WavLM Base/Base+ models, from the Hugging Face Transformers library.


## Usage

### Running the paper-aligned version

Training uses the shared configuration followed by an experiment overlay:

```bash
python train.py \
  --config configs/train_common.yaml \
  --experiment-config configs/experiments/baseline/"pickone".yaml
```

The root-level `run_pipeline.sh`, `test_asv19.sh`, `test_asv21.sh`, and scoring scripts provide the corresponding training, inference, and evaluation workflows.

### Running the modular version

The modular pipeline can train, test, and score one or more experiment configs:

```bash
./code_optimize/run_pipeline.sh \
  code_optimize/configs/experiments/baseline/"pickone".yaml
```

For researchers and developers extending this project, `code_optimize/` is the recommended starting point.

## Reproducibility and adaptation

Use the legacy version for exact paper reproduction and checkpoint compatibility, or the modular version for architectural changes, ablations, and custom downstream tasks (core model logic remains identical).

## Citation
If you find this repository or our work helpful, please consider giving this repo a star  and citing our paper:
```
@article{ng2026reimu,
  title={REIMU: Efficient Heterogeneous Hierarchical Reasoning for SSL-Based Speech Deepfake Detection},
  author={Ng, Kwok-Ho and Song, Tingting and Feng, Bingwen and Li, Peiya},
  journal={arXiv preprint arXiv:2608.00857},
  year={2026}
}
```

## Contact
If you have any questions or discussions, feel free to open an issue or contact us at [email](mailto:kwokhong@stu2024.jnu.edu.cn).