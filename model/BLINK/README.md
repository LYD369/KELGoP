# ELQ

This module is based on the implementation from the following paper: [https://arxiv.org/pdf/2010.02413.pdf](https://arxiv.org/pdf/2010.02413.pdf)

```bibtex
@inproceedings{li-etal-2020-efficient,
  title={Efficient One-Pass End-to-End Entity Linking for Questions},
  author={Li, Belinda Z and Min, Sewon and Iyer, Srinivasan and Mehdad, Yashar and Yih, Wen-tau},
  booktitle="Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)",
  pages={6433--6441},
  year={2020}
}
```

Original Repository: [https://github.com/facebookresearch/BLINK](https://github.com/facebookresearch/BLINK)

## Setup
1. Install requirements
```console
pip install -r ./requirements.txt
```

2. Download the pretrained models, indices, and entity embeddings
```console
chmod +x download_elq_models.sh
./download_elq_models.sh
```
