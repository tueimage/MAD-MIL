# MAD-MIL
__Multi-head Attention-based Deep Multiple Instance
Learning__

![alt text](https://github.com/tueimage/MAD-MIL/blob/main/model.png)

_The attention component used in ABMIL(Left), where only one attention mod- ule is utilized, and in MAD-MIL(Right), where multiple attention modules are incorporated._

This is the PyTorch implementation of the MAD-MIL, which is based on [CLAM](https://github.com/mahmoodlab/CLAM) and [WSI-finetuning](https://github.com/invoker-LL/WSI-finetuning).

**Data Preparation**

For the preprocessing of TUPAC16 and TCGA datasets, we adhere to CLAM's steps, incorporating features extracted from non-overlapping 256×256 patches at 20× magnification. We share the extracted features through the following links:

* TUPAC16
```
```

* BRCA
```
```

* LUNG
```
```

* KIDNEY
```
```

**Training**

The training can be done for different models and datasets with proper arguments like dataset_dir, task_name, model_name, lr, and reg. This is an example of training MAD-MIL on TUPAC16. 

```
python train.py --data_root_dir feat-directory --lr 1e-4 --reg 1e-6 --seed 2021 --k 5 --k_end 5 --split_dir task_tupac16 --model_type abmil_multihead --task task_1_tumor_vs_normal --csv_path ./dataset_csv/tupac16.csv --exp_code MAD_Five_reg_1e-6  --n 5
```

We sweep the weight-decay value among _{1e-2, 1e-3, 1e-4, 1-5, 1e-6, 1e-7, 1e-8}_ and choose the optimal value based on the validation loss.


**Examples**

We present attention heatmaps in the following figure to assess the interpretability of the methods.

<img src="https://github.com/tueimage/MAD-MIL/blob/main/heatmap.png">

_Attention heatmaps generated by different models for a Tumor slide selected from the TUPAC dataset. Top row: from left to right, the attention heatmap produced by the ABMIL and CLAM-MB. Bottom rows: from left to right, the attention heatmap generated by the MAD-MIL/3-head-1, MAD- MIL/3-head-2, MAD-MIL/3-head-3._


**Reference**

Please consider citing the following paper if you find our work useful for your project.

```
@misc{,
      title={Multi-head Attention-based Deep Multiple Instance Learning}, 
      author={},
      year={2024},
      eprint={},
      archivePrefix={},
      primaryClass={cs.CV}
}
```
