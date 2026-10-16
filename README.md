# Active Learning for Medical Vision-Language Segmentation


Deep Learning models combining multi-modal capabilities, particularly vision and language (vision-language models), have garnered significant attention lately, where
Vision-language segmentation models (VLSMs) enable open-vocabulary segmentation conditioned on text.
Although VLSMs are promising with their potential to allow humans to inject useful information as prompts to guide segmentation during inference, it requires finetuning foundation VLSMs with domain-specific medical data which is expensive and not readily available.
Although there are various active-learning methods studied to identify the best samples to annotate when labeling new data under a limited budget, there is no work studying active learning in VLSMs.
%in Many work explore active-learning to help helps select most promising  In such a scenario, leveraging the language capabilities of large models trained on the natural domain and adapting them to the medical domain with very few annotated training examples would be very helpful.
In this paper, we propose a novel Active Learning-Based Few-Shot Learning approach tailored for VLSMs that assists in selecting the most informative few examples for few-shot learning by leveraging the multimodal image-text understanding of such models.
Our results show that leveraging diverse prompts to generate multiple outputs from VLSMs and using a consistency-based approach to select samples provide better samples.





This repository contains experiments on active learning for few-shot adaptation of vision-language segmentation models to medical images.

The main idea is to use the model's response to different text prompts as a measure of uncertainty. For the same image, we generate segmentation masks using several prompts referring to the same object. If the predictions disagree strongly, the image is treated as informative and selected for the next fine-tuning round.

The current implementation uses CLIPSeg and CRIS on polyp segmentation datasets.

## Active learning setup

For each unlabeled image:

1. Run the model using prompts `p2` to `p9`
2. Collect the 8 predicted segmentation masks
3. Measure agreement between the masks using Multi-IoU
4. Rank images by agreement
5. Select images with the lowest Multi-IoU
6. Add them to the training set
7. Fine-tune the model and repeat

The sampling fractions used in the experiments are:

`2.5%, 5%, 10%, 20%, 40%, 60%, 80%, 100%`

A random sampling baseline is also included for comparison.

## Models

- CLIPSeg
- CRIS

## Datasets

The experiments were run mainly on:

- Kvasir-SEG
- ClinicDB
- BKAI polyp dataset

The repository also contains cross-dataset evaluation scripts for testing how models trained on one dataset generalize to the others.

## Setup

Python 3.10 is recommended.

```bash
pip install -r requirements.txt
```

The pretrained CRIS weights should be placed under `pretrained/`. CLIPSeg weights are loaded from Hugging Face.

Dataset paths are configured through Hydra.

## Active learning code

The main scripts are under:

```text
scripts/active_learning/
```

Important files:

```text
infer_al.py          # generate predictions for multiple prompts
eval_al.py           # compute prompt consistency
multi_iou.py         # Multi-IoU implementation
finetune_al.py       # fine-tune on selected samples
finetune_random.py   # random sampling baseline
```

Sample selection is handled in:

```text
utils/active_learning/metric_sampler.py
```

Random subsets are generated using:

```text
utils/active_learning/random_sampler.py
```

## Running an active learning round

Generate predictions on the remaining pool:

```bash
python scripts/active_learning/infer_al.py --train_frac=0.1
```

Compute prompt consistency:

```bash
python scripts/active_learning/eval_al.py \
    --seg_root_path=<path-to-predicted-masks> \
    --csv_path=<output-path>/consistency.csv
```

Select samples for the next fraction:

```bash
python utils/active_learning/metric_sampler.py \
    --sampling_frac=0.2 \
    --ds_root=<dataset-root> \
    --op_root=<output-root> \
    --ds_name=kvasir_polyp
```

Fine-tune on the selected subset:

```bash
python scripts/active_learning/finetune_al.py --train_frac=0.2
```

## Random baseline

Random subsets can be generated with:

```bash
python utils/active_learning/random_sampler.py \
    --ds_root=<dataset-root> \
    --ds_name=kvasir_polyp
```

and trained using:

```bash
python scripts/active_learning/finetune_random.py --train_frac=0.2
```

## Notes
The active learning code currently included in this repository implements the Multi-IoU based sampling strategy.
