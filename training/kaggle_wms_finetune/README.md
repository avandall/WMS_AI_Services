# WMS Kaggle Fine-Tuning Bundle

Upload this folder to Kaggle as a dataset or zip it and upload the zip.

## Kaggle Setup

```bash
cp -r /kaggle/input/<your-dataset-name>/kaggle_wms_finetune /kaggle/working/wms-finetune
cd /kaggle/working/wms-finetune
bash training/fine_tuning/kaggle_bootstrap.sh
```

Run the bootstrap before importing `torch`, `transformers`, `trl`, or `unsloth`.
If those packages were already imported, restart the Kaggle session after installing.
If a previous run installed a broken Torch/CUDA stack, use Kaggle's "Factory reset session"
or start a fresh notebook session before running the bootstrap again.

## Train

```bash
bash training/fine_tuning/kaggle_train.sh --skip-merge
```

Remove `--skip-merge` only if Kaggle has enough disk and VRAM to merge the final model.

## Important Files

- `training/fine_tuning/data/wms_data_enriched.jsonl`: training dataset.
- `training/fine_tuning/train_wms.py`: fine-tuning script.
- `training/fine_tuning/kaggle_bootstrap.sh`: installs pinned Kaggle dependencies.
- `training/fine_tuning/kaggle_train.sh`: runs training with the Kaggle library path configured.
- `training/fine_tuning/requirements.kaggle-unsloth.txt`: direct fine-tuning dependencies.
- `training/fine_tuning/constraints.kaggle-unsloth.txt`: pinned resolver constraints.
- `src/ai_service/pipeline/templates.py`: runtime prompt helper required by `train_wms.py`.
