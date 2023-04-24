# Training a model

## Train a model

Sample command:

```bash
python train_vit_vqvae.py \
    --output_dir output --overwrite_output_dir \
    --train_folder /mnt/disks/persist/datasets/vae_data/train \
    --valid_folder /mnt/disks/persist/datasets/vae_data/val \
    --config_name config/base/model \
    --disc_config_name config/base/discriminator \
    --do_eval --do_train \
    --batch_size_per_node 64 \
    --format rgb \
    --optim adam \
    --learning_rate 0.001 --disc_learning_rate 0.001
```
