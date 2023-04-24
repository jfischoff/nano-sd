Original Readme from https://github.com/patil-suraj/vit-vqgan

    # vit-vqgan

    JAX implementation of [ViT-VQGAN](https://arxiv.org/pdf/2110.04627.pdf).

    ## Acknowledgements

    * Jiahui Yu for his advice on the implementation based on the work on [ViT-VQGAN](https://arxiv.org/pdf/2110.04627.pdf) and [Parti](https://arxiv.org/abs/2206.10789).
    * [Phil Wang](https://github.com/lucidrains) for suggesting using convolutions after self-attention layers.
    * [Katherine Crowson](https://github.com/crowsonkb) for suggesting using convolutions in feed-forward layers.

Training Dataset:

A subset of the improved_asthetics_6plus was created in tfrecord for fast experimentation using the following commands.

Training set
img2dataset --url_list /mnt/disks/persist/datasets/subset_improved_asthetics_6plus --input_format "parquet" --url_col "URL" --caption_col "TEXT" --encode_quality 100 --encode_format webp --output_format tfrecord --output_folder /mnt/disks/persist/datasets/subset_improved_asthetics_6plus-data --number_sample_per_shard 10000 --processes_count 128 --thread_count 64 --image_size 256 --min_image_size 256  --resize_mode="center_crop" --enable_wandb True

Validation set
img2dataset --url_list /mnt/disks/persist/datasets/subsetv_improved_asthetics_6plus --input_format "parquet" --url_col "URL" --caption_col "TEXT" --encode_quality 100 --encode_format webp --output_format tfrecord --output_folder /mnt/disks/persist/datasets/subsetv_improved_asthetics_6plus-data --number_sample_per_shard 10000 --processes_count 128 --thread_count 64 --image_size 128 --min_image_size 256 --resize_mode="center_crop" --enable_wandb True

Training:

python train_vit_vqvae.py     --output_dir /home/ashwin/vit-vqgan/output --overwrite_output_dir     --train_folder /mnt/disks/persist/datasets/subset_improved_asthetics_6plus-data    --valid_folder /mnt/disks/persist/datasets/subsetv_improved_asthetics_6plus-data     --config_name config/base/model     --disc_config_name config/base/discriminator     --do_eval --do_train     --batch_size_per_node 64     --format rgb     --optim adam     --learning_rate 0.001 --disc_learning_rate 0.001