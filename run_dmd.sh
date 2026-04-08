#!/usr/bin/bash
# python posterior_sample.py \
# +data=test-imagenet \
# +model=imagenet256ddpm \
# +task=avg_pooling \
# +sampler=edm_dmd \
# task_group=pixel \
# save_dir=CVPR2026_final \
# num_runs=1 \
# sampler.diffusion_scheduler_config.num_steps=5 \
# task.pixel.operator.sigma=0.05 \
# batch_size=1 \
# data.start_id=0 \
# name=down_sampling_diverse \
# gpu=0
# python posterior_sample.py \
# +data=test-imagenet \
# +model=imagenet256ddpm \
# +task=gaussian_deblur_svd \
# +sampler=edm_dmd \
# task_group=pixel \
# save_dir=CVPR2026_final \
# num_runs=1 \
# sampler.diffusion_scheduler_config.num_steps=5 \
# task.pixel.operator.sigma=0.05 \
# batch_size=1 \
# data.start_id=0 \
# name=gaussian_deblur_svd \
# gpu=0
# python posterior_sample.py \
# +data=test-imagenet \
# +model=imagenet256ddpm \
# +task=pixel_inpainting_svd \
# +sampler=edm_dmd \
# task_group=pixel \
# save_dir=CVPR2026_final \
# num_runs=1 \
# sampler.diffusion_scheduler_config.num_steps=5 \
# task.pixel.operator.sigma_y=0.05 \
# batch_size=1 \
# data.start_id=0 \
# name=pixel_inpainting_svd \
# gpu=0
python posterior_sample.py \
+data=test-imagenet \
+model=imagenet256ddpm \
+task=box_inpainting_svd \
+sampler=edm_dmd \
task_group=pixel \
save_dir=CVPR2026_final \
num_runs=1 \
sampler.diffusion_scheduler_config.num_steps=5 \
task.pixel.operator.sigma_y=0.05 \
batch_size=1 \
data.start_id=0 \
name=box_inpainting_svd \
gpu=0