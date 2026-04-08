python posterior_sample.py \
+data=demo-ffhq \
+model=ffhq256ddpm \
+task=down_sampling \
+sampler=edm_multi_block_kaczmarz \
task_group=pixel \
save_dir=results_block_kaczmarz \
num_runs=1 \
sampler.diffusion_scheduler_config.num_steps=5 \
sampler.diffusion_scheduler_config.num_denoising_steps=200 \
task.pixel.operator.sigma=0.01 \
batch_size=1 \
data.start_id=0 data.end_id=10 \
name=down_sampling_demo \
gpu=0

## SR
python posterior_sample.py \
+data=test-ffhq \
+model=ffhq256ddpm \
+task=down_sampling \
+sampler=edm_multi_block_kaczmarz \
task_group=pixel \
save_dir=results_multi_block_kaczmarz \
num_runs=1 \
eval_only=False \
sampler.diffusion_scheduler_config.num_steps=5 \
sampler.diffusion_scheduler_config.num_denoising_steps=200 \
batch_size=1 \
name=down_sampling \
gpu=0

# inpainting box
python posterior_sample.py \
+data=test-ffhq \
+model=ffhq256ddpm \
+task=inpainting \
+sampler=edm_multi_block_kaczmarz \
task_group=pixel \
save_dir=results_block_kaczmarz \
num_runs=1 \
eval_only=False \
sampler.diffusion_scheduler_config.num_steps=5 \
sampler.diffusion_scheduler_config.num_denoising_steps=200 \
batch_size=1 \
name=inpainting_box \
gpu=0