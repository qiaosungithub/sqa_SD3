# written by sqa

from huggingface_hub import hf_hub_download

# download SD3.5 medium model
# you need to first login huggingface
hf_hub_download(repo_id="stabilityai/stable-diffusion-3.5-medium", filename="sd3.5_medium.safetensors", local_dir="models/")

# download CLIP-L
# you need to agree in the website
hf_hub_download(repo_id="stabilityai/stable-diffusion-3.5-large", filename="text_encoders/clip_l.safetensors", local_dir="models/")

hf_hub_download(repo_id="stabilityai/stable-diffusion-3.5-large", filename="text_encoders/clip_g.safetensors", local_dir="models/")

hf_hub_download(repo_id="stabilityai/stable-diffusion-3.5-large", filename="text_encoders/t5xxl_fp16.safetensors", local_dir="models/")