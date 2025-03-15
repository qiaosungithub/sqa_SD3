import torch
from safetensors import safe_open
from sd3_impls import (
    SDVAE,
)
import matplotlib.pyplot as plt

def load_into(ckpt, model, prefix, device, dtype=None, remap=None):
    """Just a debugging-friendly hack to apply the weights in a safetensors file to the pytorch module.
    
    ckpt: an opened safe_open file
    model: the pytorch model
    prefix: the prefix in the safetensors file
    """
    for key in ckpt.keys():
        model_key = key
        if remap is not None and key in remap:
            model_key = remap[key]
        if model_key.startswith(prefix) and not model_key.startswith("loss."):
            path = model_key[len(prefix) :].split(".") # remove prefix, split into list
            obj = model
            for p in path:
                if obj is list: # when obj is Module list
                    obj = obj[int(p)]
                else:
                    obj = getattr(obj, p, None)
                    if obj is None:
                        print(
                            f"Skipping key '{model_key}' in safetensors file as '{p}' does not exist in python model"
                        )
                        break
            if obj is None: # not get
                continue
            try:
                tensor = ckpt.get_tensor(key).to(device=device) # get the tensor
                if dtype is not None and tensor.dtype != torch.int32:
                    tensor = tensor.to(dtype=dtype)
                obj.requires_grad_(False)
                # print(f"K: {model_key}, O: {obj.shape} T: {tensor.shape}")
                if obj.shape != tensor.shape:
                    print(
                        f"W: shape mismatch for key {model_key}, {obj.shape} != {tensor.shape}"
                    )
                obj.set_(tensor) # set into params
            except Exception as e:
                print(f"Failed to load key '{key}' in safetensors file: {e}")
                raise e
            

class VAE:
    def __init__(self, model, dtype: torch.dtype = torch.float16):
        with safe_open(model, framework="pt", device="cpu") as f:
            self.model = SDVAE(device="cpu", dtype=dtype).eval().cpu()
            prefix = ""
            # if assembled in SD3 model, prefix is "first_stage_model."
            if any(k.startswith("first_stage_model.") for k in f.keys()):
                prefix = "first_stage_model."
            load_into(f, self.model, prefix, "cpu", dtype)

path = "/kmh-nfs-ssd-eu-mount/data/SD3.5_pretrained_models/sd3.5_medium.safetensors"

vae = VAE(path)
print("Models loaded.")

image_path = "/kmh-nfs-ssd-eu-mount/code/qiao/work/jax_SD/test_image_64/0/0/4.png"
# load the image with PIL
image = plt.imread(image_path)
image = image[:, :, :3]
image = 2 * image - 1
# image = Image.open(image_path).convert("RGB")
image = torch.tensor(image).half()
image = image[None, ...] # (B, H, W, C)
image = image.transpose(1, 3).transpose(2, 3).contiguous()
print(image.shape)
x = vae.model.encode(image)
# x = vae.model.decode(x)
print(x.shape)
x = x[0]
x = x.float()
x = (x+1)/2
x = torch.clip(x, 0, 1) # (C, H, W)
x = x.transpose(0, 2).transpose(0, 1).contiguous()
x = x[:, :, :3]
x = x.numpy()
plt.imsave("test_latent_8x8.png", x)