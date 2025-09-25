# ComfyUI-Fixes
This repo is serving fixes for multiple custom nodes
This is also for Ubuntu/Debian based OS, not Windows, not MacOS, not Arch etc..

I created this repo here, because I was dumb enough to run "uv venv" and it deleted my .venv folder, together with any fixes I did.
So now when I do this again, at least here are some notes, that you can use aswell..

# SageAttention

When fresh installing ComfyUI, I often have the issue, that triton and SageAttention itself will not work on Linux.
I use ComfyUI-KJNodes and its containing Patch Sage Attention Node.
Either because of Blackwell not being supported or it does not install the correct wheel.
Also the tutorials I found were mostly focused on Windows, not even providing any Build wheels for Linux.

I do not make a script to fix this all, as within bash I often have issues enabling venv correctly, it often just doesn't and we do not want to break your system.

> ⚠️ **Warning:** If you use the CUDA Toolkit for other applications, switching `nvcc` or installing a different cuda-toolkit might break their build process.

If you are running Ubuntu 24.04 and Cuda 12.8 like me, you can also try installing the wheel I added in this repo.
```bash
pip install sageattention-2.2.0-cp312-cp312-linux_x86_64.whl
```

## 1 - Checking Cuda versions

SageAttention has to be built again and it requires Cuda Toolkit 12.4 at least.
Sometimes though, as in my case, your OS will install their own nvcc compiler.

Cuda in itself consists of way more than just one tool.
If you have Cuda 12.8 or 12.9 as I do, here is how I fixed it:

``` bash
python - <<'PY'
import torch, sys, subprocess
print("Torch:", torch.__version__, "CUDA (torch):", torch.version.cuda)
PY

nvcc --version || echo "nvcc missing?"
nvidia-smi | sed -n '1,15p'
```
This tells you which Torch and which Cuda-Toolkit version you have installed.
Cuda Toolkit should return 12.4 at minimum.
For me I had Cuda 12.8 installed, but nvcc was at Version 12.0.

You can check which cuda-12 versions you have installed with:
``` bash
ls /usr/local/ | grep cuda-12
```
It should at least return 12-4, for me its 12.8 and 12.9

When running:
```which nvcc```
it only returned:
```
/usr/bin/nvcc
```
This means that the main version is having the wrong symlink.

Let's assume you have Cuda Toolkit 12.8 installed from now on...

``` bash
sudo update-alternatives --install /usr/bin/nvcc nvcc /usr/local/cuda-12.8/bin/nvcc 1208
```



Well... 
It seems like installing cuda-toolkit 12.8 or 12.9 also does NOT replace nvcc for some people, just like me.
I did have 
/usr/local/cuda-12.9/bin/nvcc and /usr/local/cuda-12.8/bin/nvcc
already installed.

Okay, I thought maybe following nvidia's official script may lead me somewhere..
https://developer.nvidia.com/cuda-downloads

It might show you Cuda 13.0 or newer, just go with the instructions.
Mine where for Ubuntu 24.04 and x86_64 architecture, so please use the link above if you have a different system.

``` bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-ubuntu2404.pin
sudo mv cuda-ubuntu2404.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/13.0.1/local_installers/cuda-repo-ubuntu2404-13-0-local_13.0.1-580.82.07-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2404-13-0-local_13.0.1-580.82.07-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2404-13-0-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda-toolkit-13-0
```
This didn't work for me, as cuda-toolkit-13-0 did not exist in the sources..

I kept cuda-toolkit-12-8 and tried reinstalling it:
```
sudo apt install cuda-toolkit-12-8
```

Running ```nvcc -V``` was still showing 12.0

This seems to be a broken or invalid symlink.
Next up, this worked to fix the symlink ( I am skipping a few steps here, this is in case , be cautious)

``` bash
# 1) Show current state
sudo update-alternatives --display nvcc || true
readlink -f /usr/bin/nvcc || true

# 2) Backup /usr/bin/nvcc in case it is NOT an actual symlink..
sudo mv /usr/bin/nvcc /usr/bin/nvcc.bak-120 || true

# 3) select 12.8 as an alternative update for nvcc 
sudo update-alternatives --install /usr/bin/nvcc nvcc /usr/local/cuda-12.8/bin/nvcc 1208
sudo update-alternatives --set nvcc /usr/local/cuda-12.8/bin/nvcc

# 4) Check if it worked
readlink -f /usr/bin/nvcc
nvcc --version
```

And well, this returned Version 12.8!

## 2 - Rebuild SageAttention

Although we finally got to fix the nvcc version, we now have to build the wheel again.

Make sure you have the ComfyUI venv active:
```bash
cd ComfyUI/
ls -a
# only if you have no venv or .venv showing after the last command..
python3 -m venv .venv

source .venv/bin/activate
```

Next, we should navigate OUT of the ComfyUI folder, while still remaining in the (.venv) mode.
We do that to avoid having any issues later when we try updating ComfyUI..
```bash
cd ~/

# 1) clone SageAttentions Repository
git clone https://github.com/thu-ml/SageAttention.git
cd SageAttention

# 2) Clean
python setup.py clean
rm -rf build dist *.egg-info

# 3) make sure you do not have any remains of the broken sageattention
python -m pip uninstall sageattention

```
We now have to set the system variables correctly.
The following can be run in your bash shell, don't get confused that it executes python to get there.
```bash
export TORCH_CUDA_ARCH_LIST=$(python - <<'PY'
import torch
cc = torch.cuda.get_device_capability()
print(f"{cc[0]}.{cc[1]}")
PY
)
```

If it gives you any errors, you might be missing torch, install it using the following then try the script again.
``` bash
python -m pip install torch
```
(If you got this warning, just ignore it, it does not have to be fixed.
``` .venv/lib/python3.12/site-packages/torch/cuda/__init__.py:63: FutureWarning: The pynvml package is deprecated. Please install nvidia-ml-py instead. If you did not install pynvml directly, please report this to the maintainers of the package that installed pynvml for you.
  import pynvml  # type: ignore[import]
```
)


If that worked, we can now continue with building SageAttention.

``` bash
export FORCE_CUDA=1
export MAX_JOBS=$(nproc)
echo "TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST"
pip wheel . -w dist
```
This is going to take a few minutes, seriously I have 16 threads and it took around 20 Minutes for me.

Once it is done, we install the newest created wheel file:
``` bash
WHEEL=$(ls -t dist/*.whl | head -n1)
if [ -z "$WHEEL" ]; then
  echo "❌ No wheel found in dist/"
  exit 1
fi

echo "📦 Installing $WHEEL ..."
pip install --force-reinstall --no-deps "$WHEEL"

# === Quick sanity check ===
python - <<'PY'
import torch, importlib
print("Torch version:", torch.__version__)
print("CUDA (torch):", torch.version.cuda)
print("SageAttention importable:", importlib.util.find_spec("sageattention") is not None)
PY
```


## 3 - Verify inside ComfyUI

After the installation, restart ComfyUI and use the SageAttention Patcher, or whatever Node you wanted to use.
It should now have sageattention working without any errors.

## Troubleshoot

You might get some error like:
```
TypeError: BaseLoaderKJ._patch_modules.<locals>.attention_sage() got an unexpected keyword argument 'transformer_options'
```
This has been fixed already, just update the ComfyUI-KJNodes custom Node and you should be fine. 


## TL;DR (Experienced users)
- Activate venv inside ComfyUI,
- Move out of the folder,
- Clone the official SageAttention Repo and navigate into it,
```bash
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
export TORCH_CUDA_ARCH_LIST=$(python - <<'PY'
import torch; cc=torch.cuda.get_device_capability(); print(f"{cc[0]}.{cc[1]}")
PY
)
export FORCE_CUDA=1
pip wheel . -w dist && pip install --force-reinstall --no-deps $(ls -t dist/*.whl | head -n1)
```
And you should be done.


