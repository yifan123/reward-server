# reward-server

Serves reward inference using an HTTP server.

## Install

### GenEval

```bash
# First
conda create -n reward_server python=3.10.16
conda activate reward_server
# Then
pip install torch==2.1.2+cu121 torchvision==0.16.2+cu121 --index-url https://download.pytorch.org/whl/cu121
pip install gunicorn==23.0.0 openmim==0.3.9 open-clip-torch==2.31.0 numpy==1.26.0 opencv-python==4.11.0.86 clip-benchmark==1.6.1 flask==3.1.0
```

Then install mmdet:

```bash
mim install mmcv-full mmengine
git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection; git checkout 2.x
# Modify mmdet/__init__.py: set mmcv_maximum_version = '2.3.0'
pip install -e . --no-build-isolation
```

Then download mask2former:

```bash
cd reward-server/
mkdir -p ./model/mask2former2
wget https://download.openmmlab.com/mmdetection/v2.0/mask2former/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco_20220504_001756-743b7d99.pth -O ./model/mask2former2/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.pth
```

Then modify `MY_CONFIG_PATH` and `MY_CKPT_PATH` in `reward-server/reward_server/gen_eval.py` to your own paths.

## Usage

### GenEval

Start the server side:

```bash
conda deactivate
conda activate reward_server
gunicorn "app_geneval:create_app()"
```

You must modify `gunicorn.conf.py` to change the number of GPUs. You can first use 1 GPU to download the CLIP model, and then switch to 8 GPUs.

We recommend setting `NUM_DEVICES = 8` in `gunicorn.conf.py`; this will occupy about 10GB of memory on each gpu. Then, you can use the remaining memory to run the Flow-GRPO training. Currently, we have `bind = f"127.0.0.1:{port}"`, so the reward server can only be accessed within its own node. If you are training on multiple nodes, please run `gunicorn "app_geneval:create_app()"` on each node to start a reward server.

If you have an additional machine to run the reward server, please refer to `gunicorn.conf.py` to modify `bind` to `f"0.0.0.0:{port}"`, and then change the IP in [this line](https://github.com/yifan123/flow_grpo/blob/main/flow_grpo/rewards.py#L169) to the IP of the machine where you started the reward server.

For the H type GPU, you might encounter the error "error in ms_deformable_im2col_cuda: no kernel image is available for execution on the device." You can fix it by

```bash
mim uninstall mmcv-full
TORCH_CUDA_ARCH_LIST="9.0" pip install mmcv-full
```

After starting, you can run the client for testing:

```bash
python test/test_geneval.py
```

The output will be
```
{'scores': [0.75, 1.0], 'rewards': [0.0, 1.0], 'strict_rewards': [0.0, 1.0], 'group_rewards': {'single_object': [-10.0, 1.0], 'two_object': [-10.0, -10.0], 'counting': [-10.0, -10.0], 'colors': [-10.0, -10.0], 'position': [-10.0, -10.0], 'color_attr': [0.0, -10.0]}, 'group_strict_rewards': {'single_object': [-10.0, 1.0], 'two_object': [-10.0, -10.0], 'counting': [-10.0, -10.0], 'colors': [-10.0, -10.0], 'position': [-10.0, -10.0], 'color_attr': [0.0, -10.0]}}
```

### DeQA
If there's an error, please refer to [DeQA](https://github.com/zhiyuanyou/DeQA-Score ) to install DeQA's dependencies.
Start the server side:

```bash
cd reward-server/
conda deactivate
conda activate reward_server
gunicorn "app_deqa:create_app()"
```

After starting, you can run the client for testing:

```bash
python test/test_deqa.py
```
