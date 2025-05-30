# A-VL: Adaptive Attention for Large Vision-Language Models


## Installation

#### Step 1: Create a conda environment.

You need to have installed conda first.

```
conda create -n aaai2025 python=3.10
conda activate aaai2025
```

#### Step 2: (Optional) Install PyTorch. 

We use PyTorch 2.1.2+cu118. You can choose the pytorch version according to your hardware. 

This step is not necessary, because PyTorch can be automatically installed in subsequent installation of other library. Installing PyTorch 2.1.2 is just to ensure that it is consistent with our environment.

#### Step 3: Install A-VL.

Our script will automatically help you install my customized transformers library, lmms-eval library and related support libraries. This may take some time. Please keep your network open.

```shell
cd src
bash setup.sh
```

#### Step 4: Check environment.

You can use this python script to test whether our customized transformers library is installed.

```python
import transformers
print(transformers.__file__)
```

If the output path points to the transformers library in the `src` folder under the our path, it means that you have installed our custom library.

## Usage of A-VL

First, you can run the original model:

```shell
CUDA_VISIBLE_DEVICES=0 python \
    -m lmms_eval \
    --model llava_hf \
    --model_args pretrained=llava-hf/llava-1.5-7b-hf,device_map=auto,attn_implementation=eager \
    --tasks docvqa \
    --batch_size 1 \
    --log_samples \
    --output_path ./logs/ \
    --log_samples_suffix llava_1_5_7b
```

Since our method has been integrated into the lmms-eval library, you can test any configuration of A-VL through adding these parameters:
- `--vision_adaptive_attention`: (true or false) Whether to enable vision adaptive attention
- `--vision_adaptive_attention_rate`: The proportion of image tokens to be deleted. For example, 0.6 means deleting 60% of the vision cache
- `--vision_adaptive_attention_compute_rate`: The proportion of vision cache not used for calculation. For example, 0.7 means only 30% vision cache used in computation.
- `--vision_adaptive_attention_update_interval`: The core cache set update interval. For example, set 3 means the core cache will update every 3 steps.
- `--prefill_filter_flag`: (true or false) Whether to delete the image token during the prefill phase.
- `--prefill_filter_rate`: The proportion of image tokens deleted during the prefill phase. For example, 0.3 means deleting 30% image tokens in prefill phase.
- `--text_pruning_flag`: (true or false) Whether to delete the text cache.
- `--text_pruning_rate`: The percentage of text cache deleted. For example, 0.3 means deleting 30% text cache.

The complete output and performance will be saved in the `output_path` you specified.

## Additional Notes

- We have made a lot of modifications in the LLaVA, LLaMA, LLaVA-Next of the transformers library, and you can refer to the code of these modules.
- The datasets and models will be automatically downloaded via huggingface. Please keep the network open.
- If the program does not respond, please check whether your network can access the resources of NLTK.
- Please make sure that the GPU memory size is sufficient. We conducted experiments on NVIDIA A100 and A40 GPUs.

## Paper

Please cite our article

```
@inproceedings{zhang2025vl,
  title={A-VL: Adaptive Attention for Large Vision-Language Models},
  author={Zhang, Junyang and Yuan, Mu and Zhong, Ruiguang and Luo, Puhan and Zhan, Huiyou and Zhang, Ningkang and Hu, Chengchen and Li, Xiang-Yang},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={39},
  number={21},
  pages={22461--22469},
  year={2025}
}
```