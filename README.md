<p align="center">
  <h1 align="center">Machine Mental Imagery: Empower Multimodal Reasoning with Latent Visual Tokens</h1>
  <p align="center">
    arXiv 2025
  </p>
  <p align="center">
    <a href="https://miicheyang.github.io">Zeyuan Yang*</a>,
    <a href="https://scholar.google.com/citations?user=AIm87GIAAAAJ">Xueyang Yu*</a>,
    <a href="https://chendl02.github.io/">Delin Chen</a>,
    <a href="https://maohaos2.github.io/Maohao/">Maohao Shen</a>,
    <a href="https://people.csail.mit.edu/ganchuang">Chuang Gan</a>
  </p>
  <p align="center">
    <a href="https://vlm-mirage.github.io">
      <img src='https://img.shields.io/badge/Paper-PDF-red?style=flat&logo=arXiv&logoColor=red' alt='Paper PDF'>
    </a>
    <a href='https://vlm-mirage.github.io' style='padding-left: 0.5rem;'>
      <img src='https://img.shields.io/badge/Project-Page-blue?style=flat&logo=Google%20chrome&logoColor=blue' alt='Project Page'>
    </a>
    <!-- <a href='https://huggingface.co/anyeZHY/tesseract' style='padding-left: 0.5rem;'>
      <img src='https://img.shields.io/badge/Model-Hugging%20Face-yellow?style=flat&logo=Hugging%20face&logoColor=yellow' alt='Model Hugging Face'>
    </a> -->
  </p>
</p>

We propose TesserAct, **the first open-source and generalized 4D World Model for robotics**, which takes input images and text instructions to generate RGB, depth, and normal videos, reconstructing a 4D scene and predicting actions.

<!-- <p align="center">
    <img src="asset/teaser.png" alt="Logo" width="190%">
</p> -->

<br>

<!-- TABLE OF CONTENTS -->
<details open="open" style='padding: 10px; border-radius:5px 30px 30px 5px; border-style: solid; border-width: 1px;'>
  <summary>Tabel of Contents</summary>
  <ol>
    <li>
      <a href="#installation">Installation</a>
    </li>
    <li>
      <a href="#data-preparation">Data Preparation</a>
    </li>
    <li>
      <a href="#training">Training</a>
    </li>
    <li>
      <a href="#inference">Inference</a>
    </li>
    <li>
      <a href="#acknowledgement">Acknowledgement</a>
    </li>
    <li>
      <a href="#citation">Citation</a>
    </li>
  </ol>
</details>

## News
- [2025-06-17] We have released the training and inference code!

## Installation
Create a conda environment and install the required packages:
```bash
conda create -n mirage python=3.10
conda activate mirage

git clone https://github.com/UMass-Embodied-AGI/Mirage.git
cd Mirage
pip install -r requirements.txt
pip install -e ./transformers/.
```

## Data Preparation
We provide a sample dataset of 100 examples for the VSP spatial reasoning task. Please format your data file as follows:

## Training

```bash
python src/main.py \
    --model Qwen/Qwen2.5-VL-7B-Instruct --epochs 10 \
    --task vsp-spatial-reasoning \
    --latent_size 4 --compress_strategy average --stage stage1 \
    --data_path PathToTrainData \
    --log_file ./log.txt \
    --save_model_path ./checkpoints/model_stage1 
```

```bash
python src/main.py \
    --model Qwen/Qwen2.5-VL-7B-Instruct --epochs 10 \
    --task vsp-spatial-reasoning \
    --latent_size 4 --compress_strategy average --stage stage2 \
    --data_path PathToTrainData \
    --log_file ./log.txt \
    --load_model_path ./checkpoints/model_stage1
    --save_model_path ./checkpoints/model_stage2 
```

<!-- To pre-train the full TesserAct model from CogVideoX, we provide a training script based on [Finetrainers](https://github.com/a-r-r-o-w/finetrainers). The training code supports distributed training with multiple GPUs or multi-nodes.

To pre-train our TesserAct model, run the following command:
```bash
bash train_i2v_depth_normal_sft.sh
```

To fine-tune our released TesserAct model, modify the model loading code in [tesseract/i2v_depth_normal_sft.py](tesseract/i2v_depth_normal_sft.py):
```python
transformer = CogVideoXTransformer3DModel.from_pretrained_modify(
    "anyeZHY/tesseract",
    subfolder="tesseract_v01e_rgbdn_sft",
    ...
)
```

> [!NOTE]
> We will give a detailed training guide in the future: why TesserAct has better generalization, how to set the hyperparameters and performance between different training methods (SFT vs LoRA).
>
> We will release LoRA fine-tuning code in the future for more efficient training.
>
> We don't have a clear plan for releasing the whole dataset yet, because depth data is usually stored as floats, which takes up a lot of space and makes uploading to Hugging Face very difficult. However, we will provide scripts later on to show how to prepare the data. -->

## Inference

<!-- Now TesserAct includes following models. The names of the models are in the format of `anyeZHY/tesseract/` (huggingface repo name) + `<model_name>_<version>_<modality>_<training_method>`. In `<version>`, postfix `p` indicates the model is production-ready and `e` means the model is experimental. We will keep updating the model weights and scaling the dataset to improve the performance of the models.
```
anyeZHY/tesseract/tesseract_v01e_rgbdn_sft
anyeZHY/tesseract/tesseract_v01e_rgb_lora
```

> [!IMPORTANT] 
> It is recommended to read [USAGE.MD](doc/usage.md) for more details **before running the inference code on your own data.**
We provide a guide on how to prepare inputs, such as text prompt. We also analyze the model's limitations and performance, including:
>
> - Tasks that the model can reliably accomplish.
>
> - Tasks that are achievable but with certain success rates. In the future, this may be improved by using techniques like test-time scaling.
>
> - Tasks that are currently beyond the model's capabilities.

You can run the inference code with the following command (Optional flags: `--memory_efficient`).
```bash
python inference/inference_rgbdn_sft.py \
  --weights_path anyeZHY/tesseract/tesseract_v01e_rgbdn_sft \
  --image_path asset/images/fruit_vangogh.png \
  --prompt "pick up the apple google robot"
```
This inference code will generate a video of the google robot picking up the apple in the Van Gogh Painting. Try other prompts like `pick up the pear Franka Emika Panda`! Or `asset/images/majo.jpg` with prompt `Move the cup near bottle Franka Emika Panda`!

You may find output videos in the `results` folder.
Note: When we test the model on another server, the results are exactly the same as those we uploaded to GitHub.
So if you find they are different and get unexpected results like noisy videos, please check your environment and the version of the packages you are using.

> [!WARNING]
> Because RT1 and Bridge normal data is generated by [Temporal Marigold](https://huggingface.co/docs/diffusers/en/using-diffusers/marigold_usage#frame-by-frame-video-processing-with-temporal-consistency), sometimes normal outputs are not perfect. We are working on improving the data using [NormalCrafter](https://github.com/Binyr/NormalCrafter).

Below is a list of TODOs for the inference part.
- [ ] LoRA inference code
- [ ] Blender rendering code (check package [PyBlend](https://github.com/anyeZHY/PyBlend)!)
- [ ] Normal Integration -->



## Acknowledgements
We would like to thank the following works for their code and models:
- Training: [Coconut](https://arxiv.org/abs/2412.06769), [Qwen](https://huggingface.co/Qwen) and [MVoT](https://arxiv.org/pdf/2501.07542)
- Datasets: [VSP](https://arxiv.org/abs/2407.01863), [Blink](https://zeyofu.github.io/blink/), [COMT](https://arxiv.org/abs/2412.12932) and [SAT](https://arxiv.org/abs/2412.07755)

We are extremely grateful to Haoyu Zhen, Bairu Hou, Guangtao, Zeng, Yuncong Yang, Ziwei Liu,
Jiaben Chen, Zonghan Yang, Sunli Chen, Lixing Fang, and many other friends in our [Embodied AGI Lab](https://embodied-agi.cs.umass.edu/)
for their helpful feedback and insightful discussions.


## Citation
If you find our work useful, please consider citing:
<!-- ```bibtex
@article{zhen2025tesseract,
  title={TesserAct: Learning 4D Embodied World Models}, 
  author={Haoyu Zhen and Qiao Sun and Hongxin Zhang and Junyan Li and Siyuan Zhou and Yilun Du and Chuang Gan},
  year={2025},
  eprint={2504.20995},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2504.20995}, 
}
``` -->