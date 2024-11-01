# GPT4Video

**[GPT4Video: A Unified Multimodal Large Language Model for lnstruction-Followed Understanding and Safety-Aware Generation](https://arxiv.org/abs/2311.16511)**

[Zhanyu Wang](https://wang-zhanyu.github.io/), [Longyue Wang](http://www.longyuewang.com/)\*, [Zhen Zhao](http://zhaozhen.me/), [Minghao Wu](https://minghao-wu.github.io/), [Chenyang Lyu](https://lyuchenyang.github.io/), [Huayang Li](https://sites.google.com/view/huayangli), [Deng Cai](https://jcyk.github.io/), [Luping Zhou](https://sites.google.com/view/lupingzhou)\*, [Shuming Shi](https://shumingshi.github.io/), [Zhaopeng Tu](http://www.zptu.net/)

**Tencent AI Lab**, **University of Sydney**  (\*Correspondence)

<div align="center">
<a href='https://gpt4video.github.io/'><img src='https://img.shields.io/badge/Project-Page-Green'></a> <a href='https://arxiv.org/abs/2311.16511'><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a> <a href="https://github.com/gpt4video/GPT4Video"><img src='https://img.shields.io/badge/Resource-Data-blue'></a>
</div>

## ✨ Demo video
[Demo Video](https://github.com/gpt4video/GPT4Video/assets/151513068/739b05f4-a945-4519-9cae-83e1d35c7a1e)

## Framework
![image-20230924124604776](__assets__/framework.png)
**Video Encoding stage:** The video encoding module employs a frozen ViT-L/14 model to capture raw video features, while the video abstraction module utilizes a transformer-based cross attention layer and two novel learnable tokens, designed to condense information along the temporal and spatial axes.

**LLM reasoning:** The core of GPT4Video is powered by a frozen LLaMA model, efficiently fine-tuned via LoRA. The LLM is trained with custom video-centric and safety-aligned data, enabling it to comprehend videos and generate appropriate video prompts (_indicated by underlined text_).

**Video Generation:** The prompts generated by LLM are then used as text inputs for the models in the Text-to-Video Model Gallery to create videos. We use ZeroScope as our video generation model in this work.


## Training
first, install the requestments.
```shell
   pip install -r requestments.txt
```

training model with two gpus for 10 epoches.
```python
    python train.py --devices 2 --max_epochs 10
```

## Citation

```
@articles{wang2023gpt4video,
  title={GPT4Video: A Unified Multimodal Large Language Model for lnstruction-Followed Understanding and Safety-Aware Generation},
  author={Zhanyu Wang, Longyue Wang, Minghao Wu, Zhen Zhao, Chenyang Lyu, Huayang Li, Deng Cai, Luping Zhou, Shuming Shi, Zhaopeng Tu},
  journal = {CoRR},
  year={2023}
}
```
