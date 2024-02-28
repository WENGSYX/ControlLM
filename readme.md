# ControlLM: Crafting Diverse Personalities for Language Models

<p align="center">
    <img alt="GitHub" src="https://img.shields.io/github/license/WENGSYX/ControlLM.svg?color=blue&style=flat-square">
    <img alt="GitHub repo size" src="https://img.shields.io/github/repo-size/WENGSYX/ControlLM">
    <img alt="GitHub top language" src="https://img.shields.io/github/languages/top/WENGSYX/ControlLM">
    <img alt="GitHub last commit" src="https://img.shields.io/github/last-commit/WENGSYX/ControlLM">
</p>



**Authors**: Yixuan Weng, Shizhu He, Kang Liu, Shengping Liu, Jun Zhao 😎

**[Contact]** If you have any questions, feel free to contact me via (wengsyx@gmail.com).

This repository contains code, models, and other related resources of our paper ["ControlLM: Crafting Diverse Personalities for Language Models"](https://arxiv.org/abs/2304.01665).

* [2024/02/15] We have published the paper!
* [2024/02/14] We created the Github library!

****

**Welcome to the ControlLM Project - ControlLM is a method to control the personality traits and behaviors of language models in real-time at inference without costly training interventions.**🚀🚅

### Setup

**Install**

```
git clone https://github.com/WENGSYX/ControlLM
pip install .
```

To run neural comprehension, you need to install `PyTorch` and`transformers`.
### How to use the ControlLM?

```python
from ControlLM.llama import get_model

model, tokenizer = get_model(model_name='meta-llama/Llama-2-7b-chat-hf')
```

ControlLM supports `Llama` and `Falcon` models. If you need to use the `Falcon` model, just replace `ControlLM.llama` with `ControlLM.falcon`.



#### Get and Save Control Activate

```python
from datasets import load_dataset

personal = 'Conscientiousness'
path = './Activate/' + personal

datasets = load_dataset('WENGSYX/ControlLM_Personalities')[personal].to_list()
model.get_and_save_activations(dataset=datasets, save_path=path)

>>> Processing prompts  99% ━━━━━━━━━━━━━━╸ 163/164  [ 0:00:10 < 0:00:01 , 17 it/s ]
```

We have pre-processed the text pairs required for extracting personalities. In `/preprocess_dataset` we provide the corresponding data processing code.



Please use `datasets.load_dataset` to load [them](https://huggingface.co/datasets/WENGSYX/ControlLM_Personalities) and use `model.get_and_save_activations for processing`.



In ControlLM, we provide the behavioral patterns for the following personalities:

> ```
> 'Extraversion',
> 'Neuroticism',
> 'Conscientiousness',
> 'Agreeableness',
> 'Openness'
> ```



#### Control Language Models

Let's use the `load_and_set_activate` function to set the control activates, then illustrate what the language model discourse with a Conscientiousness personality would look like with an example :



```
model.load_and_set_activate(load_path=path, layer=2, gamma=1.5)

print(tokenizer.batch_decode(model.generate_text('When working on a group project, what role do you usually take? How do you ensure all members contribute equally?',256)))

>>> Great, I'm glad to be of assistance! 😊
When working on a group project, I usually take on a facilitation role, ensuring that all members are on the same page and contributing equally. Here are some strategies I use to achieve this:
1. Clearly define the project's objectives and expectations: I make sure that everyone is aware of the project's goals, deadlines, and what is expected of them. This helps to avoid confusion and ensures that everyone is working towards the same objectives.
2. Establish a clear communication plan: I encourage group members to communicate openly and regularly, using a designated platform (e.g., project management tool, email, or group chat). This helps to ensure that everyone is informed and can provide input when needed.
3. Assign specific tasks and responsibilities: I work with the group to identify the tasks that need to be completed and assign them to each member based on their strengths and expertise. This helps to ensure that everyone is contributing to the project in a meaningful way.

```





### Re-set

```
model.reset_all()
```

We can easily reset our language model!



#### Batch Generate

```
text = ['Walk me through your decision-making process when taking on a new responsibility or task. What factors do you consider?',
        'Tell me about a time you were disorganized or fell behind schedule on something. What was the result and what did you learn?',
        'Describe a group project experience where there was unequal participation. Why do you think that happened and would you approach it differently next time?']
        
        
output = model.generate([tokenizer(t).input_ids for t in text], max_length=512)
output_text = tokenizer.batch_decode(output)
```



###### 

### Bench

We have included the code required to reproduce the `/Bench/MPI,` `/Bench/Reasoning`, and `/Bench/Language_Modeling`  in the `Bench` folder respectively. Please browse the corresponding folders! 



### AutoControlActivate

Considering existing text datasets can hardly encompass all personality traits, we introduce AutoControlActivate to obtain corresponding behavioral traits of personalities from popular language models, and automatically fabricate control activations based on this 😃



```
# https://beta.openai.com/account/api-keys
export OPENAI_API_KEY=(YOUR OPENAI API KEY)
```

Firstly, you need to set `OpenAI API Key` for `AutoControlActivate`



```
from ControlLM.AutoControlActivate import make_dataset

personal = 'Obsequiousness'
path = './Activate/' + personal

dataset = make_dataset(Personality = personal)
model.get_and_save_activations(dataset=datasets, save_path=path)
```



**Only need 1 minute! Successed!** 





### 🙏Cite🙏


###### If you are interested in our paper, please feel free to cite it.
```
@misc{weng2023mastering,
      title={Mastering Symbolic Operations: Augmenting Language Models with Compiled Neural Networks}, 
      author={Yixuan Weng and Minjun Zhu and Fei Xia and Bin Li and Shizhu He and Kang Liu and Jun Zhao},
      year={2023},
      eprint={2304.01665},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
