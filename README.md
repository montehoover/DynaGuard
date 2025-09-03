# DynaGuard

DynaGuard: A Dynamic Guardrail Model With User-Defined Policies.

A  project by: Monte Hoover, Vatsal Baherwani, Neel Jain, Khalid Saifullah, Joseph Vincent, Chirag Jain, Melissa Kazemi Rad, C. Bayan Bruss, Ashwinee Panda, and Tom Goldstein.

<p align="center">
<a target="_blank" href="https://arxiv.org/abs/2509.02563">
<img style="height:22pt" src="https://img.shields.io/badge/-Paper-B31B1B?style=flat&logo=arxiv">
<a target="_blank" href="https://taruschirag.github.io/DynaGuard/">
<img style="height:22pt" src="https://img.shields.io/badge/-🌐%20Website-1E8BC3?style=flat">
<a target="_blank" href="https://huggingface.co/collections/tomg-group-umd/dynaguard-68af4d916ae81d06ef774523">
<img style="height:22pt" src="https://img.shields.io/badge/-🤗%20Models-red?style=flat"></a>
<br>
</p>


## Getting Started
1. Install dependencies:
    ```
    pip install -r requirements.txt
    ```
2. Evaluate a model on custom policies:
    ```
    python eval.py
    ```
    or
    ```
    python eval.py --model tomg-group-umd/DynaGuard-8B --dataset tomg-group-umd/DynaBench --subset DynaBench --split test
    ```


## Citing Our Work
To cite our work, please use this bibtex.
```
@article{hoover2025dynaguard,
    title={DynaGuard: A Dynamic Guardrail Model With User-Defined Policies}, 
    author={Monte Hoover and Vatsal Baherwani and Neel Jain and Khalid Saifullah and Joseph Vincent and Chirag Jain and Melissa Kazemi Rad and C. Bayan Bruss and Ashwinee Panda and Tom Goldstein},
    journal={arXiv preprint},
    year={2025},
    url={https://arxiv.org/abs/2509.02563}, 
}
```

## Contact
Please, feel free to contact us with any questions, or open an issue on Github.

