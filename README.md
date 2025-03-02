## Official implementation of [Calibrating LLMs with Information-Theoretic Evidential Deep Learning (ICLR 2025)](https://openreview.net/forum?id=YcML3rJl0N)

### Installation

1. Install "setuptools": `pip install setuptools`.
2. Git clone this repository.
3. Navigate to the root level of the repository (where `setup.cfg` is located) and run `pip install -e .`
   (Note: Don't forget the dot `.` at the end of the command).
4. (Optional) Run `huggingface-cli login` to log in to HuggingFace-Hub, and use `wandb login` to log in to WandB.
5. (Optional) Go through the Docs of [mmengine.Config](https://mmengine.readthedocs.io/en/latest/api/generated/mmengine.config.Config.html#config)
   to know how to use the `Config`.

### IB-EDL Training

Assume you want to fine-tune Llama3-8B on the OBQA dataset using IB-EDL. Run the following command:

```bash
python tools/evidential_ft.py configs/obqa_llama3_8b/ib_obqa_llama3_8b.yaml \
  -w workdirs/ib_edl/obqa/ \
  -n ib_obqa_llama3_8b 
```
To run the training with a different IB regularization strength, you can use `-o vib.beta=NEW-VALUE` in the training 
command.

Since the configuration file contains the following entry:
```yaml
process_preds:
  npz_file: "obqa.npz"
```
The training program will save the predictions as a file named `obqa.npz`.

### OOD Detection
After completing the fine-tuning using the command above, you should have:
* A LoRA checkpoint of Llama3-8B trained on OBQA.
* The `obqa.npz` file, which contains predictions on the OBQA dataset (assumed to be the in-distribution (ID) dataset
  for OOD detection).

The OOD detection can be done as follows:

Step 1: Obtain predictions on the OOD Dataset: Assume that the CSQA dataset is the OOD dataset. You can generate predictions 
on this dataset using the checkpoint trained on OBQA:
```bash
python tools/evidential_ft.py configs/ood_llama3_8b/ib_obqa_csqa_llama3_8b.yaml \
  -w workdirs/ib_edl/csqa/ \
  -s \
  -o model.peft_path=workdirs/ib_edl/obqa/checkpoint-XXX
```
This command will evaluate the model on CSQA and store the predictions in a file named `csqa.npz`. 

Step 2: Run OOD detection script as follows:
```bash
python tools/ood.py workdirs/ib_edl/obqa/obqa.npz workdirs/ib_edl/csqa/csqa.npz
```

### Post-hoc Calibration Technique

In the Appendix of the paper, we introduced a post-hoc calibration technique to further enhance the performance of 
IB-EDL. To use this technique, follow these steps:

Step 1: Visualize the calibration curve: Open a Jupyter Notebook, load the predictions using `numpy`, and use 
`ib_edl.plot_calibration_curve_and_ece` to visualize the calibration curve.

Step 2: Set the sigma multiplier value: Based on the calibration curve, choose an appropriate value for the sigma 
multiplier in the post-hoc calibration technique. Then, re-run the inference on the validation set with the chosen sigma multiplier:

```bash
python tools/evidential_ft.py path/to/config.yaml \
  -s \
  -o model.peft=path/to/model/checkpoint-XXX vib.sigma_mult=NEW-VALUE
```
Or you can modify the configuration file directly by updating the following entry:
```yaml
vib:
  sigma_mult: NEW-VALUE
```
It is recommended to repeat this process on the validation set to determine the best hyperparameter value for 
`vib.sigma_mult`.

### Adaptation to Generative Tasks
Currently, IB-EDL is implemented only for multiple-choice QA tasks, which follow a classification setting. To extend it 
for open-ended generation tasks, some adaptations to the implementation are required. Contributions are welcome—feel 
free to submit a pull request!

### Citation
```latex
@inproceedings{
    li2025calibrating,
    title={Calibrating {LLM}s with Information-Theoretic Evidential Deep Learning},
    author={Yawei Li and David R{\"u}gamer and Bernd Bischl and Mina Rezaei},
    booktitle={The Thirteenth International Conference on Learning Representations},
    year={2025},
    url={https://openreview.net/forum?id=YcML3rJl0N}
}
```