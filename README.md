# LLM-FineTune
Repository for the Assignment 1.C for the NLP course.

## Usage

### Environment
You can install the required dependencies with the following command:
```bash
conda env create -f environment.yml
```

Then activate the environment:
```bash
conda activate torch-main
```

### Training

Everything in this project is controlled via hydra configurations. you can take a look at the default config by running this command:
```bash
python train.py --cfg job
```
each and every entry of the configuration you see is changable both through the config files and through command line arguments. you can change any value by simply typing its dot path and assign a new value. For example:
```bash
python train.py data.split_seed=2024 --cfg job
```
will print the same configuration as before with only the split_seed value changed.

to actually run the training, you can run:
```bash
accelerate launch --config_file accelerate_config.yaml train.py
```

you can change the model by creating a model configuration file or use one of the pre-created configs. For example, to use Mistral instead of Llama-2, run:
```bash
accelerate launch --config_file accelerate_config.yaml train.py +experiment=train-mistral
```

### Evaluation

Evaluation is alsp controlled through the same framework. to evaluate the model on some popular metrics that are defined in `conf/experiment/eval.yaml`, run:
```bash
accelerate launch --config_file accelerate_config.yaml eval.py +experiment=eval
```
This will use the default model, Llama-2, to generate responses and evaluate using the metrics, to change the model, run:
```bash
accelerate launch --config_file accelerate_config.yaml eval.py model=mistral +experiment=eval 
```