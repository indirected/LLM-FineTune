import wandb
import hydra
import torch
import datasets
from omegaconf import DictConfig
import transformers
import trl
import peft
import torchinfo
import accelerate
import wandb
import os
import utils


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg):
    # print(cfg.data.ttt)
    # exit()

    os.environ["WANDB_PROJECT"] = cfg.wandb.project_name
    os.environ["WANDB_LOG_MODEL"] = cfg.wandb.log_model
    os.environ["WANDB_WATCH"] = cfg.wandb.wandb_watch

    accelerator = accelerate.Accelerator()

    dataset = utils.prepare_data(cfg.data)

    model = hydra.utils.instantiate(cfg.model.model_HF_cls)
    tokenizer = hydra.utils.instantiate(cfg.model.tokenizer_HF_cls)
    tokenizer.pad_token = tokenizer.eos_token

    peft_config = hydra.utils.instantiate(cfg.peft_config.HF_cls)
    trainer_args = hydra.utils.instantiate(
        cfg.trainer_args.HF_cls,
        output_dir=hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    )

    trainer = trl.SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=trainer_args,
        peft_config=peft_config,
        train_dataset=dataset['train'],
        eval_dataset=dataset['val'],
        dataset_text_field='text'
    )

    if accelerator.is_local_main_process:
        torchinfo.summary(trainer.model)

    trainer.train()
    wandb.finish()

    save_path = os.path.join(cfg.save_to, cfg.model.name, 'final')

    trainer.model.save_pretrained(save_path)
    trainer.tokenizer.save_pretrained(save_path)


if __name__ == "__main__":
    main()