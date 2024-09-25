from src.finetuning import finetune_models
from src.finetuning.dataset import build_finetuning_datasets
from src.finetuning.evaluate_seed import evaluate_seed_datasets

if __name__ == "__main__":
    evaluate_seed_datasets()
    build_finetuning_datasets()
    finetune_models()
