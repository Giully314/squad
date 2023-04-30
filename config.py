from dataclasses import dataclass
import torch

@dataclass
class Paths:
    data_dir: str
    output_dir: str
    checkpoint_dir: str
    test_output_dir: str

@dataclass
class Generic:
    device: str
    fix_random: bool
    debug: bool

    def __post_init__(self):
        self.device = torch.device(self.device)


@dataclass
class Dataloader:
    batch_size: int
    shuffle: bool
    num_workers: int
    persistent_workers: bool 
    pin_memory: bool 


@dataclass 
class Dataset:
    train_file: str
    valid_file: str
    test_file: str
    word_emb_file: str
    char_emb_file: str
    word_to_idx_file: str
    char_to_idx_file: str


@dataclass
class Model:
    char_cnn_kernel_width: int
    char_cnn_channels: int 
    hidden_dim: int
    contextual_layers: int
    contextual_dropout: float 
    attention_dropout: float
    modeling_layers: int
    modeling_dropout: float

@dataclass
class Optimizer:
    lr: float
    beta1: float
    beta2: float
    weight_decay: float
    fused: bool

@dataclass
class Scheduler:
    min_lr: float 
    t_max: int


@dataclass
class Train:
    epochs: int
    evaluate_every_n_epochs: int
    start_from_checkpoint: bool
    gradient_clipping_max_norm: float


@dataclass
class Test:
    should_test: bool
    num_visuals: int
    test_file: str



@dataclass
class ProjectConfig:
    paths: Paths
    generic: Generic
    dataloader: Dataloader
    dataset: Dataset
    model: Model
    optimizer: Optimizer
    scheduler: Scheduler
    train: Train
    test: Test