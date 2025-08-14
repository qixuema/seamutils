import yaml
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List, ClassVar
import os
class NestedDictToClass:
    def __init__(self, dictionary):
        self._convert(dictionary)
    
    def _convert(self, dictionary):
        for key, value in dictionary.items():
            if isinstance(value, dict):
                value = NestedDictToClass(value)
            elif isinstance(value, list):
                value = tuple(value)

            setattr(self, key.lower(), value)

# General config
def load_config(path, default_path=None):
    ''' Loads config file.

    Args:  
        path (str): path to config file
        default_path (bool): whether to use default path
    '''
    # Load configuration from file itself
    with open(path, 'r') as f:
        cfg_special = yaml.safe_load(f)

    # Check if we should inherit from a config
    inherit_from = cfg_special.get('inherit_from')

    # If yes, load this config first as default
    # If no, use the default_path
    if inherit_from is not None:
        cfg = load_config(inherit_from, default_path)
    elif default_path is not None:
        with open(default_path, 'r') as f:
            cfg = yaml.load(f)
    else:
        cfg = dict()

    # Include main configuration
    update_recursive(cfg, cfg_special)

    return cfg

def update_recursive(dict1, dict2):
    ''' Update two config dictionaries recursively.

    Args:
        dict1 (dict): first dictionary to be updated
        dict2 (dict): second dictionary which entries should be used

    '''
    for k, v in dict2.items():
        if k not in dict1:
            dict1[k] = dict()
        if isinstance(v, dict):
            update_recursive(dict1[k], v)
        else:
            dict1[k] = v


def print_custom_attributes(obj, indent=''):
    attributes = vars(obj)
    for attr, value in attributes.items():
        if not attr.startswith("__"):
            if isinstance(value, NestedDictToClass):
                print(f"{indent}{attr}:")
                print_custom_attributes(value, indent + "  ")
            else:
                print(f"{indent}{attr}: {value}")

class AugmentDict(BaseModel):
    scale_min: float = 1.0
    scale_max: float = 1.0
    rotation: int = 0
    jitter_strength: float = 0.01
    jitter_mask: bool = False
    adaptive_jitter: bool = False

# class ModelConfig(BaseModel):
#     architecture: str = 'plain'
#     copilot_architecture: Optional[str] = None
#     dim: int = 1024
#     depth: int = 24
#     attn_depth: Optional[str] = None  # For hourglass model when depth=-1
#     shorten_factor: Optional[str] = None  # For hourglass model when depth=-1
#     attn_dim_head: int = 64
#     attn_heads: int = 16
#     max_seq_len: int = 600
#     dropout: float = 0.0
#     ff_dropout: float = 0.0
#     quant_bit: int = 7
#     pc_encoder_name: str = 'miche-point-query-structural-vae'
#     kl_weight: float = 0.0
#     use_rope: bool = False
#     train_truncated: bool = True
#     truncated_len: int = 36864
#     length_cond: bool = True
#     quad_cond: bool = True
#     use_uncond_face_token: bool = False
#     cross_attn_interval: bool = True
#     shift_type: str = 'post'
#     use_qk_norm: bool = False
#     cross_attn_qk_norm: bool = False
#     new_cond_proj: bool = True
#     use_ori_abs_pos_emb: bool = False
#     mask_pc_embeds: bool = True
#     dropout_faces: bool = True
#     use_face_mask: bool = True
#     decoder_type: str = 'causal'
#     fix_cond_len: bool = True
#     bos_token_id: int = 1
#     eos_token_id: int = 2
#     pad_token_id: int = 0
#     checkpointing: bool = False
#     block_size: int = 16
#     offset_size: int = 16
#     in_context: bool = False
#     use_cfg: bool = False
#     random_crop: bool = False
#     random_crop_p: float = 0.2
#     min_seg_ratio: float = 0.1
#     use_segmentation: bool = False
#     seg_token: int = -3
#     use_segmentation_p: float = 0.2
#     token_format: str = 'xyzxyz'
#     clip_partbox_count: int = 32
#     use_rotation: bool = False
#     rotation_format: str = '6d'

# class DatasetConfig(BaseModel):
#     dataset_name: str = 'baseline'
#     dataset_paths: Dict[str, str] = {
#         'baseline': 'splits/all_data.json',
#         'scale_up': 'splits/all_data_full_data.json',
#         'scale_up_obb': 'splits/all_data_full_data_obb.json'
#     }
#     sort_verts_in_face: bool = False
#     sort_faces_by_components: bool = False
#     remove_inside_points: bool = False
#     sample_surface_pc: bool = False
#     use_surface_prob: float = 1.0
#     pc_num: int = 40960
#     merge_verts: bool = True
#     important_sampling: bool = False
#     random_scale: bool = False
#     normalization_scale: float = 0.99
#     quantization_scale: float = 1.0
#     conditioned_on_pc: bool = True
#     augment: bool = True
#     augment_dict: AugmentDict = Field(default_factory=AugmentDict)
#     smooth_augment: bool = False
#     jitter_pc: bool = False
#     use_augment_codes_permute_bounds: bool = False
#     rescale_pc: bool = True

# class TrainingConfig(BaseModel):
#     resume: Optional[str] = None
#     batch_size: int = 2
#     pc_encoder_freeze: bool = False
#     copilot_freeze: bool = False
#     pc_encoder_freeze_before_kl: bool = False
#     learning_rate: float = 0.0001
#     grad_accum_every: int = 1
#     warmup_steps: int = 100
#     weight_decay: float = 0.0
#     max_grad_norm: float = 0.5
#     codes_path: Optional[str] = None
#     checkpoint_every: int = 2000
#     val_every: int = 500000
#     num_train_steps: int = 500000

class Config(BaseModel):
    # exp_name: str
    # wandb_run_name: str
    # checkpoint_folder: str
    model: Dict
    dataset: Dict
    training: Dict

    @classmethod
    def from_yaml(cls, path: str) -> 'Config':
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        return cls(**data)


def load_config_CHUNSHI(config_path: str, cli_args: Optional[List[str]] = None) -> Config:
    with open(config_path, 'r') as f:
        configs = yaml.safe_load(f)

    if 'base_config' in configs:
        base_config_path = os.path.join(os.path.dirname(config_path), configs['base_config'])
        with open(base_config_path, 'r') as f:
            base_data = yaml.safe_load(f)
        
        def deep_update(d, u):
            for k, v in u.items():
                if isinstance(v, dict):
                    d[k] = deep_update(d.get(k, {}), v)
                else:
                    d[k] = v
            return d
        
        configs = deep_update(base_data, configs)

    config = Config(
        model=configs['MODEL'], 
        dataset=configs['DATA'], 
        training=configs['TRAINING']
    )

    if cli_args:
        for arg in cli_args:
            key, value = arg.split('=', 1)
            keys = key.split('.')
            c = config
            for k in keys[:-1]:
                c = getattr(c, k)
            
            field_type = c.__annotations__[keys[-1]]
            if field_type == bool:
                value = value.lower() in ['true', '1', 't', 'y', 'yes']
            else:
                value = field_type(value)

            setattr(c, keys[-1], value)
            
    return config

if __name__ == '__main__':
    cfg = load_config('train.yaml')
    args = NestedDictToClass(cfg)

    print_custom_attributes(args)