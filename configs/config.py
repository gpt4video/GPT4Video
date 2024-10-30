import argparse
import ast


parser = argparse.ArgumentParser(description="hyper-parameter for GVLL")

# ====================== Dataset Config ===========================
parser.add_argument('--dataset', metavar='DATASET', default='./data/WebVid_tiny/annotation/meta.json', help='training datasets to use')
parser.add_argument('--base_dir', default='./data/WebVid_tiny', type=str, help='Dataset directory containing image folders.')
parser.add_argument('--savedmodel_path', default='./runs/v1', type=str, help='Dataset directory containing image folders.')
parser.add_argument('--ckpt_file', default=None, type=str, help='Dataset directory containing image folders.')
parser.add_argument('--delta_file', default=None, type=str, help='Dataset directory containing image folders.')
parser.add_argument('--text_embed', default='clip_embeds_zero', type=str, help='Dataset directory containing image folders.')
parser.add_argument('--use_embed', default=False, type=lambda x: (str(x).lower() == 'true'), help='load video embedding or video')

# ====================== Model Config ===========================
parser.add_argument('--llm_model', default='MAGAer13/mplug-owl-llama-7b', help='LLM to use, meta-llama/Llama-2-7b-chat-hf')
parser.add_argument('--freeze_vm', default=True, type=lambda x: (str(x).lower() == 'true'), help="freeze visual model or not")
parser.add_argument('--freeze_llm', default=True, type=lambda x: (str(x).lower() == 'true'), help="freeze llm model or not")
parser.add_argument('--llm_use_lora', default=True, type=lambda x: (str(x).lower() == 'true'), help="freeze llm model or not")
parser.add_argument('--lora_inference', default=False, type=lambda x: (str(x).lower() == 'true'), help="freeze llm model or not")
parser.add_argument('--llm_r', default=16, type=int, help='The dimension used by the LoRA update matrices')
parser.add_argument('--llm_alpha', default=16, type=int, help='Scaling factor.')
parser.add_argument('--lora_dropout', default=0.1, type=float, help='lora dropout')

# ====================== Training Config ===========================
parser.add_argument('--batch_size', default=2, type=int, help='mini-batch size for training')
parser.add_argument('--val_batch_size', default=2, type=int, help='mini-batch size for validation')
parser.add_argument('--num_workers', default=6, type=int, metavar='N', help='number of data loading workers (default: 4)')
parser.add_argument('--prefetch_factor', default=2, type=int, metavar='N', help='Number of batches loaded in advance by each worker')
parser.add_argument('--learning_rate', default=0.0001, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--num_frames', default=8, type=int, metavar='N', help='Number of frames to use.')

parser.add_argument('--max_length', default=512, type=int, metavar='N', help='Maximum length to truncate captions / generations to.')
parser.add_argument('--repetition_penalty', type=float, default=1)
parser.add_argument('--length_penalty', type=float, default=1.0)
parser.add_argument('--diversity_penalty', type=float, default=0)
parser.add_argument('--temperature', type=float, default=0.1)
parser.add_argument('--beam_size', type=int, default=3)
parser.add_argument('--do_sample', type=bool, default=False)

===================== Pytorch Lightning ===========================
parser.add_argument('--devices', type=int, default=1, help='how many gpus to use')
parser.add_argument('--num_nodes', type=int, default=1, help='Number of GPU nodes for distributed training.')
parser.add_argument('--accelerator', type=str, default="gpu", choices=["cpu", "gpu", "tpu", "ipu", "hpu", "mps"], help='accelerator types')
parser.add_argument('--strategy', type=str, default="deepspeed", help='default ddp for multi-gpus')
parser.add_argument('--precision', type=str, default='16', help='16 or 32 bf16-mixed, using for original pytorch amp auto cast')
parser.add_argument('--limit_val_batches', type=float, default=1.0, help='How much of validation dataset to check (float = fraction, int = num_batches).')
parser.add_argument('--limit_train_batches', type=float, default=1.0, help='How much of training dataset to check (float = fraction, int = num_batches)')
parser.add_argument('--max_steps', default=1500000, type=int, metavar='N', help='Stop training after this number of steps. ')
parser.add_argument('--max_epochs', type=int, default=30, help='Stop training once this number of epochs is reached')
parser.add_argument('--every_n_train_steps', type=int, default=0, help='How many training steps to save a checkpoint')
parser.add_argument('--every_n_epochs', type=int, default=0, help='How many training epochs to save a checkpoint')
parser.add_argument('--val_check_interval', type=float, default=1.0, help='How often to check the validation set')
parser.add_argument('--log_every_n_steps', type=int, default=20, help='How often to log within steps')
parser.add_argument('--accumulate_grad_batches', type=int, default=1, help='Accumulates gradients over k batches before stepping the optimizer')
parser.add_argument("--num_sanity_val_steps", type=int, default=2, help='Sanity check runs n validation batches before starting the training routine')
