import json
import numpy as np
from numpy.random import randint
import torch.utils.data as data
from transformers import AutoTokenizer
from models.mplug_owl_video.processing_mplug_owl import MplugOwlImageProcessor, MplugOwlProcessor

pretrained_ckpt = 'MAGAer13/mplug-owl-llama-7b'

class FieldParser:
    def __init__(
            self,
            args
    ):
        super().__init__()
        self.args = args
        self.base_dir = args.base_dir
        self.text_embed = args.text_embed
        image_processor = MplugOwlImageProcessor.from_pretrained(pretrained_ckpt)
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_ckpt)
        self.processor = MplugOwlProcessor(image_processor, self.tokenizer)
        self.system = "The following is a conversation between a curious human and AI assistant GPT4Video. GPT4Video generates video prompts at the most appropriate time and gives helpful, detailed, and polite answers to the user's questions.\n"

    def tokenize(self, text):
        out = self.tokenizer(
            text,
            return_tensors="pt",
            padding='longest',
            truncation=True,
            max_length=self.args.max_length)
        input_ids = out.input_ids[0]
        return input_ids

    def parse(self, features):
        if "video" in features:
            video_path = features['video']
            pixel_values = self.processor.process_video([video_path]).squeeze()
        
        instruct = features.get("instruction", "")
        # If instruction too long, cut it off
        sp = instruct.split('\n')
        if len(sp) > 10:
            instruct = "\n".join(instruct.split('\n')[:10])
            
        # 统一输入格式
        # instruct = instruct.replace('</s>', "").replace('Assistant', "AI")
        # instruct = instruct.replace('<vid0>', '<video>').replace('</vid0>', '</video>').replace('<vid1>', '<video>').replace('</vid1>', '</video>')
        instruct = self.system + instruct
        input_ids = self.tokenize(instruct)

        if "video" in features:
            to_return = {
                "video": pixel_values,
                "input_ids": input_ids,
            }
        else:
            to_return = {
                "input_ids": input_ids,
            }
        return to_return


    def transform_with_parse(self, inputs):
        return self.parse(inputs)
    

class ParseDataset(data.Dataset):
    def __init__(self, args, split='train'):
        # pdb.set_trace()
        self.train = split == "train"
        meta = json.load(open(args.dataset, 'r'))
        if split == "train":
            self.df = meta['train']
        else:
            self.df = meta['test']
        self.parser = FieldParser(args)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        try:
            return self.parser.transform_with_parse(self.df[index])
        except Exception as e:
            print(f'Error reading for {self.df[index]["image_id"]} with caption {self.df[index]["caption"]}: {e}')
            # Pick a new example at random.
            idx = np.random.randint(0, len(self.df)-1)


def create_datasets(args):
    train_dataset = ParseDataset(args, 'train')
    dev_dataset = ParseDataset(args, 'val')
    test_dataset = ParseDataset(args, 'test')
    return train_dataset, dev_dataset, test_dataset

