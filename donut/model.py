"""
Donut
Copyright (c) 2022-present NAVER Corp.
MIT License
"""
import math
import os
import re
from typing import Any, List, Optional, Union

import numpy as np
import PIL
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import ImageOps
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.swin_transformer import SwinTransformer
from timm.models.swin_transformer_v2 import SwinTransformerV2
from timm.models.vision_transformer import VisionTransformer
from donut.swin_v2_with_vit import SwinV2WithVit
# from donut.swin_transformer import SwinTransformer
from torchvision import transforms
from torchvision.transforms.functional import resize, rotate
from transformers import MBartConfig, MBartForCausalLM, AutoTokenizer, XLMRobertaTokenizer, MBartModel, BertModel
from transformers.models.mbart.modeling_mbart import MBartEncoder
from transformers.file_utils import ModelOutput
from transformers.modeling_utils import PretrainedConfig, PreTrainedModel
from transformers import CLIPProcessor, CLIPModel


class SwinEncoder(nn.Module):
    r"""
    Donut encoder based on SwinTransformer
    Set the initial weights and configuration with a pretrained SwinTransformer and then
    modify the detailed configurations as a Donut Encoder

    Args:
        input_size: Input image size (width, height)
        align_long_axis: Whether to rotate image if height is greater than width
        window_size: Window size(=patch size) of SwinTransformer
        encoder_layer: Number of layers of SwinTransformer encoder
        name_or_path: Name of a pretrained model name either registered in huggingface.co. or saved in local.
                      otherwise, `swin_base_patch4_window12_384` will be set (using `timm`).
    """

    def __init__(
            self,
            input_size: List[int],
            align_long_axis: bool,
            window_size: int,
            encoder_layer: List[int],
            name_or_path: Union[str, bytes, os.PathLike] = None,
            vision_model_name='SwinTransformer',
            swin_pretrained_path='swin_base_patch4_window12_384_in22k',
            swin_model_size='base',
            ape=False,
            swin_name_or_path=None,
            depth_last_block=2,
            num_heads_last_block=8,  # must be "d_model / num_heads_last_block = 0"
            drop_path_rate_last_block=0.1,  # or 0.0
            init_values_last_block=None,  # or 1e-5
            ape_last_block=False
    ):
        super().__init__()
        self.input_size = input_size
        self.align_long_axis = align_long_axis
        self.window_size = window_size
        self.encoder_layer = encoder_layer
        self.vision_model_name = vision_model_name

        self.to_tensor = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
            ]
        )
        if swin_model_size == 'base':
            embed_dim = 128
            num_heads = [4, 8, 16, 32]
        elif swin_model_size == 'large':
            embed_dim = 192
            num_heads = [6, 12, 24, 48]

        if vision_model_name == "SwinTransformer":
            self.model = SwinTransformer(
                img_size=self.input_size,
                depths=self.encoder_layer,
                window_size=self.window_size,
                patch_size=4,
                embed_dim=embed_dim,
                num_heads=num_heads,
                num_classes=0,
                ape=ape
            )
        elif vision_model_name == "SwinTransformerV2":
            self.model = SwinTransformerV2(
                img_size=self.input_size,
                depths=self.encoder_layer,
                window_size=self.window_size,
                patch_size=4,
                embed_dim=embed_dim,
                num_heads=num_heads,
                num_classes=0,
                ape=ape
            )
        elif vision_model_name == "SwinV2WithVit":
            self.model = SwinV2WithVit(
                img_size=self.input_size,
                depths=self.encoder_layer,
                window_size=self.window_size,
                patch_size=4,
                embed_dim=embed_dim,
                num_heads=num_heads,
                num_classes=0,
                ape=ape,
                depth_last_block=depth_last_block,
                num_heads_last_block=num_heads_last_block,  # must be "d_model / num_heads_last_block = 0"
                drop_path_rate_last_block=drop_path_rate_last_block,  # or 0.1
                init_values_last_block=init_values_last_block,  # or 1e-5
                ape_last_block=ape_last_block
            )

        # weight init with swin
        if not swin_name_or_path:
            swin_state_dict = timm.create_model(swin_pretrained_path, pretrained=True).state_dict()
        else:
            checkpoint_dict = torch.load(swin_name_or_path)
            swin_state_dict = checkpoint_dict['state_dict']
        new_swin_state_dict = self.model.state_dict()
        for x in new_swin_state_dict:
            if x.endswith("relative_position_index") or x.endswith("attn_mask"):
                pass
            elif (
                    x.endswith("relative_position_bias_table")
                    and self.model.layers[0].blocks[0].attn.window_size[0] != 12
            ):
                pos_bias = swin_state_dict[x].unsqueeze(0)[0]
                old_len = int(math.sqrt(len(pos_bias)))
                new_len = int(2 * window_size - 1)
                pos_bias = pos_bias.reshape(1, old_len, old_len, -1).permute(0, 3, 1, 2)
                pos_bias = F.interpolate(pos_bias, size=(new_len, new_len), mode="bicubic", align_corners=False)
                new_swin_state_dict[x] = pos_bias.permute(0, 2, 3, 1).reshape(1, new_len ** 2, -1).squeeze(0)
            else:
                if x in swin_state_dict:
                    new_swin_state_dict[x] = swin_state_dict[x]
        self.model.load_state_dict(new_swin_state_dict)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, num_channels, height, width)
        """
        # print("================")
        # print("*** input size", x.shape)
        x = self.model.patch_embed(x)
        x = self.model.pos_drop(x)
        # print("patch embed", x.shape)
        if self.vision_model_name in ["SwinTransformerV2", "SwinV2WithVit"]:
            for layer in self.model.layers:
                x = layer(x)
        else:
            x = self.model.layers(x)

        # print("swin encoder output", x.shape)
        return x

    def prepare_input(self, img: PIL.Image.Image, random_padding: bool = False) -> torch.Tensor:
        """
        Convert PIL Image to tensor according to specified input_size after following steps below:
            - resize
            - rotate (if align_long_axis is True and image is not aligned longer axis with canvas)
            - pad
        """
        img = img.convert("RGB")
        if self.align_long_axis and (
                (self.input_size[0] > self.input_size[1] and img.width > img.height)
                or (self.input_size[0] < self.input_size[1] and img.width < img.height)
        ):
            img = rotate(img, angle=-90, expand=True)
        img = resize(img, min(self.input_size))
        img.thumbnail((self.input_size[1], self.input_size[0]))
        delta_width = self.input_size[1] - img.width
        delta_height = self.input_size[0] - img.height
        if random_padding:
            pad_width = np.random.randint(low=0, high=delta_width + 1)
            pad_height = np.random.randint(low=0, high=delta_height + 1)
        else:
            pad_width = delta_width // 2
            pad_height = delta_height // 2
        padding = (
            pad_width,
            pad_height,
            delta_width - pad_width,
            delta_height - pad_height,
        )
        return self.to_tensor(ImageOps.expand(img, padding))


class BARTDecoder(nn.Module):
    """
    Donut Decoder based on Multilingual BART
    Set the initial weights and configuration with a pretrained multilingual BART model,
    and modify the detailed configurations as a Donut decoder

    Args:
        decoder_layer:
            Number of layers of BARTDecoder
        max_position_embeddings:
            The maximum sequence length to be trained
        name_or_path:
            Name of a pretrained model name either registered in huggingface.co. or saved in local,
            otherwise, `hyunwoongko/asian-bart-ecjk` will be set (using `transformers`)
    """

    def __init__(
            self, decoder_layer: int, max_position_embeddings: int, name_or_path: Union[str, bytes, os.PathLike] = None,
            use_fast_tokenizer=False, bart_prtrained_path="hyunwoongko/asian-bart-en", special_tokens=None,
            d_model=1024
    ):
        super().__init__()
        self.decoder_layer = decoder_layer
        self.max_position_embeddings = max_position_embeddings

        if use_fast_tokenizer:
            self.tokenizer = AutoTokenizer.from_pretrained(
                name_or_path
            )
        else:
            self.tokenizer = XLMRobertaTokenizer.from_pretrained(
                name_or_path
            )
        self.model = MBartForCausalLM(
            config=MBartConfig(
                d_model=d_model,
                is_decoder=True,
                is_encoder_decoder=False,
                add_cross_attention=True,
                decoder_layers=self.decoder_layer,
                max_position_embeddings=self.max_position_embeddings,
                vocab_size=len(self.tokenizer),
                scale_embedding=True,
                add_final_layer_norm=True,
            )
        )
        self.model.forward = self.forward  # to get cross attentions and utilize `generate` function

        self.model.config.is_encoder_decoder = True  # to get cross-attention
        self.add_special_tokens(["<sep/>"])  # <sep/> is used for representing a list in a JSON
        self.model.model.decoder.embed_tokens.padding_idx = self.tokenizer.pad_token_id
        self.model.prepare_inputs_for_generation = self.prepare_inputs_for_inference
        if special_tokens:
            self.add_special_tokens(special_tokens)
            # self.add_special_tokens(["<s_tableocr>"])

        # weight init with asian-bart
        if not name_or_path:
            # bart_state_dict = MBartForCausalLM.from_pretrained("hyunwoongko/asian-bart-ecjk").state_dict()
            bart_state_dict = MBartForCausalLM.from_pretrained(bart_prtrained_path).state_dict()
            new_bart_state_dict = self.model.state_dict()
            for x in new_bart_state_dict:
                if x.endswith("embed_positions.weight") and self.max_position_embeddings != 1024:
                    new_bart_state_dict[x] = torch.nn.Parameter(
                        self.resize_bart_abs_pos_emb(
                            bart_state_dict[x],
                            self.max_position_embeddings
                            + 2,
                            # https://github.com/huggingface/transformers/blob/v4.11.3/src/transformers/models/mbart/modeling_mbart.py#L118-L119
                        )
                    )
                # elif x.endswith("embed_tokens.weight") or x.endswith("lm_head.weight"):
                # new_bart_state_dict[x] = bart_state_dict[x][: len(self.tokenizer), :]
                elif x.endswith("embed_tokens.weight") or x.endswith("lm_head.weight"):
                    if len(new_bart_state_dict[x]) != len(bart_state_dict[x]):
                        if len(new_bart_state_dict[x]) < len(bart_state_dict[x]):
                            new_bart_state_dict[x] = bart_state_dict[x][: len(new_bart_state_dict[x]), :]
                        else:
                            new_bart_state_dict[x][: len(bart_state_dict[x]), :] = bart_state_dict[x]
                else:
                    new_bart_state_dict[x] = bart_state_dict[x]
            self.model.load_state_dict(new_bart_state_dict)

    def add_special_tokens(self, list_of_tokens: List[str]):
        """
        Add special tokens to tokenizer and resize the token embeddings
        """
        newly_added_num = self.tokenizer.add_special_tokens({"additional_special_tokens": sorted(set(list_of_tokens))})
        if newly_added_num > 0:
            self.model.resize_token_embeddings(len(self.tokenizer))

    def prepare_inputs_for_inference(self, input_ids: torch.Tensor, past=None, use_cache: bool = None, **model_kwargs):
        """
        Args:
            input_ids: (batch_size, sequence_lenth)
        Returns:
            input_ids: (batch_size, sequence_length)
            attention_mask: (batch_size, sequence_length)
            encoder_hidden_states: (batch_size, sequence_length, embedding_dim)
        """
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id).long()
        if past is not None:
            input_ids = input_ids[:, -1:]
        output = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "past_key_values": past,
            "use_cache": use_cache,
            "encoder_hidden_states": model_kwargs["encoder_outputs"].last_hidden_state,
        }
        return output

    def forward(
            self,
            input_ids,
            attention_mask: Optional[torch.Tensor] = None,
            encoder_hidden_states: Optional[torch.Tensor] = None,
            past_key_values: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None,
            use_cache: bool = None,
            output_attentions: Optional[torch.Tensor] = None,
            output_hidden_states: Optional[torch.Tensor] = None,
            return_dict: bool = None,
    ):
        """
        A forward fucntion to get cross attentions and utilize `generate` function

        Source:
        https://github.com/huggingface/transformers/blob/v4.11.3/src/transformers/models/mbart/modeling_mbart.py#L1669-L1810

        Args:
            input_ids: (batch_size, sequence_length)
            attention_mask: (batch_size, sequence_length)
            encoder_hidden_states: (batch_size, sequence_length, hidden_size)

        Returns:
            loss: (1, )
            logits: (batch_size, sequence_length, hidden_dim)
            hidden_states: (batch_size, sequence_length, hidden_size)
            decoder_attentions: (batch_size, num_heads, sequence_length, sequence_length)
            cross_attentions: (batch_size, num_heads, sequence_length, sequence_length)
        """
        output_attentions = output_attentions if output_attentions is not None else self.model.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.model.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.model.config.use_return_dict
        outputs = self.model.model.decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        logits = self.model.lm_head(outputs[0])

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(logits.view(-1, self.model.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return ModelOutput(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            decoder_attentions=outputs.attentions,
            cross_attentions=outputs.cross_attentions,
        )

    @staticmethod
    def resize_bart_abs_pos_emb(weight: torch.Tensor, max_length: int) -> torch.Tensor:
        """
        Resize position embeddings
        Truncate if sequence length of Bart backbone is greater than given max_length,
        else interpolate to max_length
        """
        if weight.shape[0] > max_length:
            weight = weight[:max_length, ...]
        else:
            weight = (
                F.interpolate(
                    weight.permute(1, 0).unsqueeze(0),
                    size=max_length,
                    mode="linear",
                    align_corners=False,
                )
                    .squeeze(0)
                    .permute(1, 0)
            )
        return weight


class DonutConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`DonutModel`]. It is used to
    instantiate a Donut model according to the specified arguments, defining the model architecture

    Args:
        input_size:
            Input image size (canvas size) of Donut.encoder, SwinTransformer in this codebase
        align_long_axis:
            Whether to rotate image if height is greater than width
        window_size:
            Window size of Donut.encoder, SwinTransformer in this codebase
        encoder_layer:
            Depth of each Donut.encoder Encoder layer, SwinTransformer in this codebase
        decoder_layer:
            Number of hidden layers in the Donut.decoder, such as BART
        max_position_embeddings
            Trained max position embeddings in the Donut decoder,
            if not specified, it will have same value with max_length
        max_length:
            Max position embeddings(=maximum sequence length) you want to train
        name_or_path:
            Name of a pretrained model name either registered in huggingface.co. or saved in local
    """

    model_type = "donut"

    def __init__(
            self,
            input_size: List[int] = [2560, 1920],
            align_long_axis: bool = False,
            window_size: int = 10,
            encoder_layer: List[int] = [2, 2, 14, 2],
            decoder_layer: int = 4,
            max_position_embeddings: int = None,
            max_length: int = 1536,
            name_or_path: Union[str, bytes, os.PathLike] = "",
            tokenizer_name_or_path: Union[str, bytes, os.PathLike] = "",
            use_fast_tokenizer=False,
            vision_model_name='SwinTransformer',
            bart_prtrained_path='hyunwoongko/asian-bart-en',
            swin_pretrained_path='swin_base_patch4_window12_384_in22k',
            special_tokens=None,
            swin_model_size='base',
            ape=False,
            swin_name_or_path=None,
            d_model=1024,
            swin_depth_last_block=2,
            swin_num_heads_last_block=8,  # must be "d_model / num_heads_last_block = 0"
            swin_drop_path_rate_last_block=0.,  # or 0.0
            swin_init_values_last_block=None,  # or 1e-5
            ape_last_block=False,
            **kwargs,
    ):
        super().__init__()
        self.input_size = input_size
        self.d_model = d_model
        self.align_long_axis = align_long_axis
        self.window_size = window_size
        self.encoder_layer = encoder_layer
        self.decoder_layer = decoder_layer
        self.ape = ape
        self.max_position_embeddings = max_length if max_position_embeddings is None else max_position_embeddings
        self.max_length = max_length
        self.name_or_path = name_or_path
        self.tokenizer_name_or_path = tokenizer_name_or_path
        self.use_fast_tokenizer = use_fast_tokenizer
        self.vision_model_name = vision_model_name
        self.bart_prtrained_path = bart_prtrained_path
        self.special_tokens = special_tokens
        self.swin_pretrained_path = swin_pretrained_path
        self.swin_model_size = swin_model_size
        self.swin_name_or_path = swin_name_or_path
        self.swin_depth_last_block = swin_depth_last_block
        self.swin_num_heads_last_block = swin_num_heads_last_block
        self.swin_drop_path_rate_last_block = swin_drop_path_rate_last_block
        self.swin_init_values_last_block = swin_init_values_last_block
        self.ape_last_block = ape_last_block


class DonutModel(PreTrainedModel):
    r"""
    Donut: an E2E OCR-free Document Understanding Transformer.
    The encoder maps an input document image into a set of embeddings,
    the decoder predicts a desired token sequence, that can be converted to a structured format,
    given a prompt and the encoder output embeddings
    """
    config_class = DonutConfig
    base_model_prefix = "donut"

    def __init__(self, config: DonutConfig):
        super().__init__(config)
        self.config = config
        self.encoder = SwinEncoder(
            input_size=self.config.input_size,
            align_long_axis=self.config.align_long_axis,
            window_size=self.config.window_size,
            encoder_layer=self.config.encoder_layer,
            name_or_path=self.config.name_or_path,
            vision_model_name=self.config.vision_model_name,
            swin_pretrained_path=self.config.swin_pretrained_path,
            swin_model_size=self.config.swin_model_size,
            ape=self.config.ape,
            swin_name_or_path=self.config.swin_name_or_path,
            depth_last_block=self.config.swin_depth_last_block,
            num_heads_last_block=self.config.swin_num_heads_last_block,  # must be "d_model / num_heads_last_block = 0"
            drop_path_rate_last_block=self.config.swin_drop_path_rate_last_block,  # or 0.0
            init_values_last_block=self.config.swin_init_values_last_block,  # or 1e-5
            ape_last_block=self.config.ape_last_block
        )
        self.decoder = BARTDecoder(
            d_model=self.config.d_model,
            max_position_embeddings=self.config.max_position_embeddings,
            decoder_layer=self.config.decoder_layer,
            use_fast_tokenizer=self.config.use_fast_tokenizer,
            name_or_path=self.config.tokenizer_name_or_path,
            bart_prtrained_path=self.config.bart_prtrained_path,
            special_tokens=self.config.special_tokens
        )

    def forward(self, image_tensors: torch.Tensor, decoder_input_ids: torch.Tensor, decoder_labels: torch.Tensor):
        """
        Calculate a loss given an input image and a desired token sequence,
        the model will be trained in a teacher-forcing manner

        Args:
            image_tensors: (batch_size, num_channels, height, width)
            decoder_input_ids: (batch_size, sequence_length, embedding_dim)
            decode_labels: (batch_size, sequence_length)
        """
        encoder_outputs = self.encoder(image_tensors)
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            encoder_hidden_states=encoder_outputs,
            labels=decoder_labels,
        )
        return decoder_outputs

    def inference(
            self,
            image: PIL.Image = None,
            prompt: str = None,
            image_tensors: Optional[torch.Tensor] = None,
            prompt_tensors: Optional[torch.Tensor] = None,
            return_json: bool = True,
            return_attentions: bool = False,
    ):
        """
        Generate a token sequence in an auto-regressive manner,
        the generated token sequence is convereted into an ordered JSON format

        Args:
            image: input document image (PIL.Image)
            prompt: task prompt (string) to guide Donut Decoder generation
            image_tensors: (1, num_channels, height, width)
                convert prompt to tensor if image_tensor is not fed
            prompt_tensors: (1, sequence_length)
                convert image to tensor if prompt_tensor is not fed
        """
        # prepare backbone inputs (image and prompt)
        if image is None and image_tensors is None:
            raise ValueError("Expected either image or image_tensors")
        if all(v is None for v in {prompt, prompt_tensors}):
            raise ValueError("Expected either prompt or prompt_tensors")

        if image_tensors is None:
            image_tensors = self.encoder.prepare_input(image).unsqueeze(0)

        if self.device.type == "cuda":  # half is not compatible in cpu implementation.
            image_tensors = image_tensors.half()
            image_tensors = image_tensors.to(self.device)
        else:
            image_tensors = image_tensors.to(torch.bfloat16)

        if prompt_tensors is None:
            prompt_tensors = self.decoder.tokenizer(prompt, add_special_tokens=False, return_tensors="pt")["input_ids"]
            if len(image_tensors) > 1:
                prompt_tensors = prompt_tensors.expand(len(image_tensors), -1)

        prompt_tensors = prompt_tensors.to(self.device)

        last_hidden_state = self.encoder(image_tensors)
        if self.device.type != "cuda":
            last_hidden_state = last_hidden_state.to(torch.float32)

        encoder_outputs = ModelOutput(last_hidden_state=last_hidden_state, attentions=None)
        if len(encoder_outputs.last_hidden_state.size()) == 1:
            encoder_outputs.last_hidden_state = encoder_outputs.last_hidden_state.unsqueeze(0)

        # if len(prompt_tensors.size()) == 1:
        #     prompt_tensors = prompt_tensors.unsqueeze(0)
        # get decoder output
        decoder_output = self.decoder.model.generate(
            decoder_input_ids=prompt_tensors,
            encoder_outputs=encoder_outputs,
            max_length=self.config.max_length,
            early_stopping=True,
            pad_token_id=self.decoder.tokenizer.pad_token_id,
            eos_token_id=self.decoder.tokenizer.eos_token_id,
            use_cache=True,
            num_beams=1,
            bad_words_ids=[[self.decoder.tokenizer.unk_token_id]],
            return_dict_in_generate=True,
            output_attentions=return_attentions,
            output_scores=True
        )

        # print(decoder_output.sequences)
        print(len(decoder_output.sequences))
        output = {"predictions": list()}
        for seq in self.decoder.tokenizer.batch_decode(decoder_output.sequences):
            seq = seq.replace(self.decoder.tokenizer.eos_token, "").replace(self.decoder.tokenizer.pad_token, "")
            seq = re.sub(r"<.*?>", "", seq, count=1).strip()  # remove first task start token
            if return_json:
                output["predictions"].append(self.token2json(seq))
            else:
                output["predictions"].append(seq)

        if return_attentions:
            output["attentions"] = {
                "self_attentions": decoder_output.decoder_attentions,
                "cross_attentions": decoder_output.cross_attentions,
            }

        return output

    def inference_one(
            self,
            image: PIL.Image = None
    ):
        prompt = "<s_tableocr>"
        image_tensors = self.encoder.prepare_input(image).unsqueeze(0)

        if self.device.type == "cuda":  # half is not compatible in cpu implementation.
            image_tensors = image_tensors.half()
            image_tensors = image_tensors.to(self.device)
        else:
            image_tensors = image_tensors.to(torch.bfloat16)

        prompt_tensors = self.decoder.tokenizer(prompt, add_special_tokens=False, return_tensors="pt")["input_ids"]

        prompt_tensors = prompt_tensors.to(self.device)

        last_hidden_state = self.encoder(image_tensors)
        if self.device.type != "cuda":
            last_hidden_state = last_hidden_state.to(torch.float32)

        encoder_outputs = ModelOutput(last_hidden_state=last_hidden_state, attentions=None)

        if len(encoder_outputs.last_hidden_state.size()) == 1:
            encoder_outputs.last_hidden_state = encoder_outputs.last_hidden_state.unsqueeze(0)
        if len(prompt_tensors.size()) == 1:
            prompt_tensors = prompt_tensors.unsqueeze(0)

        # get decoder output
        decoder_output = self.decoder.model.generate(
            decoder_input_ids=prompt_tensors,
            encoder_outputs=encoder_outputs,
            max_length=self.config.max_length,
            early_stopping=True,
            pad_token_id=self.decoder.tokenizer.pad_token_id,
            eos_token_id=self.decoder.tokenizer.eos_token_id,
            use_cache=True,
            num_beams=1,
            bad_words_ids=[[self.decoder.tokenizer.unk_token_id]],
            return_dict_in_generate=True,
            output_attentions=False,
        )

        return self.decoder.tokenizer.batch_decode(decoder_output.sequences)[0]
        # for seq in self.decoder.tokenizer.batch_decode(decoder_output.sequences):
        #     seq = seq.replace(self.decoder.tokenizer.eos_token, "").replace(self.decoder.tokenizer.pad_token, "")
        #     seq = re.sub(r"<.*?>", "", seq, count=1).strip()  # remove first task start token
        #     if return_json:
        #         output["predictions"].append(self.token2json(seq))
        #     else:
        #         output["predictions"].append(seq)

    def json2token(self, obj: Any, update_special_tokens_for_json_key: bool = True, sort_json_key: bool = True):
        """
        Convert an ordered JSON object into a token sequence
        """
        if type(obj) == dict:
            if len(obj) == 1 and "text_sequence" in obj:
                text_sequence = obj["text_sequence"]
                if text_sequence.startswith("<table>"):
                    text_sequence = text_sequence[7:-8]
                return text_sequence
            else:
                output = ""
                if sort_json_key:
                    keys = sorted(obj.keys(), reverse=True)
                else:
                    keys = obj.keys()
                for k in keys:
                    if update_special_tokens_for_json_key:
                        self.decoder.add_special_tokens([fr"<s_{k}>", fr"</s_{k}>"])
                    output += (
                            fr"<s_{k}>"
                            + self.json2token(obj[k], update_special_tokens_for_json_key, sort_json_key)
                            + fr"</s_{k}>"
                    )
                return output
        elif type(obj) == list:
            return r"<sep/>".join(
                [self.json2token(item, update_special_tokens_for_json_key, sort_json_key) for item in obj]
            )
        else:
            obj = str(obj)
            if f"<{obj}/>" in self.decoder.tokenizer.all_special_tokens:
                obj = f"<{obj}/>"  # for categorical special tokens
            return obj

    def token2json(self, tokens, is_inner_value=False):
        """
        Convert a (generated) token seuqnce into an ordered JSON format
        """
        output = dict()

        while tokens:
            start_token = re.search(r"<s_(.*?)>", tokens, re.IGNORECASE)
            if start_token is None:
                break
            key = start_token.group(1)
            end_token = re.search(fr"</s_{key}>", tokens, re.IGNORECASE)
            start_token = start_token.group()
            if end_token is None:
                tokens = tokens.replace(start_token, "")
            else:
                end_token = end_token.group()
                start_token_escaped = re.escape(start_token)
                end_token_escaped = re.escape(end_token)
                content = re.search(f"{start_token_escaped}(.*?){end_token_escaped}", tokens, re.IGNORECASE)
                if content is not None:
                    content = content.group(1).strip()
                    if r"<s_" in content and r"</s_" in content:  # non-leaf node
                        value = self.token2json(content, is_inner_value=True)
                        if value:
                            if len(value) == 1:
                                value = value[0]
                            output[key] = value
                    else:  # leaf nodes
                        output[key] = []
                        for leaf in content.split(r"<sep/>"):
                            leaf = leaf.strip()
                            if (
                                    leaf in self.decoder.tokenizer.get_added_vocab()
                                    and leaf[0] == "<"
                                    and leaf[-2:] == "/>"
                            ):
                                leaf = leaf[1:-2]  # for categorical special tokens
                            output[key].append(leaf)
                        if len(output[key]) == 1:
                            output[key] = output[key][0]

                tokens = tokens[tokens.find(end_token) + len(end_token):].strip()
                if tokens[:6] == r"<sep/>":  # non-leaf nodes
                    return [output] + self.token2json(tokens[6:], is_inner_value=True)

        if len(output):
            return [output] if is_inner_value else output
        else:
            return [] if is_inner_value else {"text_sequence": tokens}

    @classmethod
    def from_pretrained(
            cls,
            pretrained_model_name_or_path: Union[str, bytes, os.PathLike],
            *model_args,
            **kwargs,
    ):
        r"""
        Instantiate a pretrained donut model from a pre-trained model configuration

        Args:
            pretrained_model_name_or_path:
                Name of a pretrained model name either registered in huggingface.co. or saved in local,
                e.g., `naver-clova-ix/donut-base`, or `naver-clova-ix/donut-base-finetuned-rvlcdip`
        """
        model = super(DonutModel, cls).from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)

        # truncate or interplolate position embeddings of donut decoder
        max_length = kwargs.get("max_length", model.config.max_position_embeddings)
        if (
                max_length != model.config.max_position_embeddings
        ):  # if max_length of trained model differs max_length you want to train
            model.decoder.model.model.decoder.embed_positions.weight = torch.nn.Parameter(
                model.decoder.resize_bart_abs_pos_emb(
                    model.decoder.model.model.decoder.embed_positions.weight,
                    max_length
                    + 2,
                    # https://github.com/huggingface/transformers/blob/v4.11.3/src/transformers/models/mbart/modeling_mbart.py#L118-L119
                )
            )
            model.config.max_position_embeddings = max_length

        return model


class DonutClipConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`DonutModel`]. It is used to
    instantiate a Donut model according to the specified arguments, defining the model architecture

    Args:
        input_size:
            Input image size (canvas size) of Donut.encoder, SwinTransformer in this codebase
        align_long_axis:
            Whether to rotate image if height is greater than width
        window_size:
            Window size of Donut.encoder, SwinTransformer in this codebase
        encoder_layer:
            Depth of each Donut.encoder Encoder layer, SwinTransformer in this codebase
        bart_encoder_layer:
            Number of hidden layers in the Donut.decoder, such as BART
        max_position_embeddings
            Trained max position embeddings in the Donut decoder,
            if not specified, it will have same value with max_length
        max_length:
            Max position embeddings(=maximum sequence length) you want to train
        name_or_path:
            Name of a pretrained model name either registered in huggingface.co. or saved in local
    """

    model_type = "donut"

    def __init__(
            self,
            input_size: List[int] = [2560, 1920],
            align_long_axis: bool = False,
            window_size: int = 10,
            swin_encoder_layer: List[int] = [2, 2, 14, 2],
            bart_encoder_layer: int = 4,
            max_position_embeddings: int = None,
            max_length: int = 1536,
            name_or_path: Union[str, bytes, os.PathLike] = "",
            tokenizer_name_or_path: Union[str, bytes, os.PathLike] = "",
            use_fast_tokenizer=False,
            vision_model_name='SwinTransformer',
            bart_pretrained_path='facebook/mbart-large-cc25',
            swin_pretrained_path='swin_base_patch4_window12_384_in22k',
            special_tokens=None,
            swin_model_size='base',
            d_model=1024,
            projection_dim=512,
            ape=False,
            swin_name_or_path=None,
            **kwargs,
    ):
        super().__init__()
        self.input_size = input_size
        self.align_long_axis = align_long_axis
        self.window_size = window_size
        self.swin_encoder_layer = swin_encoder_layer
        self.bart_encoder_layer = bart_encoder_layer
        self.max_position_embeddings = max_length if max_position_embeddings is None else max_position_embeddings
        self.max_length = max_length
        self.ape = ape
        self.name_or_path = name_or_path
        self.tokenizer_name_or_path = tokenizer_name_or_path
        self.use_fast_tokenizer = use_fast_tokenizer
        self.vision_model_name = vision_model_name
        self.bart_pretrained_path = bart_pretrained_path
        self.special_tokens = special_tokens
        self.swin_pretrained_path = swin_pretrained_path
        self.swin_model_size = swin_model_size
        self.d_model = d_model
        self.projection_dim = projection_dim
        self.swin_name_or_path = swin_name_or_path


def contrastive_loss(logits: torch.Tensor) -> torch.Tensor:
    return nn.functional.cross_entropy(logits, torch.arange(len(logits), device=logits.device))


def clip_loss(similarity: torch.Tensor) -> torch.Tensor:
    caption_loss = contrastive_loss(similarity)
    image_loss = contrastive_loss(similarity.T)
    return (caption_loss + image_loss) / 2.0


class DonutClipModel(PreTrainedModel):
    r"""
    Donut: an E2E OCR-free Document Understanding Transformer.
    The encoder maps an input document image into a set of embeddings,
    the decoder predicts a desired token sequence, that can be converted to a structured format,
    given a prompt and the encoder output embeddings
    """
    config_class = DonutClipConfig
    base_model_prefix = "donutclip"

    def __init__(self, config: DonutClipConfig):
        super().__init__(config)
        self.config = config
        self.encoder = SwinEncoder(
            input_size=self.config.input_size,
            align_long_axis=self.config.align_long_axis,
            window_size=self.config.window_size,
            encoder_layer=self.config.swin_encoder_layer,
            name_or_path=self.config.name_or_path,
            vision_model_name=self.config.vision_model_name,
            swin_pretrained_path=self.config.swin_pretrained_path,
            swin_model_size=self.config.swin_model_size,
            ape=self.config.ape,
            swin_name_or_path=self.config.swin_name_or_path
        )
        self.visual_projection = nn.Linear(self.config.d_model, self.config.projection_dim, bias=False)
        self.post_layernorm = nn.LayerNorm(self.config.d_model)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.text_encoder = BARTEncoder(
            max_position_embeddings=self.config.max_position_embeddings,
            encoder_layer=self.config.bart_encoder_layer,
            use_fast_tokenizer=self.config.use_fast_tokenizer,
            name_or_path=self.config.tokenizer_name_or_path,
            bart_pretrained_path=self.config.bart_pretrained_path,
            special_tokens=self.config.special_tokens,
        )

        self.text_projection = nn.Linear(self.config.d_model, self.config.projection_dim, bias=False)

    def forward(self, image_tensors: torch.Tensor, text_tensors: torch.Tensor,
                return_loss=False,
                return_dict: Optional[bool] = None):
        """
        Calculate a loss given an input image and a desired token sequence,
        the model will be trained in a teacher-forcing manner

        Args:
            image_tensors: (batch_size, num_channels, height, width)
            decoder_input_ids: (batch_size, sequence_length, embedding_dim)
            decode_labels: (batch_size, sequence_length)
        """

        vision_outputs = self.encoder(image_tensors)
        image_features = vision_outputs[:, 0, :]
        image_features = self.post_layernorm(image_features)
        image_features = self.visual_projection(image_features)
        # [1, 1200, 1024]
        text_outputs = self.text_encoder(text_tensors, return_dict=True)
        text_features = text_outputs[1]
        text_features = self.text_projection(text_features)

        # [batch, output_embed_dim]

        # normalized features
        image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_text = torch.matmul(text_features, image_features.t()) * logit_scale
        logits_per_image = logits_per_text.T

        loss = None
        if return_loss:
            loss = clip_loss(logits_per_text)

        if not return_dict:
            output = (logits_per_image, logits_per_text, text_features, image_features, text_outputs, vision_outputs)
            return ((loss,) + output) if loss is not None else output

        return ModelOutput(
            loss=loss,
            logits_per_image=logits_per_image,
            logits_per_text=logits_per_text,
            text_embeds=text_features,
            image_embeds=image_features,
            text_model_output=text_outputs,
            vision_model_output=vision_outputs,
        )

    def json2token(self, obj: Any, update_special_tokens_for_json_key: bool = True, sort_json_key: bool = True):
        """
        Convert an ordered JSON object into a token sequence
        """
        if type(obj) == dict:
            if len(obj) == 1 and "text_sequence" in obj:
                text_sequence = obj["text_sequence"]
                if text_sequence.startswith("<table>"):
                    text_sequence = text_sequence[7:-8]
                return text_sequence
            else:
                output = ""
                if sort_json_key:
                    keys = sorted(obj.keys(), reverse=True)
                else:
                    keys = obj.keys()
                for k in keys:
                    if update_special_tokens_for_json_key:
                        self.decoder.add_special_tokens([fr"<s_{k}>", fr"</s_{k}>"])
                    output += (
                            fr"<s_{k}>"
                            + self.json2token(obj[k], update_special_tokens_for_json_key, sort_json_key)
                            + fr"</s_{k}>"
                    )
                return output
        elif type(obj) == list:
            return r"<sep/>".join(
                [self.json2token(item, update_special_tokens_for_json_key, sort_json_key) for item in obj]
            )
        else:
            obj = str(obj)
            if f"<{obj}/>" in self.decoder.tokenizer.all_special_tokens:
                obj = f"<{obj}/>"  # for categorical special tokens
            return obj

    def token2json(self, tokens, is_inner_value=False):
        """
        Convert a (generated) token seuqnce into an ordered JSON format
        """
        output = dict()

        while tokens:
            start_token = re.search(r"<s_(.*?)>", tokens, re.IGNORECASE)
            if start_token is None:
                break
            key = start_token.group(1)
            end_token = re.search(fr"</s_{key}>", tokens, re.IGNORECASE)
            start_token = start_token.group()
            if end_token is None:
                tokens = tokens.replace(start_token, "")
            else:
                end_token = end_token.group()
                start_token_escaped = re.escape(start_token)
                end_token_escaped = re.escape(end_token)
                content = re.search(f"{start_token_escaped}(.*?){end_token_escaped}", tokens, re.IGNORECASE)
                if content is not None:
                    content = content.group(1).strip()
                    if r"<s_" in content and r"</s_" in content:  # non-leaf node
                        value = self.token2json(content, is_inner_value=True)
                        if value:
                            if len(value) == 1:
                                value = value[0]
                            output[key] = value
                    else:  # leaf nodes
                        output[key] = []
                        for leaf in content.split(r"<sep/>"):
                            leaf = leaf.strip()
                            if (
                                    leaf in self.decoder.tokenizer.get_added_vocab()
                                    and leaf[0] == "<"
                                    and leaf[-2:] == "/>"
                            ):
                                leaf = leaf[1:-2]  # for categorical special tokens
                            output[key].append(leaf)
                        if len(output[key]) == 1:
                            output[key] = output[key][0]

                tokens = tokens[tokens.find(end_token) + len(end_token):].strip()
                if tokens[:6] == r"<sep/>":  # non-leaf nodes
                    return [output] + self.token2json(tokens[6:], is_inner_value=True)

        if len(output):
            return [output] if is_inner_value else output
        else:
            return [] if is_inner_value else {"text_sequence": tokens}

    @classmethod
    def from_pretrained(
            cls,
            pretrained_model_name_or_path: Union[str, bytes, os.PathLike],
            *model_args,
            **kwargs,
    ):
        r"""
        Instantiate a pretrained donut model from a pre-trained model configuration

        Args:
            pretrained_model_name_or_path:
                Name of a pretrained model name either registered in huggingface.co. or saved in local,
                e.g., `naver-clova-ix/donut-base`, or `naver-clova-ix/donut-base-finetuned-rvlcdip`
        """
        model = super(DonutClipModel, cls).from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)

        # truncate or interplolate position embeddings of donut decoder
        max_length = kwargs.get("max_length", model.config.max_position_embeddings)
        if (
                max_length != model.config.max_position_embeddings
        ):  # if max_length of trained model differs max_length you want to train
            model.decoder.model.model.decoder.embed_positions.weight = torch.nn.Parameter(
                model.decoder.resize_bart_abs_pos_emb(
                    model.decoder.model.model.decoder.embed_positions.weight,
                    max_length
                    + 2,
                    # https://github.com/huggingface/transformers/blob/v4.11.3/src/transformers/models/mbart/modeling_mbart.py#L118-L119
                )
            )
            model.config.max_position_embeddings = max_length

        return model


class BARTEncoder(nn.Module):
    """
    Donut Decoder based on Multilingual BART
    Set the initial weights and configuration with a pretrained multilingual BART model,
    and modify the detailed configurations as a Donut decoder

    Args:
        encoder_layer:
            Number of layers of BARTDecoder
        max_position_embeddings:
            The maximum sequence length to be trained
        name_or_path:
            Name of a pretrained model name either registered in huggingface.co. or saved in local,
            otherwise, `hyunwoongko/asian-bart-ecjk` will be set (using `transformers`)
    """

    def __init__(
            self, encoder_layer: int, max_position_embeddings: int, name_or_path: Union[str, bytes, os.PathLike] = None,
            use_fast_tokenizer=False, bart_pretrained_path="hyunwoongko/asian-bart-en", special_tokens=None,
            d_model=1024
    ):
        super().__init__()
        self.encoder_layer = encoder_layer
        self.max_position_embeddings = max_position_embeddings

        if use_fast_tokenizer:
            self.tokenizer = AutoTokenizer.from_pretrained(
                name_or_path
            )
        else:
            self.tokenizer = XLMRobertaTokenizer.from_pretrained(
                name_or_path
            )

        self.tokenizer.add_special_tokens(
            {"additional_special_tokens": ["<sep/>"]})  # <sep/> is used for representing a list in a JSON
        if special_tokens:
            self.tokenizer.add_special_tokens({"additional_special_tokens": sorted(set(special_tokens))})
            # self.add_special_tokens(["<s_tableocr>"])

        self.model = MBartEncoder(MBartConfig(
            encoder_layers=self.encoder_layer,
            max_position_embeddings=self.max_position_embeddings,
            vocab_size=len(self.tokenizer),
            scale_embedding=True,
            add_final_layer_norm=True,
            pad_token_id=self.tokenizer.pad_token_id
        ))
        self.final_layer_norm = nn.LayerNorm(d_model)

        # weight init with asian-bart
        if not name_or_path:
            bart_state_dict = MBartModel.from_pretrained(bart_pretrained_path).state_dict()["encoder"]
            new_bart_state_dict = self.model.state_dict()
            for x in new_bart_state_dict:
                if x.endswith("embed_positions.weight") and self.max_position_embeddings != 1024:
                    new_bart_state_dict[x] = torch.nn.Parameter(
                        self.resize_bart_abs_pos_emb(
                            bart_state_dict[x],
                            self.max_position_embeddings
                            + 2,
                            # https://github.com/huggingface/transformers/blob/v4.11.3/src/transformers/models/mbart/modeling_mbart.py#L118-L119
                        )
                    )
                # elif x.endswith("embed_tokens.weight") or x.endswith("lm_head.weight"):
                # new_bart_state_dict[x] = bart_state_dict[x][: len(self.tokenizer), :]
                elif x.endswith("embed_tokens.weight") or x.endswith("lm_head.weight"):
                    if len(new_bart_state_dict[x]) != len(bart_state_dict[x]):
                        if len(new_bart_state_dict[x]) < len(bart_state_dict[x]):
                            new_bart_state_dict[x] = bart_state_dict[x][: len(new_bart_state_dict[x]), :]
                        else:
                            new_bart_state_dict[x][: len(bart_state_dict[x]), :] = bart_state_dict[x]
                else:
                    new_bart_state_dict[x] = bart_state_dict[x]
            self.model.load_state_dict(new_bart_state_dict)

    def forward(
            self,
            input_ids,
            attention_mask: Optional[torch.Tensor] = None,
            output_attentions: Optional[torch.Tensor] = None,
            output_hidden_states: Optional[torch.Tensor] = None,
            return_dict: bool = None,
    ):
        """
        A forward fucntion to get cross attentions and utilize `generate` function

        Source:
        https://github.com/huggingface/transformers/blob/v4.11.3/src/transformers/models/mbart/modeling_mbart.py#L1669-L1810

        Args:
            input_ids: (batch_size, sequence_length)
            attention_mask: (batch_size, sequence_length)
            encoder_hidden_states: (batch_size, sequence_length, hidden_size)

        Returns:
            loss: (1, )
            logits: (batch_size, sequence_length, hidden_dim)
            hidden_states: (batch_size, sequence_length, hidden_size)
            decoder_attentions: (batch_size, num_heads, sequence_length, sequence_length)
            cross_attentions: (batch_size, num_heads, sequence_length, sequence_length)
        """
        output_attentions = output_attentions if output_attentions is not None else self.model.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.model.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.model.config.use_return_dict
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # last_hidden_state = outputs[0]
        last_hidden_state = outputs['last_hidden_state']
        last_hidden_state = self.final_layer_norm(last_hidden_state)

        pooled_output = last_hidden_state[torch.arange(last_hidden_state.shape[0]), input_ids.argmax(dim=-1)]

        if not return_dict:
            return (last_hidden_state, pooled_output) + outputs[1:]
        return ModelOutput(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=last_hidden_state,
            # attentions=outputs['attentions'],
        )

    def add_special_tokens(self, list_of_tokens: List[str]):
        """
        Add special tokens to tokenizer and resize the token embeddings
        """
        newly_added_num = self.tokenizer.add_special_tokens({"additional_special_tokens": sorted(set(list_of_tokens))})
        if newly_added_num > 0:
            self.model.resize_token_embeddings(len(self.tokenizer))

    @staticmethod
    def resize_bart_abs_pos_emb(weight: torch.Tensor, max_length: int) -> torch.Tensor:
        """
        Resize position embeddings
        Truncate if sequence length of Bart backbone is greater than given max_length,
        else interpolate to max_length
        """
        if weight.shape[0] > max_length:
            weight = weight[:max_length, ...]
        else:
            weight = (
                F.interpolate(
                    weight.permute(1, 0).unsqueeze(0),
                    size=max_length,
                    mode="linear",
                    align_corners=False,
                )
                    .squeeze(0)
                    .permute(1, 0)
            )
        return weight
