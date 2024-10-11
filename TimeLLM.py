import torch
import json
import torch.nn as nn
from models.llama import Llama
from layers.StandardNorm import Normalize
from layers.mapping_layer import MappingLayer
from layers.reprogramming_layer import ReprogrammingLayer
from layers.output_projection import FlattenHead
from utils.Prompt import Prompt
from embeddings.patch_embedding import PatchEmbedding

configs = json.load(open("configs.json"))


class TimeLLM(nn.Module):
    def __init__(self):
        super(TimeLLM, self).__init__()
        self.pred_len = configs['pred_len']
        self.seq_len = configs['seq_len']
        self.d_ff = 128
        self.top_k = 5
        self.d_llm = 4096
        self.patch_len = 16
        self.stride = 8

        self.normalize_layers = Normalize(7, affine=False)

        # Embedding layers
        self.patch_embedding = PatchEmbedding(32, self.patch_len, self.stride, 0.1)

        self.llama = Llama()
        self.mapping_layer = MappingLayer(self.llama.get_vocab_size())
        self.reprogramming_layer = ReprogrammingLayer(32, configs["n_heads"], self.d_ff, self.d_llm)
        self.output_projection = FlattenHead()

    def __run_inference_on_llama(self, llama_enc_out, n_vars):
        dec_out = self.llama(llama_enc_out)
        dec_out = dec_out[:, :, :self.d_ff]

        dec_out = torch.reshape(
            dec_out, (-1, n_vars, dec_out.shape[-2], dec_out.shape[-1]))
        dec_out = dec_out.permute(0, 1, 3, 2).contiguous()

        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # normalize the input
        x_enc = self.normalize_layers(x_enc, 'norm')
        print(f"x_enc shape {x_enc.shape}")

        # Extract trends and information from input and Generate prompt and get embeddings
        prmpt = Prompt()
        prompt = prmpt.generate_prompt(x_enc, configs["description"], self.pred_len, self.seq_len)
        prompt_embeddings = prmpt.tokenize_prompt_and_get_prompt_embeddings(prompt,
                                                                            self.llama.get_tokenizer(),
                                                                            self.llama.get_model(),
                                                                            x_enc.device)

        print(f"prompt embedding shape {prompt_embeddings.shape}")
        print(f"prompt embedding {prompt_embeddings.shape}")

        # Get source embeddings
        word_embeddings = self.llama.get_model().get_input_embeddings().weight
        source_embeddings = self.mapping_layer(word_embeddings.permute(1, 0)).permute(1, 0)
        print(f"source embeddings shape {source_embeddings.shape}")

        # Get patch embeddings
        x_enc = x_enc.permute(0, 2, 1).contiguous()
        enc_out, n_vars = self.patch_embedding(x_enc.to(torch.float32))

        print(f"patch embeddings shape {enc_out.shape}")

        # Target embeddings, Source embeddings, and Value embeddings
        enc_out = self.reprogramming_layer(enc_out, source_embeddings, source_embeddings)

        llama_enc_out = torch.cat([prompt_embeddings, enc_out], dim=1)

        # Run inference on llama
        dec_out = self.__run_inference_on_llama(llama_enc_out, n_vars)
        print(f"dec_out shape {dec_out.shape}")

        dec_out = self.output_projection(dec_out)
        dec_out = dec_out.permute(0, 2, 1).contiguous()

        dec_out = self.normalize_layers(dec_out, 'denorm')
        print(f"dec_out shape {dec_out.shape}")

        return dec_out[:, -configs["pred_len"]:, :]
