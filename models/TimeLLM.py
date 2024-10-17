import torch
import json
import torch.nn as nn
from models.backbone.llama import Llama
from models.layers.StandardNorm import Normalize
from models.layers.mapping_layer import MappingLayer
from models.layers.reprogramming_layer import ReprogrammingLayer
from models.layers.output_projection import FlattenHead
from models.embeddings.patch_embedding import PatchEmbedding
from utils.Prompt import Prompt


class TimeLLM(nn.Module):
    def __init__(self, configs):
        super(TimeLLM, self).__init__()
        self.configs = configs
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
        self.output_projection = FlattenHead(configs)

    def __run_inference_on_llama(self, llama_enc_out, n_vars):
        dec_out = self.llama(llama_enc_out)
        dec_out = dec_out[:, :, :self.d_ff]

        dec_out = torch.reshape(
            dec_out, (-1, n_vars, dec_out.shape[-2], dec_out.shape[-1]))
        dec_out = dec_out.permute(0, 1, 3, 2).contiguous()

        return dec_out

    def _get_source_embeddings(self):
        word_embeddings = self.llama.get_model().get_input_embeddings().weight
        source_embeddings = self.mapping_layer(word_embeddings.permute(1, 0)).permute(1, 0)
        print(f"source embeddings shape {source_embeddings.shape}")
        return source_embeddings

    def _get_prompt_embeddings(self, x_enc):
        # Extract trends and information from input and Generate prompt and get embeddings
        prmpt = Prompt()
        prompt = prmpt.generate_prompt(x_enc, self.configs["description"], self.pred_len, self.seq_len)
        prompt_embeddings = prmpt.tokenize_prompt_and_get_prompt_embeddings(prompt,
                                                                            self.llama.get_tokenizer(),
                                                                            self.llama.get_model(),
                                                                            x_enc.device)
        return prompt_embeddings

    def _get_reprogramming_output(self, x):
        source_embeddings = self._get_source_embeddings()
        enc_out = self.reprogramming_layer(x, source_embeddings, source_embeddings)
        return enc_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # normalize the input
        print(f"x_enc shape {x_enc.shape}")
        x_enc = self.normalize_layers(x_enc, 'norm')

        prompt_embeddings = self._get_prompt_embeddings(x_enc)
        print(f"prompt embedding shape {prompt_embeddings.shape}")

        # Get patch embeddings
        patch_embedding, n_vars = self.patch_embedding(x_enc.to(torch.float32))
        print(f"patch embeddings shape {patch_embedding.shape}")

        # Target embeddings, Source embeddings, and Value embeddings
        # Get source embeddings
        reprogrammed_embeddings = self._get_reprogramming_output(patch_embedding)
        print(f"enc_out shape {reprogrammed_embeddings.shape}")

        llama_enc_out = torch.cat([prompt_embeddings, reprogrammed_embeddings], dim=1)

        # Run inference on llama
        dec_out = self.__run_inference_on_llama(llama_enc_out, n_vars)
        print(f"dec_out shape {dec_out.shape}")

        dec_out = self.output_projection(dec_out)
        print(f"output after projection shape {dec_out.shape}")

        dec_out = self.normalize_layers(dec_out, 'denorm')
        print(f"dec_out shape {dec_out.shape}")

        return dec_out[:, -self.configs["pred_len"]:, :]
