from transformers import LlamaConfig, LlamaModel, LlamaTokenizer

class Llama:
    def __init__(self):
        self.llama_config = LlamaConfig.from_pretrained('huggyllama/llama-7b')
        self.llama_config.num_hidden_layers = 6
        self.llama_config.output_attentions = True
        self.llama_config.output_hidden_states = True
        self.description = 'The Electricity Transformer Temperature (ETT) is a crucial indicator in the electric power long-term deployment.'
        try:
            self.llm_model = LlamaModel.from_pretrained(
                # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/",
                'huggyllama/llama-7b',
                trust_remote_code=True,
                local_files_only=True,
                config=self.llama_config,
                # load_in_4bit=True
            )
        except EnvironmentError:  # downloads models from HF is not already done
            print("Local models files not found. Attempting to download...")
            self.llm_model = LlamaModel.from_pretrained(
                # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/",
                'huggyllama/llama-7b',
                trust_remote_code=True,
                local_files_only=False,
                config=self.llama_config,
                # load_in_4bit=True
            )
        try:
            self.tokenizer = LlamaTokenizer.from_pretrained(
                # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/tokenizer.models",
                'huggyllama/llama-7b',
                trust_remote_code=True,
                local_files_only=True
            )
        except EnvironmentError:  # downloads the tokenizer from HF if not already done
            print("Local tokenizer files not found. Atempting to download them..")
            self.tokenizer = LlamaTokenizer.from_pretrained(
                # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/tokenizer.models",
                'huggyllama/llama-7b',
                trust_remote_code=True,
                local_files_only=False
            )

        pad_token = '[PAD]'
        self.tokenizer.add_special_tokens({'pad_token': pad_token})
        self.tokenizer.pad_token = pad_token
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def get_config(self):
        return self.llama_config

    def get_input_embeddings(self):
        return self.llm_model.get_input_embeddings().weight

    def get_tokenizer(self):
        return self.tokenizer

    def get_model(self):
        return self.llm_model