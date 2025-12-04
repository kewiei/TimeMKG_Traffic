from math import sqrt
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from layers.Embed import PatchEmbedding, DataEmbedding_inverted
import transformers
import torch.nn.functional as F
from layers.Transformer_EncDec import Encoder, EncoderLayer, Decoder, DecoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.StandardNorm import Normalize
import os
import warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore", message="The current process just got forked")
transformers.logging.set_verbosity_error()

class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x


class Model(nn.Module):

    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.d_ff = configs.d_ff
        self.d_llm = configs.llm_dim
        self.max_len = configs.max_len
        self.data_path = os.path.join(configs.root_path, configs.data_path)
        self.configs = configs
        
        with open(self.data_path, 'r') as f:
            first_line = f.readline().strip()
        self.variables = ['arrived','density','departed','entered','laneChangedFrom','laneChangedTo','laneDensity','left','occupancy','overlapTraveltime','sampledSeconds','speed','speedRelative','teleported','timeLoss','traveltime','waitingTime']
        self.prompt_bank = configs.prompt_path
        
        with open(self.prompt_bank, 'r') as f:
            lines = f.readlines()
        self.prompts = {}
        for line in lines:
            line = line.strip()
            if line:
                variable, prompt = line.split(':', 1)
                self.prompts[variable.strip()] = prompt.strip()
        
        # 预计算prompt_embeddings的保存路径
        self.prompt_embeddings_path = os.path.join(configs.checkpoints, f"prompt_embeddings_bs{configs.batch_size}.pt")
        
        # 只加载tokenizer（始终需要）
        model_path = "/mnt/petrelfs/sunyifei/qwen3"
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 检查是否已有保存的prompt_embeddings
        self.has_saved_embeddings = os.path.exists(self.prompt_embeddings_path)
        if self.has_saved_embeddings:
            print(f"Found saved prompt_embeddings at {self.prompt_embeddings_path}")
            print("Will not load Qwen model unless necessary")
        else:
            print("No saved prompt_embeddings found. Will load Qwen model when needed")

        # Embedding
        self.enc_embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
        # Time Series Encoder
        self.TS_encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=False), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        # Prompt Encoder
        self.Prompt_encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=False), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        self.sentenceAggregator = nn.Sequential(
            nn.Linear(self.max_len, self.max_len // 2),
            nn.ReLU(),
            nn.Linear(self.max_len // 2, 1)
        )
        self.LLMtodim = nn.Linear(self.d_llm, configs.d_model)

        # Cross Attention
        self.cross_attention = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                        output_attention=False), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                ) for l in range(configs.d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model),
        )
        # Decoder
        self.decoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=False), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        # Decoder
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.projection = nn.Linear(configs.d_model, configs.pred_len, bias=True)
        if self.task_name == 'classification':
            self.act = F.gelu
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(configs.d_model * configs.enc_in, configs.num_class)
        
        # 缓存
        self.prompt_embeddings_dict = {}
        self.LLMmodel = None  # 延迟加载Qwen模型

    def _load_llm_model(self):
        """延迟加载Qwen模型（仅在需要时）"""
        if self.LLMmodel is not None:
            return self.LLMmodel
        
        print("Loading Qwen model...")
        model_path = "/mnt/petrelfs/sunyifei/qwen3"
        self.LLMmodel = AutoModelForCausalLM.from_pretrained(
            model_path, 
            trust_remote_code=True, 
            device_map="auto" if torch.cuda.is_available() else "cpu",
            low_cpu_mem_usage=True
        )
        self.LLMmodel.eval()
        print("Qwen model loaded successfully")
        return self.LLMmodel

    def _get_prompt_embeddings(self, B, N, device):
        """获取或预计算prompt_embeddings"""
        key = f"{B}_{N}"
        
        # 如果已缓存，直接返回
        if key in self.prompt_embeddings_dict:
            return self.prompt_embeddings_dict[key].to(device)
        
        # 检查是否有预计算的文件
        if os.path.exists(self.prompt_embeddings_path):
            print(f"Loading precomputed prompt_embeddings from {self.prompt_embeddings_path}")
            try:
                # 先加载到CPU再移到目标设备，避免内存问题
                prompt_embeddings = torch.load(self.prompt_embeddings_path, map_location="cpu")
                
                # 检查形状是否匹配
                expected_shape = [B * N, -1]
                actual_shape = list(prompt_embeddings.shape)
                
                if actual_shape[0] == B * N:
                    self.prompt_embeddings_dict[key] = prompt_embeddings
                    return prompt_embeddings.to(device)
                else:
                    print(f"Stored embeddings shape {actual_shape} doesn't match current {[B*N, -1]}, regenerating...")
            except Exception as e:
                print(f"Error loading saved embeddings: {e}, regenerating...")
        
        # 需要重新计算，必须加载LLM模型
        print(f"Generating prompt_embeddings for batch size {B}...")
        LLMmodel = self._load_llm_model()
        
        # 构建prompts列表
        prompts = []
        for b in range(B):
            for n in range(N):
                var_name = self.variables[n % len(self.variables)]  # 处理N不匹配的情况
                response = self.prompts.get(var_name, self.prompts[self.variables[0]])
                prompts.append(response)
        
        with torch.no_grad():
            inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=1024)
            inputs = inputs.to(next(LLMmodel.parameters()).device)
            prompt_embeddings = LLMmodel.model.embed_tokens(inputs.input_ids)
        
        # 保存到文件
        os.makedirs(os.path.dirname(self.prompt_embeddings_path), exist_ok=True)
        torch.save(prompt_embeddings.cpu(), self.prompt_embeddings_path)
        print(f"Prompt_embeddings saved to {self.prompt_embeddings_path}")
        self.has_saved_embeddings = True
        
        # 缓存并返回
        self.prompt_embeddings_dict[key] = prompt_embeddings.cpu()
        return prompt_embeddings.to(device)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        if self.task_name == 'classification':
            dec_out = self.classification(x_enc, x_mark_enc)
            return dec_out  # [B, N]
        return None

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev
        B, T, N = x_enc.size()
        device = x_enc.device

        # 获取预计算的prompt_embeddings
        prompt_embeddings = self._get_prompt_embeddings(B, N, device)
        
        self.d_llm = prompt_embeddings.size(-1)
        prompt_embeddings = prompt_embeddings.view(B, N, -1, self.d_llm)
        prompt_embeddings = prompt_embeddings[:, :, :self.max_len, :]
        prompt_embeddings = prompt_embeddings.permute(0, 1, 3, 2).contiguous()

        # 这些层需要参与训练
        prompt_out = self.sentenceAggregator(prompt_embeddings)
        prompt_out = prompt_out.squeeze(-1)
        prompt_out = self.LLMtodim(prompt_out)
        # Prompt Encoder
        prompt_out, attns = self.Prompt_encoder(prompt_out, attn_mask=None)

        # Time Series encoder
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.TS_encoder(enc_out, attn_mask=None)

        # Cross Attention
        dec_out = self.cross_attention(enc_out, prompt_out, x_mask=None, cross_mask=None)

        # decoder
        dec_out, attn = self.decoder(dec_out, attn_mask=None)
        dec_out = self.projection(dec_out).permute(0, 2, 1)[:, :, :N]

        # De-Normalization
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        return dec_out

    def classification(self, x_enc, x_mark_enc):
        B, T, N = x_enc.size()
        device = x_enc.device
        
        # 获取预计算的prompt_embeddings
        prompt_embeddings = self._get_prompt_embeddings(B, N, device)

        self.d_llm = prompt_embeddings.size(-1)
        prompt_embeddings = prompt_embeddings.view(B, N, -1, self.d_llm)
        prompt_embeddings = prompt_embeddings[:, :, :self.max_len, :]
        prompt_embeddings = prompt_embeddings.permute(0, 1, 3, 2).contiguous()

        # 这些层需要参与训练
        prompt_out = self.sentenceAggregator(prompt_embeddings)
        prompt_out = prompt_out.squeeze(-1)
        prompt_out = self.LLMtodim(prompt_out)
        # Prompt Encoder
        prompt_out, attns = self.Prompt_encoder(prompt_out, attn_mask=None)

        # Time Series encoder
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.TS_encoder(enc_out, attn_mask=None)

        # Cross-Modality Attention
        dec_out = self.cross_attention(enc_out, prompt_out, x_mask=None, cross_mask=None)

        # decoder
        dec_out, attn = self.decoder(dec_out, attn_mask=None)

        # Output
        output = self.act(dec_out)
        output = self.dropout(output)
        output = output.reshape(output.shape[0], -1)
        output = self.projection(output)
        return output
    
    def __del__(self):
        """清理资源"""
        if hasattr(self, 'LLMmodel') and self.LLMmodel is not None:
            del self.LLMmodel
            self.LLMmodel = None
        torch.cuda.empty_cache()