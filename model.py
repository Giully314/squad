import torch 
from torch import nn
import layers
import os
import config
import utils
import inspect


class BiDAFModel(nn.Module):
    def __init__(self, proj_conf: config.ProjectConfig):
        """ 
        TODO: describe what are the params.
        """
        super().__init__() 
        self.config = proj_conf.model
        cfg = self.config
        char_cnn_kernel_width = cfg.char_cnn_kernel_width
        char_cnn_channels = cfg.char_cnn_channels
        hidden_dim = cfg.hidden_dim
        contextual_layers = cfg.contextual_layers
        contextual_dropout = cfg.contextual_dropout
        attention_dropout = cfg.attention_dropout
        modeling_layers = cfg.modeling_layers
        modeling_dropout = cfg.modeling_dropout

        word_emb, word_to_idx, char_emb, char_to_idx = self._load_embeddings(proj_conf)
        
        self.word_emb_layer = layers.WordEmbeddingLayer(word_emb, word_to_idx, char_emb, char_to_idx, 
                                                        char_cnn_kernel_width, char_cnn_channels)
        
        embedding_dim = self.word_emb_layer.word_emb.embedding_dim
        self.contexual_layer = layers.LSTMEncoder(embedding_dim + char_cnn_channels, hidden_dim, contextual_layers, 
                                                    contextual_dropout)
        self.attention_flow_layer = layers.AttentionFlowLayer(hidden_dim, attention_dropout)
        self.modeling_layer = layers.LSTMEncoder(hidden_dim * 8, hidden_dim, modeling_layers, modeling_dropout)
        self.output_layer = layers.QAOutputLayer(hidden_dim)


    def forward(self, context: torch.Tensor, context_char: torch.Tensor, 
                query: torch.Tensor, query_char: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """ 
        context: batch_size, max_context_len
        context: batch_size, max_query_len
        """

        # Compute the real length for each context and query
        c_mask = torch.zeros_like(context) != context
        c_len = c_mask.sum(-1)
        q_mask = torch.zeros_like(query) != query
        q_len = q_mask.sum(-1)


        context_emb_out = self.word_emb_layer(context, context_char)
        query_emb_out = self.word_emb_layer(query, query_char)

        context_encoded = self.contexual_layer(context_emb_out, c_len)
        query_encoded = self.contexual_layer(query_emb_out, q_len)

        attention_out = self.attention_flow_layer(context_encoded, query_encoded)

        modeling_out = self.modeling_layer(attention_out, c_len)

        out = self.output_layer(attention_out, modeling_out, c_len)

        return out
    
    
    def _load_embeddings(self, proj_conf: config.ProjectConfig):
        data_dir = proj_conf.paths.data_dir
        dataset = proj_conf.dataset
        word_emb_file = os.path.join(data_dir, dataset.word_emb_file)
        word_to_idx_file = os.path.join(data_dir, dataset.word_to_idx_file)
        char_emb_file = os.path.join(data_dir, dataset.char_emb_file)
        char_to_idx_file = os.path.join(data_dir, dataset.char_to_idx_file)

        return utils.numpy_from_json(word_emb_file), utils.load(word_to_idx_file),\
                utils.numpy_from_json(char_emb_file), utils.load(char_to_idx_file)
    
    def get_simple_optimizer(self, proj_conf: config.ProjectConfig):
        opt = proj_conf.optimizer
        return torch.optim.AdamW(self.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2),
                                 weight_decay=opt.weight_decay, fused=opt.fused)

    # from https://github.com/karpathy/nanoGPT/blob/ae3a8d5fdd3ddb8b13fab182723476523961e3ab/model.py#L269
    def configure_optimizers(self, proj_conf: config.ProjectConfig):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """
        device_type = proj_conf.generic.device
        opt = proj_conf.optimizer
        learning_rate = opt.lr
        betas = (opt.beta1, opt.beta2)
        weight_decay = opt.weight_decay

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.LSTM)
        blacklist_weight_modules = (torch.nn.Embedding, )
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name
                # random note: because named_modules and named_parameters are recursive
                # we will see the same tensors p many many times. but doing it this way
                # allows us to know which parent module any tensor p belongs to...
                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # subtle: 'transformer.wte.weight' and 'lm_head.weight' are tied, so they
        # will appear in the no_decay and decay sets respectively after the above.
        # In addition, because named_parameters() doesn't return duplicates, it
        # will only return the first occurence, key'd by 'transformer.wte.weight', below.
        # so let's manually remove 'lm_head.weight' from decay set. This will include
        # this tensor into optimization via transformer.wte.weight only, and not decayed.
        # decay.remove('lm_head.weight')

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        # new PyTorch nightly has a new 'fused' option for AdamW that is much faster
        use_fused = (device_type == 'cuda') and ('fused' in inspect.signature(torch.optim.AdamW).parameters)
        print(f"using fused AdamW: {use_fused}")
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)

        return optimizer