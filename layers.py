import torch 
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class CharEmbeddingLayer(nn.Module):
    """ 
    Character embedding as described in the paper .
    First we take the embedding for each char. Then we pass these embeddings into a cnn to extract features and then we max pool 1d
    over width.
    """
    def __init__(self, char_emb: np.ndarray, char_to_idx: dict, 
                width_kernel_size: int, out_channels: int):
        super().__init__()
        self.char_emb = nn.Embedding.from_pretrained(torch.from_numpy(char_emb), padding_idx=0)
        self.char_to_idx = char_to_idx
        self.char_cnn = nn.Sequential(
            nn.Conv2d(1, out_channels, (self.char_emb.embedding_dim, width_kernel_size)),
            nn.SELU(),  
        )

        self.dp = nn.Dropout(0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ 
        x: batch_size, num_words, num_chars
        return: batch_size, num_words, out_channels
        """ 

        batch_size = x.shape[0]

        # batch_size, num_words, num_chars, emb_size
        char_embs = self.dp(self.char_emb(x))

        # batch_size, num_words, emb_size, num_chars
        char_embs = torch.transpose(char_embs, 2, 3)

        # batch_size * num_words, emb_size, num_chars
        char_embs = char_embs.view(-1, char_embs.shape[2], char_embs.shape[3])

        # add a channel dim for the cnn
        # batch_size * num_words, 1, emb_size, num_chars
        char_embs = char_embs.unsqueeze(1)
        
        # batch_size * num_words, out_channels, 1, width_out
        out = self.char_cnn(char_embs)

        # batch_size * num_words, out_channels, width_out
        out = out.squeeze()

        # batch_size * num_words, out_channels
        out = F.max_pool1d(out, out.shape[-1]).squeeze()

        # batch_size, num_words, out_channels
        out = out.view(batch_size, -1, out.shape[-1])

        return out


class HighwayLayer(nn.Module):
    """ 
    Simple implementation with a single layer. No dim change for the input x.
    """
    def __init__(self, layer_dim):
        super().__init__()

        self.transform = nn.Sequential(
            nn.Linear(layer_dim, layer_dim),
            nn.LeakyReLU(),
        )

        self.gate = nn.Sequential(
            nn.Linear(layer_dim, layer_dim),
            nn.Sigmoid(),
        )

    def forward(self, x):
        t = self.transform(x)
        g = self.transform(x)
        return t * g + x * (1 - g)
    

class WordEmbeddingLayer(nn.Module):
    """ 
    This layer describes the part 2) of the paper. 
    It combines the character embedding layer, word embedding + the highway layer.
    """

    def __init__(self, word_emb: np.ndarray, word_to_idx: dict, char_emb: np.ndarray, char_to_idx: dict, 
                 char_cnn_kernel_size: int, char_out_channels: int):
        super().__init__()
        self.word_emb = nn.Embedding.from_pretrained(torch.from_numpy(word_emb), padding_idx=0)
        self.word_to_idx = word_to_idx
        self.pad = "<PAD>"
        self.oov = "<OOV>"

        self.char_emb_layer = CharEmbeddingLayer(char_emb, char_to_idx, char_cnn_kernel_size, char_out_channels)

        self.highway_layer = HighwayLayer(self.word_emb.embedding_dim + char_out_channels)

    def forward(self, x: torch.Tensor, x_char: torch.Tensor) -> torch.Tensor:
        """ 
        x: batch_size, num_words 
        return: batch_size, num_words, word_emb + out_channels
        """
        
        # batch_size, num_words, out_channels
        char_embs = self.char_emb_layer(x_char)
        
        # batch_size, num_words, word_emb
        word_embs = self.word_emb(x)

        # batch_size, num_words, word_emb + out_channels
        out = torch.cat((word_embs, char_embs), -1)
        out = self.highway_layer(out)
        return out
    


class LSTMEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 1, dropout=0):
        super().__init__()
        self.dp = nn.Dropout(dropout)
        self.lstm = nn.LSTM(input_dim, hidden_dim, bidirectional=True, num_layers=num_layers, batch_first=True, 
                            dropout=dropout if num_layers > 1 else 0)
    
    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """ 
        x: batch_first, num_words, word_emb + out_channels
        return: batch_first, num_words, hidden_dim * 2
        """
        # Save original padded length for use by pad_packed_sequence
        orig_len = x.size(1)

        # TODO: check if it is better to do sorting in the dataloader collate func.
        # Sort by length and pack sequence for RNN
        lengths = lengths.cpu()

        lengths, sort_idx = lengths.sort(0, descending=True)
        # lengths = torch.tensor([l.item() for l in lengths], dtype=torch.int64, device="cpu")
        # batch_size, seq_len, word_emb + out_channels
        x = x[sort_idx]     
        x = torch.nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True)

        # batch_size, seq_len, hidden_dim * 2
        x, (h, c) = self.lstm(x) 

        # Unpack and reverse sort
        x, _ = torch.nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        _, unsort_idx = sort_idx.sort(0)
        # batch_size, seq_len, hidden_dim * 2
        x = x[unsort_idx]  
        
        x = self.dp(x)
        return x
    

class AttentionFlowLayer(nn.Module):
    """ 
    Add dropout?
    """

    def __init__(self, hidden_dim, dropout):
        super().__init__()
        # self.proj_c = nn.Linear(hidden_dim * 2, 1)
        # self.proj_q = nn.Linear(hidden_dim * 2, 1)
        # self.proj_cq = nn.Linear(hidden_dim * 2, 1)

        self.dp = nn.Dropout(dropout)

        self.proj_c = nn.Parameter(torch.zeros(hidden_dim * 2, 1))  
        self.proj_q = nn.Parameter(torch.zeros(hidden_dim * 2, 1))  
        self.proj_cq = nn.Parameter(torch.zeros(1, 1, hidden_dim * 2))

        for weight in (self.proj_c, self.proj_q, self.proj_cq):
            nn.init.xavier_uniform_(weight)

    def forward(self, c, q):
        """ 
        c: batch_size, context_len, hidden_dim * 2
        q: batch_size, query_len, hidden_dim * 2
        """

        context_len = c.shape[1]
        # query_len = q.shape[1]

        # batch_size, context_len, query_len
        similarity = self.compute_similarity(c, q)

        # context to query
        # batch_size, context_len, query_len
        att_weights_context_to_query = F.softmax(similarity, dim=-1)
        # batch_size, context_len, hidden_dim * 2
        context_to_query = torch.bmm(att_weights_context_to_query, q)    

        # query to context 
        # batch_size, 1, context_len
        att_weights_query_to_context = F.softmax(torch.max(similarity, dim=-1).values, dim=-1).unsqueeze(1)
        
        # batch_size, 1, hidden_dim * 2
        query_to_context = torch.bmm(att_weights_query_to_context, c) 
        
        # batch_size, context_len, hidden_dim * 2
        query_to_context = query_to_context.repeat(1, context_len, 1)


        # for now i use the simplest method to get the combined representation, a simple concat.
        # batch_size, context_len, hidden_dim * 8
        out = torch.cat((c, context_to_query, c * query_to_context, c * context_to_query), dim=-1)

        return out

    def compute_similarity(self, c, q):
        """ 
        c: batch, context_len, 2 * hidden_dim
        q: batch, query_len, 2 * hidden_dim
        
        """  
        c = self.dp(c)
        q = self.dp(q)

        s0 = (c @ self.proj_c).expand(-1, -1, q.shape[1])
        s1 = (q @ self.proj_q).transpose(1, 2).expand(-1, c.shape[1], -1)
        s2 = (c * self.proj_cq) @ q.transpose(1, 2)

        return s0 + s1 + s2
    


class QAOutputLayer(nn.Module):
    """ 
    Question answering output layer as described in the paper.
    The final layer can be adapted to the task to be learned.
    """
    def __init__(self, hidden_dim):
        super().__init__() 
        self.p1 = nn.Linear(hidden_dim * 10, 1)

        self.p2 = nn.Linear(hidden_dim * 10, 1)

        self.lstm = LSTMEncoder(hidden_dim * 2, hidden_dim)


    def forward(self, attention, modeling, lengths):
        # log_softmax because we use NLL.
        p1 = self.p1(torch.cat((attention, modeling), dim=-1)).squeeze()
        p1 = F.log_softmax(p1, dim=-1)

        p2 = self.lstm(modeling, lengths)
        p2 = self.p2(torch.cat((attention, p2), dim=-1)).squeeze()
        p2 = F.log_softmax(p2, dim=-1)

        return p1, p2
    