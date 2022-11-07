import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import sys


class PHVMConfig:
    def __init__(self):
        # rnn
        self.PHVM_rnn_direction = 'bi'
        self.PHVM_rnn_type = 'gru'

        # embedding
        self.share_vocab = False
        self.PHVM_word_dim = 300
        self.PHVM_key_dim = 30
        self.PHVM_val_dim = 100
        self.PHVM_cate_dim = 10

        # group
        self.PHVM_group_selection_threshold = 0.5
        self.PHVM_stop_threshold = 0.5
        self.PHVM_max_group_cnt = 30
        self.PHVM_max_sent_cnt = 10

        # type
        self.PHVM_use_type_info = False
        self.PHVM_type_dim = 30
        self.PHVM_rnn_bidirectional = True
        self.PHVM_rnn_bidirectional_num = 2

        # encoder
        self.PHVM_encoder_input_dim = self.PHVM_key_dim+self.PHVM_val_dim
        self.PHVM_encoder_output_dim = 100
        self.PHVM_encoder_num_layer = 1

        # group_decoder
        self.PHVM_group_decoder_input_dim = self.PHVM_encoder_output_dim*2
        self.PHVM_group_decoder_output_dim = 100
        self.PHVM_group_decoder_num_layer = 1

        # group encoder
        self.PHVM_group_encoder_input_dim = self.PHVM_encoder_output_dim*2
        self.PHVM_group_encoder_output_dim = 100
        self.PHVM_group_encoder_num_layer = 1
        
        # attention
        self.attention_hidden_size = 300

        # decoder
        self.PHVM_decoder_input_dim = self.PHVM_encoder_output_dim*2+self.PHVM_word_dim
        self.PHVM_decoder_output_dim = 300
        self.PHVM_decoder_num_layer = 2

        # latent
        self.PHVM_plan_latent_dim = 200
        self.PHVM_sent_latent_dim = 200

        # latent_decoder
        self.PHVM_latent_decoder_input_dim = self.PHVM_decoder_output_dim*2+self.PHVM_sent_latent_dim
        self.PHVM_latent_decoder_output_dim = 300
        self.PHVM_latent_decoder_num_layer = 1

        # sent_top_encoder
        self.PHVM_sent_top_encoder_dim = 300
        self.PHVM_sent_top_encoder_num_layer = 1

        # text post encoder
        self.PHVM_text_post_encoder_input_dim = self.PHVM_word_dim
        self.PHVM_text_post_encoder_output_dim = 300
        self.PHVM_text_post_encoder_num_layer = 1

        # sent_post_encoder
        self.PHVM_sent_post_encoder_input_dim = self.PHVM_word_dim
        self.PHVM_sent_post_encoder_output_dim = 300
        self.PHVM_sent_post_encoder_num_layer = 1

        # bow
        self.PHVM_bow_hidden_dim = 200

        # training
        self.PHVM_learning_rate = 0.001
        self.PHVM_num_training_step = 100000
        self.PHVM_sent_full_KL_step = 20000
        self.PHVM_plan_full_KL_step = 40000
        self.PHVM_dropout = 0

        # inference
        self.PHVM_beam_width = 10
        self.PHVM_maximum_iterations = 50


class Attention(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size*2
        self.attn = nn.Linear(input_size, self.hidden_size)
        self.v = nn.Parameter(torch.rand(self.hidden_size))
        stdv = 1. / math.sqrt(self.v.size(0))
        self.v.data.uniform_(-stdv, stdv)

    def forward(self, hidden, encoder_outputs):
        timestep = encoder_outputs.size(1)
        h = hidden.repeat(timestep, 1, 1).transpose(0, 1)  # [32, 512]=>[32, 27, 512]
        attn_energies = self.score(h, encoder_outputs)  # =>[B*T]
        return F.softmax(attn_energies, dim=1).unsqueeze(1)  # [B*T]=>[B*1*T]

    def score(self, hidden, encoder_outputs):
        # [B*T*2H]->[B*T*H]
        energy = F.relu(self.attn(torch.cat([hidden, encoder_outputs], 2)))
        energy = energy.transpose(1, 2)  # [B*H*T]
        v = self.v.repeat(encoder_outputs.size(0), 1).unsqueeze(1)  # [B*1*H]
        energy = torch.bmm(v, energy)  # [B*1*H] bmm [B*H*T]=>[B*1*T]
        return energy.squeeze(1)  # [B*T]


class AttentionV2(nn.Module):

    def __init__(self, input_size, hidden_size):
        super(AttentionV2, self).__init__()
        self.i2h = nn.Linear(input_size, hidden_size)
        self.h2h = nn.Linear(hidden_size, hidden_size)  # either i2i or h2h should have bias
        self.score = nn.Linear(hidden_size, 1)
        self.hidden_size = hidden_size

    def forward(self, prev_hidden, batch_H):
        # [batch_size x num_encoder_step x num_channel] -> [batch_size x num_encoder_step x hidden_size]
        batch_H_proj = self.i2h(batch_H)
        prev_hidden_proj = self.h2h(prev_hidden[0]).unsqueeze(1)
        e = self.score(torch.tanh(batch_H_proj + prev_hidden_proj)).squeeze(2)  # batch_size x num_encoder_step * 1
        # print(batch_H.size(), prev_hidden[0].size(), batch_H_proj.size(), prev_hidden_proj.size())

        alpha = F.softmax(e, dim=1)     # batch_size x num_encoder_step
        return alpha.unsqueeze(1)


class PHVM(nn.Module):
    def __init__(self, train_flag, batch_size, device, key_vocab_size, val_vocab_size, tgt_vocab_size, cate_vocab_size, 
                 type_vocab_size, key_wordvec=None, val_wordvec=None, tgt_wordvec=None):
        super(PHVM, self).__init__()
        self.batch_size = batch_size
        self.train_flag = train_flag
        self.device = device
        self.global_step = 0
        self.config = PHVMConfig()
        self.key_vocab_size = key_vocab_size
        self.val_vocab_size = val_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.cate_vocab_size = cate_vocab_size
        self.type_vocab_size = type_vocab_size
        
        self.sent_KL_loss = torch.tensor(0, dtype=torch.float, device=self.device)
        self.group_dec_loss = torch.tensor(0, dtype=torch.float, device=self.device)
        self.bow_loss = torch.tensor(0, dtype=torch.float, device=self.device)
        self.sent_dec_loss = torch.tensor(0, dtype=torch.float, device=self.device)
        
        self.key_val_encode_rnn = nn.LSTM(self.config.PHVM_encoder_input_dim, 
                                     self.config.PHVM_encoder_output_dim, 
                                     bidirectional=self.config.PHVM_rnn_bidirectional)
        self.group_decoder_rnn = nn.LSTM(self.config.PHVM_group_decoder_input_dim, 
                                     self.config.PHVM_group_decoder_output_dim, 
                                     bidirectional=self.config.PHVM_rnn_bidirectional)
        self.group_encoder_rnn = nn.LSTM(self.config.PHVM_group_encoder_input_dim, 
                                     self.config.PHVM_group_encoder_output_dim, 
                                     bidirectional=self.config.PHVM_rnn_bidirectional)
        self.latent_decoder_rnn = nn.LSTM(self.config.PHVM_latent_decoder_input_dim, 
                                     self.config.PHVM_latent_decoder_output_dim, 
                                     bidirectional=self.config.PHVM_rnn_bidirectional)
        self.decoder_rnn = nn.LSTM(self.config.PHVM_decoder_input_dim, 
                                     self.config.PHVM_decoder_output_dim, 
                                     self.config.PHVM_decoder_num_layer, 
                                     bidirectional=self.config.PHVM_rnn_bidirectional)
        self.text_encode_rnn = nn.LSTM(self.config.PHVM_text_post_encoder_input_dim,
                                     self.config.PHVM_text_post_encoder_output_dim,
                                     bidirectional=self.config.PHVM_rnn_bidirectional)
        self.sent_encode_rnn = nn.LSTM(self.config.PHVM_sent_post_encoder_input_dim,
                                     self.config.PHVM_sent_post_encoder_output_dim,
                                     bidirectional=self.config.PHVM_rnn_bidirectional)
        
        self.prior_input_fc_Linear = nn.Linear(self.config.PHVM_encoder_output_dim*2+self.config.PHVM_cate_dim, self.config.PHVM_plan_latent_dim * 2)
        self.prior_fc_nd_Linear = nn.Linear(self.config.PHVM_plan_latent_dim * 2, self.config.PHVM_plan_latent_dim * 2)
        self.prior_fc_actv = nn.Tanh()
        
        self.post_input_fc_Linear = nn.Linear(self.config.PHVM_encoder_output_dim*2+self.config.PHVM_cate_dim+self.config.PHVM_text_post_encoder_output_dim*2, self.config.PHVM_plan_latent_dim * 2)
        
        self.group_init_h0_fc = nn.Linear(self.config.PHVM_encoder_output_dim*2+self.config.PHVM_cate_dim+self.config.PHVM_plan_latent_dim, self.config.PHVM_group_decoder_output_dim)
        self.group_init_c0_fc = nn.Linear(self.config.PHVM_encoder_output_dim*2+self.config.PHVM_cate_dim+self.config.PHVM_plan_latent_dim, self.config.PHVM_group_decoder_output_dim)

        self.stop_clf = nn.Linear(self.config.PHVM_group_decoder_output_dim*2, 1)
        
        self.group_fc_1 = nn.Linear(self.config.PHVM_encoder_output_dim*2+self.config.PHVM_group_decoder_output_dim*2, self.config.PHVM_encoder_output_dim)
        self.group_fc_2 = nn.Linear(self.config.PHVM_encoder_output_dim, 2)
        
        self.prior_sent_fc_layer = nn.Linear(self.config.PHVM_latent_decoder_output_dim*2+self.config.PHVM_encoder_output_dim*2, self.config.PHVM_sent_latent_dim*2)
        self.post_sent_fc_layer = nn.Linear(self.config.PHVM_latent_decoder_output_dim*2+self.config.PHVM_encoder_output_dim*2+self.config.PHVM_sent_post_encoder_output_dim*2, self.config.PHVM_sent_latent_dim*2)
        
        self.plan_init_h_state_fc = nn.Linear(self.config.PHVM_encoder_output_dim*2+self.config.PHVM_cate_dim+self.config.PHVM_plan_latent_dim+self.config.PHVM_group_encoder_output_dim*2, self.config.PHVM_latent_decoder_output_dim)
        self.plan_init_c_state_fc = nn.Linear(self.config.PHVM_encoder_output_dim*2+self.config.PHVM_cate_dim+self.config.PHVM_plan_latent_dim+self.config.PHVM_group_encoder_output_dim*2, self.config.PHVM_latent_decoder_output_dim)
        
        self.prior_fc_layer = nn.Linear(self.config.PHVM_latent_decoder_output_dim*2+self.config.PHVM_encoder_output_dim, self.config.PHVM_sent_latent_dim * 2)
        
        self.sent_dec_h_state_fc = nn.Linear(self.config.PHVM_latent_decoder_output_dim*2+self.config.PHVM_encoder_output_dim*2+self.config.PHVM_sent_latent_dim, self.config.PHVM_decoder_output_dim)
        self.sent_dec_c_state_fc = nn.Linear(self.config.PHVM_latent_decoder_output_dim*2+self.config.PHVM_encoder_output_dim*2+self.config.PHVM_sent_latent_dim, self.config.PHVM_decoder_output_dim)
        
        self.bow_fc_1 = nn.Linear(self.config.PHVM_latent_decoder_output_dim*2+self.config.PHVM_encoder_output_dim*2+self.config.PHVM_sent_latent_dim, self.config.PHVM_bow_hidden_dim)
        self.bow_fc_2 = nn.Linear(self.config.PHVM_bow_hidden_dim, self.tgt_vocab_size)
        
        self.word_attention_fc = nn.Linear(self.config.PHVM_encoder_output_dim*2+self.config.PHVM_decoder_output_dim*2, self.tgt_vocab_size)
        
        self.attention = Attention(self.config.PHVM_encoder_output_dim*2+self.config.PHVM_decoder_output_dim*2, self.config.attention_hidden_size)
        
        for name, param in self.named_parameters():
            if "weight" in name:
                nn.init.xavier_normal_(param)
            else:
                nn.init.zeros_(param)
        
        # 需要放到模型初始化后面，防止nn.Embedding的参数被置为0
        self.make_embedding(key_wordvec, val_wordvec, tgt_wordvec)
    
    def make_embedding(self, key_wordvec, val_wordvec, tgt_wordvec):
        
        if tgt_wordvec is None:
            self.word_embedding = nn.Embedding(self.tgt_vocab_size, self.config.PHVM_word_dim)
        else:
            self.word_embedding = nn.Embedding.from_pretrained(torch.FloatTensor(tgt_wordvec))
            
        if key_wordvec is None:
            self.key_embedding = nn.Embedding(self.key_vocab_size, self.config.PHVM_key_dim)
        else:
            self.key_embedding = nn.Embedding.from_pretrained(torch.FloatTensor(key_wordvec))
        
        if val_wordvec is None:
            self.val_embedding = nn.Embedding(self.val_vocab_size, self.config.PHVM_val_dim)
        else:
            self.val_embedding = nn.Embedding.from_pretrained(torch.FloatTensor(val_wordvec))
        
        self.cate_embedding = nn.Embedding(self.cate_vocab_size, self.config.PHVM_cate_dim)
        '''
        input = torch.tensor([[1,2,4,5],[4,3,2,9]])
        text = self.word_embedding(input)
        print(text.max())
        input = torch.tensor([[1,2,4,5],[4,3,2,9]])
        text = self.key_embedding(input)
        print(text.max())
        '''
    
    def sample_gaussian(self, shape, mu, logvar):
        x = torch.normal(0, 1, size=shape).cuda()
        
        return mu + torch.exp(logvar / 2) * x

    def KL_divergence(self, prior_mu, prior_logvar, post_mu, post_logvar, reduce_mean=True):
        #print(torch.exp(post_logvar - prior_logvar), torch.pow(post_mu - prior_mu, 2), \
        #torch.pow(post_mu - prior_mu, 2) / torch.exp(prior_logvar), (post_logvar - prior_logvar))
        
        #print(post_mu - prior_mu, torch.pow(post_mu - prior_mu, 2))
        divergence = 0.5 * torch.sum(torch.exp(post_logvar - prior_logvar)
                                         + torch.pow(post_mu - prior_mu, 2) / torch.exp(prior_logvar)
                                         - 1 - (post_logvar - prior_logvar), dim=1)
        #print(divergence)
        if reduce_mean:
            return torch.mean(divergence, dim=0)
        else:
            return divergence
    
    def sequence_mask(self, lengths, maxlen=None, dtype=torch.bool):
        if maxlen is None:
            maxlen = lengths.max()
        row_vector = torch.arange(0, maxlen, 1, device=self.device)
        matrix = torch.unsqueeze(lengths, dim=-1)
        mask = row_vector < matrix

        mask.type(dtype)
        return mask
    
    def select(self, group_prob, max_gcnt):
        gid = []
        glen = []
        np_group_prob = group_prob.cpu().detach().numpy()
        for gfid, prob in enumerate(np_group_prob):
            tmp = []
            max_gsid = -1
            max_p = -1
            for gsid, p in enumerate(prob):
                if p >= self.config.PHVM_group_selection_threshold:
                    tmp.append(gsid)
                if p > max_p:
                    max_gsid = gsid
                    max_p = p
            if len(tmp) == 0:
                tmp.append(max_gsid)
            gid.append(tmp)
            glen.append(len(tmp))
        for item in gid:
            if len(item) < max_gcnt:
                item += [0] * (max_gcnt - len(item))
        return gid, glen
    
    def gather_nd_for_group_decode(self, params, indices):
    
        gather_tensor = torch.zeros((1, params.size(1), params.size(2)), dtype=params.dtype, device=self.device)
        for idx, ids in enumerate(indices):
            slct_tensor = torch.select(params, 0, idx)
            ids_tensor = torch.tensor(ids, device=self.device)
            g_tensor = torch.index_select(slct_tensor, 0, ids_tensor)
            expended_g_tensor = g_tensor.unsqueeze(0)
            gather_tensor = torch.cat((gather_tensor, expended_g_tensor), 0)
        
        return gather_tensor[1:, :, :]
    
    def gather_nd_for_group_encode(self, params, indices):
    
        #print(params.size(), indices.size())
        batch_tensor = torch.zeros((1, indices.size(1), indices.size(2), params.size(2)), dtype=params.dtype, device=self.device)   # shape[batch_size, group_cnt, ]
        for i in range(indices.size(0)):
            group_tensor = torch.zeros((1, indices.size(2), params.size(2)), dtype=params.dtype, device=self.device)
            batch_slct_tensor = torch.select(params, 0, i)
            batch_ids_tensor = torch.select(indices, 0, i)
            for j in range(indices.size(1)):
                group_ids_tensor = torch.select(batch_ids_tensor, 0, j)
                g_tensor = torch.index_select(batch_slct_tensor, 0, group_ids_tensor)
                expended_g_tensor = g_tensor.unsqueeze(0)
                #print(group_tensor.size(), expended_g_tensor.size())
                group_tensor = torch.cat((group_tensor, expended_g_tensor), 0)
            expended_group_tensor = group_tensor[1:, :, :].unsqueeze(0)
            batch_tensor = torch.cat((batch_tensor, expended_group_tensor), 0)
        
        return batch_tensor[1:, :, :, :]
    
    def gather_nd(self, params, indices):
        '''
        4D example
        params: tensor shaped [n_1, n_2, n_3, n_4] --> 4 dimensional
        indices: tensor shaped [m_1, m_2, m_3, m_4, 4] --> multidimensional list of 4D indices
        
        returns: tensor shaped [m_1, m_2, m_3, m_4]
        
        ND_example
        params: tensor shaped [n_1, ..., n_p] --> d-dimensional tensor
        indices: tensor shaped [m_1, ..., m_i, d] --> multidimensional list of d-dimensional indices
        
        returns: tensor shaped [m_1, ..., m_1]
        '''

        out_shape = indices.shape[:-1]
        indices = indices.unsqueeze(0).transpose(0, -1) # roll last axis to fring
        ndim = indices.shape[0]
        indices = indices.long()
        idx = torch.zeros_like(indices[0], device=indices.device).long()
        m = 1
        
        for i in range(ndim)[::-1]:
            idx += indices[i] * m 
            m *= params.size(i)
        out = torch.take(params, idx)
        return out.view(out_shape)
    
    def input_embedding(self, key_input, val_input, cate_input, text=None):
        key_embed = self.key_embedding(key_input)   # shape[batch_size, key_cnt, dim]
        val_embed = self.val_embedding(val_input)   # shape[batch_size, value_cnt, dim]
        key_val_embed = torch.cat((key_embed, val_embed), 2)    # shape[batch_size, key_cnt, dim]
        cate_embed = self.cate_embedding(cate_input)    # shape[batch_size, cate_cnt, dim]
        #input = torch.tensor([[1,2,4,5],[4,3,2,9]], dtype=torch.long, device=self.device)
        text_embed = self.word_embedding(text)  # shape[batch_size, text_cnt, dim]
        #print(text_embed.max())
        
        return key_val_embed, cate_embed, text_embed
    
    def input_encode(self, key_val_embed):
        key_val_embed = key_val_embed.permute(1, 0, 2)  # shape[key_cnt, batch_size, dim]
        #print(key_val_embed.max())
        key_val_encode_output, (key_val_encode_state, _) = self.key_val_encode_rnn(key_val_embed)   # shape[key_cnt, batch_size, dim], shape[D*num_layers, batch_size, dim]
        #print(key_val_encode_output.max())
        key_val_encode_embed = torch.cat((key_val_encode_state[-2, :, :], key_val_encode_state[-1, :, :]), 1)  # 需调试判断是否用index0和1, shape[batch_size, dim]
        
        return key_val_encode_output.permute(1, 0, 2), key_val_encode_embed
    
    def text_encode(self, text_embed):
        text_embed = text_embed.permute(1, 0, 2)    # shape[text_cnt, batch_size, dim]
        text_encode_output, (text_encode_h_state, _) = self.text_encode_rnn(text_embed) # shape[text_cnt, batch_size, dim], shape[D*num_layers, batch_size, dim]
        text_encode_embed = torch.cat((text_encode_h_state[-2, :, :], text_encode_h_state[-1, :, :]), 1)  # 需调试判断是否用index0和1, shape[batch_size, dim]
        
        return text_encode_output, text_encode_embed
    
    def input_sample_encode(self, cate_embed, key_val_encode_embed, tgt_embed=None):
    
        plan_KL_weight = torch.tensor(0, dtype=torch.float, device=self.device)
        
        # 计算先验随机采样向量
        #print(key_val_encode_embed.max())
        prior_input = torch.cat((key_val_encode_embed, cate_embed), 1)  # shape[batch_size, dim]
        prior_fc = self.prior_input_fc_Linear(prior_input)  # shape[batch_size, dim]
        prior_fc = self.prior_fc_actv(prior_fc) # shape[batch_size, dim]
        prior_fc_nd = self.prior_fc_nd_Linear(prior_fc) # shape[batch_size, dim]
        prior_mu, prior_logvar = torch.split(prior_fc_nd, self.config.PHVM_plan_latent_dim, 1)  # shape[batch_size, dim]
        prior_z_plan = self.sample_gaussian((self.batch_size, self.config.PHVM_plan_latent_dim), prior_mu, prior_logvar)    # shape[batch_size, dim]
        
        if self.train_flag:
            # 计算后验随机采样向量
            post_input = torch.cat((key_val_encode_embed, cate_embed, tgt_embed), 1)
            post_fc = self.post_input_fc_Linear(post_input)
            post_mu, post_logvar = torch.split(post_fc, self.config.PHVM_plan_latent_dim, 1)
            post_z_plan = self.sample_gaussian((self.batch_size, self.config.PHVM_plan_latent_dim), post_mu, post_logvar)
            
            #print(post_z_plan)
            #print(post_mu.max())
            #print(prior_mu.max())
            # 计算随机采样向量的先验和后验的KL散度
            #print(prior_mu, prior_logvar, post_mu, post_logvar)
            self.plan_KL_loss = self.KL_divergence(prior_mu, prior_logvar, post_mu, post_logvar)
            plan_KL_weight = torch.minimum(torch.tensor(1, dtype=torch.float, device=self.device), torch.tensor(self.global_step, dtype=torch.float, device=self.device) / torch.tensor(self.config.PHVM_plan_full_KL_step, dtype=torch.float, device=self.device))
            
            dec_input = torch.cat((cate_embed, key_val_encode_embed, post_z_plan), 1)   # shape[batch_size, dim]
        else:
            dec_input = torch.cat((cate_embed, key_val_encode_embed, prior_z_plan), 1)  # shape[batch_size, dim]
        
        return dec_input, plan_KL_weight
    
    def sent_word_decode_train(self, dec_input, key_val_encode_output, input_lens, output_lens, groups, glens, group_cnt, target_input, target_output):
        
        i = 0
        stop_sign = []
        
        # planning过程初始化
        tile_dec_input = torch.tile(dec_input.unsqueeze(0), (self.config.PHVM_rnn_bidirectional_num, 1, 1))
        group_h_state = self.group_init_h0_fc(tile_dec_input)   # shape[1, batch_size, dim]
        group_c_state = self.group_init_c0_fc(tile_dec_input)   # shape[1, batch_size, dim]
        gbow = torch.zeros((self.batch_size, self.config.PHVM_encoder_output_dim*2), dtype=torch.float32, device=self.device)
        
        # 利用label信息得到group编码
        group_bow, group_mean_bow, group_embed = self.group_encode(key_val_encode_output, groups, glens, group_cnt)
        
        # 初始化
        sent_state = torch.zeros((self.batch_size, self.config.PHVM_decoder_output_dim*2), dtype=torch.float32, device=self.device)
        sent_z = torch.zeros((self.batch_size, self.config.PHVM_sent_latent_dim), dtype=torch.float32, device=self.device)
        input_group_concat = torch.cat((dec_input, group_embed), 1)
        tile_input_group_concat = torch.tile(input_group_concat.unsqueeze(0), (self.config.PHVM_rnn_bidirectional_num, 1, 1))
        plan_h_state = self.plan_init_h_state_fc(tile_input_group_concat)   # shape[1, batch_size, dim]
        plan_c_state = self.plan_init_c_state_fc(tile_input_group_concat)   # shape[1, batch_size, dim]
        # latent_decoder_rnn是单向时
        #plan_h_state = self.plan_init_h_state_fc(input_group_concat).unsqueeze(0)
        #plan_c_state = self.plan_init_c_state_fc(input_group_concat).unsqueeze(0)
        
        #sent_logit = torch.zeros((target_input.size(0), target_input.size(1), target_input.size(2), self.tgt_vocab_size), device=self.device)
        #print('1111111', sent_logit.requires_grad)
        #print(sent_logit)
        sent_sign = []
        input_group_len = target_input.size(1)
        
        while i < input_group_len:

            sent_gid = groups[:, i, :]  # shape[batch_size, id_cnt]
            sent_group = group_bow[:, i, :, :]  # shape[batch_size, id_cnt, dim]
            
            # planning过程
            gout, (group_h_state, group_c_state) = self.group_decoder_rnn(gbow.unsqueeze(0), (group_h_state, group_c_state))
            tile_gout = torch.tile(gout.permute(1, 0, 2), (1, key_val_encode_output.size(1), 1))
            group_fc_input = torch.cat((key_val_encode_output, tile_gout), 2)
            group_logit = self.group_fc_2(torch.tanh(self.group_fc_1(group_fc_input)))   # shape[batch_size, key_value_cnt, 2]
            
            # 计算关于group所包含的key-value对id的真实值与预测值的loss
            #print(sent_gid, glens[:, i])
            group_label = F.one_hot(sent_gid.long(), num_classes=group_logit.size(1))  # shape[batch_size, key_value_cnt, key_value_cnt]
            #print(group_label)
            group_label = torch.sum(group_label, 1) # shape[batch_size, 1, key_value_cnt]
            len_tensor = torch.zeros(sent_gid.size(0), dtype=torch.int32, device=self.device)
            lenght = torch.tensor(sent_gid.size(1), dtype=torch.int32, device=self.device)
            len_tensor += lenght
            len_tensor -= glens[:, i]
            tmp_group_label = group_label[:, 0] - len_tensor
            group_label[:, 0] = tmp_group_label
            #print(masked_group_logit.transpose(1, 2).size(), group_label.squeeze(1).size())
            self.group_dec_loss += F.cross_entropy(group_logit.transpose(1, 2), group_label.squeeze(1), ignore_index=0)
            
            # 计算group stop值
            stop_sign.append(self.stop_clf(gout.squeeze(0)))
            
            gbow = group_mean_bow[:, i, :]
            
            # 句子级先验知识编码
            sent_input = self.word_embedding(target_input[:, i, :]).permute(1, 0, 2)
            sent_encoder_output, (sent_encoder_h_state, sent_encoder_c_state) = self.sent_encode_rnn(sent_input)
            sent_encode_embed = torch.cat((sent_encoder_h_state[-2, :, :], sent_encoder_h_state[-1, :, :]), 1)
            
            # 句子级编码
            plan_input = torch.cat((sent_state, sent_z), 1).unsqueeze(0)
            sent_latent_embed, (plan_h_state, plan_c_state) = self.latent_decoder_rnn(plan_input, (plan_h_state, plan_c_state))
            #sent_cond_embed = torch.squeeze(sent_latent_embed, 0)
            sent_cond_embed = torch.cat((plan_h_state[-2, :, :], plan_h_state[-1, :, :]), 1)
            
            # 句子级先验随机向量
            sent_prior_input = torch.cat((sent_cond_embed, gbow), 1)
            sent_prior_fc = self.prior_sent_fc_layer(sent_prior_input)
            sent_prior_mu, sent_prior_logvar = torch.split(sent_prior_fc, self.config.PHVM_sent_latent_dim, 1)
            
            # 句子级后验随机向量
            sent_post_input = torch.cat((sent_cond_embed, gbow, sent_encode_embed), 1)
            sent_post_fc = self.post_sent_fc_layer(sent_post_input)
            sent_post_mu, sent_post_logvar = torch.split(sent_post_fc, self.config.PHVM_sent_latent_dim, 1)
            
            # 计算先验分布和后验分布的KL损失
            self.sent_KL_loss += self.KL_divergence(sent_prior_mu, sent_prior_logvar, sent_post_mu, sent_post_logvar)
            
            sent_z = self.sample_gaussian((self.batch_size, self.config.PHVM_sent_latent_dim), sent_prior_mu, sent_prior_logvar)
            sent_cond_z_embed = torch.cat((sent_cond_embed, sent_z), 1)
            
            # word级解码初始化
            sent_dec_state = torch.cat((sent_cond_z_embed, gbow), 1)    # shape[batch_size, dim]
            sent_dec_state = sent_dec_state.unsqueeze(0)    # shape[1, batch_size, dim]
            tile_sent_dec_state = torch.tile(sent_dec_state, (self.config.PHVM_decoder_num_layer*2, 1, 1))    # shape[self.config.PHVM_decoder_num_layer*Direction, batch_size, dim]
            sent_dec_h_state = self.sent_dec_h_state_fc(tile_sent_dec_state)
            sent_dec_c_state = self.sent_dec_c_state_fc(tile_sent_dec_state)
            
            word_idx = torch.zeros((sent_group.size(0)), dtype=torch.int32, device=self.device)
            
            group_sign = []
            for t in range(target_input.size(2)):

                embedded = self.word_embedding(word_idx).unsqueeze(0)
                sent_dec_h_state_for_attn = torch.cat((sent_dec_h_state[0, :, :], sent_dec_h_state[1, :, :]), 1)    # shape[batch_size, dim]
                attn_weights  = self.attention(sent_dec_h_state_for_attn, sent_group)
                context = attn_weights.bmm(sent_group)
                context = context.transpose(0, 1)
                rnn_input = torch.cat((embedded, context), 2)
                output, (sent_dec_h_state, sent_dec_c_state) = self.decoder_rnn(rnn_input, (sent_dec_h_state, sent_dec_c_state))
                output = output.squeeze(0)  # (1,B,N) -> (B,N)
                context = context.squeeze(0)
                output = self.word_attention_fc(torch.cat([output, context], 1))  # [32, 512] cat [32, 512] => [32, 512*2] => [32, tgt_vocab_size]
                
                group_sign.append(output.unsqueeze(1))
                #sent_logit[:, i, t, :] = output
                #print('2222222', sent_logit.requires_grad)
                word_idx = target_input[:, i, t]
                
                t += 1
            
            group_sign_logit = torch.cat(group_sign, 1)
            #sent_sign.append(group_sign_logit.unsqueeze(1))
            #sent_state = sent_dec_h_state[-1]
            sent_state = torch.cat((sent_dec_h_state[-2, :, :], sent_dec_h_state[-1, :, :]), 1)
            
            self.sent_dec_loss += F.cross_entropy(group_sign_logit.permute(0, 2, 1), target_output[:, i, :].long(), ignore_index=0)
            
            # 计算bow loss
            #'''
            tile_sent_dec_state = torch.tile(sent_dec_state, (target_output.size(2), 1, 1)) # shape[id_cnt_per_group, batch_size, dim]
            bow_logit = self.bow_fc_2(torch.tanh(self.bow_fc_1(tile_sent_dec_state)))    # shape[id_cnt_per_group, batch_size, tgt_vocab_size]
            bow_logit = bow_logit.permute(1, 2, 0)  # shape[batch_size, tgt_vocab_size, id_cnt_per_group]
            #print(tile_bow_logit.size(), target_output[:, i, :].size())
            self.bow_loss += F.cross_entropy(bow_logit, target_output[:, i, :].long())
            #'''
            i += 1
        
        self.sent_dec_loss /= torch.tensor(input_group_len, dtype=torch.float, device=self.device)
        #self.bow_loss /= torch.tensor(input_group_len, dtype=torch.float, device=self.device)
        
        #sent_logit = torch.cat(sent_sign, 1)
        #print(sent_logit)
        #self.sent_dec_loss = F.cross_entropy(sent_logit.permute(0, 3, 1, 2), target_input.long(), ignore_index=1)
        
        # stop loss
        stop_logit = torch.cat(stop_sign, 1)
        stop_label = group_cnt - (1 - torch.eq(group_cnt, 0).int())
        self.stop_loss = F.cross_entropy(stop_logit, stop_label.long())
    
    def loss_computation(self, plan_KL_weight):
        
        sent_KL_weight = torch.minimum(torch.tensor(1, dtype=torch.float, device=self.device), torch.tensor(self.global_step, dtype=torch.float, device=self.device) / torch.tensor(self.config.PHVM_sent_full_KL_step, dtype=torch.float, device=self.device))
        #self.bow_loss /= torch.tensor(self.batch_size, dtype=torch.float, device=self.device)
        anneal_sent_KL = sent_KL_weight * self.sent_KL_loss
        anneal_plan_KL = plan_KL_weight * self.plan_KL_loss
        #self.elbo_loss = self.sent_dec_loss + self.group_dec_loss + self.sent_KL_loss + self.plan_KL_loss
        #self.elbo_loss = self.sent_dec_loss + self.group_dec_loss + self.plan_KL_loss
        #self.elbo_loss /= torch.tensor(self.batch_size, dtype=torch.float, device=self.device)
        self.anneal_elbo_loss = self.sent_dec_loss + self.group_dec_loss + anneal_sent_KL + anneal_plan_KL
        #self.anneal_elbo_loss /= torch.tensor(self.batch_size, dtype=torch.float, device=self.device)
        self.train_loss = self.anneal_elbo_loss + self.stop_loss + self.bow_loss
        
        print(sent_KL_weight.item(), '=======' ,self.sent_KL_loss.item(), '=======' ,plan_KL_weight.item(), \
        '=======', self.plan_KL_loss.item(), '=======' , self.anneal_elbo_loss.item(), '=======' , \
        self.sent_dec_loss.item(), '=======' ,self.group_dec_loss.item(), '=======' ,anneal_sent_KL.item(), \
        '=======' ,anneal_plan_KL.item(), '=======' ,self.stop_loss.item(), '=======' ,self.bow_loss.item(), '=======' ,self.train_loss.item())
    
    def clear_zero(self):
        
        self.sent_KL_loss = torch.tensor(0, dtype=torch.float, device=self.device)
        '''
        self.plan_KL_loss = torch.tensor(0, dtype=torch.float, device=self.device)
        self.group_decode_loss = torch.tensor(0, dtype=torch.float, device=self.device)
        self.bow_loss = torch.tensor(0, dtype=torch.float, device=self.device)
        #self.elbo_loss = torch.tensor(0, dtype=torch.float, device=self.device)
        self.stop_loss = torch.tensor(0, dtype=torch.float, device=self.device)
        self.anneal_elbo_loss = torch.tensor(0, dtype=torch.float, device=self.device)
        self.sent_dec_loss = torch.tensor(0, dtype=torch.float, device=self.device)
        self.group_dec_loss = torch.tensor(0, dtype=torch.float, device=self.device)
        self.train_loss = torch.tensor(0, dtype=torch.float, device=self.device)
        '''

    def group_cond(self, i, stop):
        if i == 0:
            return True
        elif torch.eq(torch.min(stop), 0).item():
            if i >= self.config.PHVM_max_sent_cnt:
                return False
            else:
                return True
        else:
            return False
    
    def group_decode(self, dec_input, gourp_key_value_cnt, key_val_encode_output, input_lens):
    
        i = 0
        tile_dec_input = torch.tile(dec_input.unsqueeze(0), (2, 1, 1))
        group_h_state = self.group_init_h0_fc(tile_dec_input)   # shape[1, batch_size, dim]
        group_c_state = self.group_init_c0_fc(tile_dec_input)   # shape[1, batch_size, dim]
        # group_decoder_rnn是单向时
        #group_h_state = self.group_init_h0_fc(dec_input).unsqueeze(0)
        #group_c_state = self.group_init_c0_fc(dec_input).unsqueeze(0)
        gbow = torch.zeros((self.batch_size, self.config.PHVM_encoder_output_dim*2), dtype=torch.float32, device=self.device)
        stop = torch.zeros(self.batch_size, dtype=torch.int32, device=self.device)
        groups = torch.zeros((self.batch_size, 1, gourp_key_value_cnt), dtype=torch.int32, device=self.device)
        glens = torch.zeros((self.batch_size, 1), dtype=torch.int32, device=self.device)
        
        while self.group_cond(i, stop):
            # 计算group的解码向量
            gout, (group_h_state, group_c_state) = self.group_decoder_rnn(gbow.unsqueeze(0), (group_h_state, group_c_state))

            # 
            next_stop = torch.gt(torch.squeeze(self.stop_clf(gout.squeeze(0)), 1), self.config.PHVM_stop_threshold)
            stop += torch.mul(torch.eq(stop, 0.0).int() * next_stop.int(), i+1)     # 计算整个句子的group数

            # 基于rnn编码向量计算每队key和val属于该group的概率
            tile_gout = torch.tile(gout.permute(1, 0, 2), (1, gourp_key_value_cnt, 1))
            group_fc_input = torch.cat((key_val_encode_output, tile_gout), 2)
            group_logit = self.group_fc_2(torch.tanh(self.group_fc_1(group_fc_input)))
            src_mask = self.sequence_mask(input_lens, group_logit.size(1), dtype=torch.float32)
            group_prob = torch.sigmoid(group_logit[:, :, 1]) * src_mask
            
            p_gid, p_glen = self.select(group_prob, gourp_key_value_cnt)
            #gid = torch.reshape(gid, (b, -1, 2))
            gid = torch.tensor(p_gid, dtype=torch.int32, device=self.device)  # shape[batch_size, word_id_cnt]
            glen = torch.tensor(p_glen, dtype=torch.int32, device=self.device)    # shape[batch_size, 1]
            expanded_glen = torch.unsqueeze(glen, 1)
            expanded_gid = torch.unsqueeze(gid, 1)
            groups = torch.cat((groups, expanded_gid), 1)
            glens = torch.cat((glens, expanded_glen), 1)
            
            group = self.gather_nd_for_group_decode(key_val_encode_output, p_gid)
            group_mask = self.sequence_mask(glen, group.size(1), dtype=torch.float32)
            expanded_group_mask = torch.unsqueeze(group_mask, 2)
            gbow = torch.sum(group * expanded_group_mask, axis=1) / expanded_glen.float()
            
            i += 1
        
        groups = groups[:, 1:, :]
        glens = glens[:, 1:]
        
        return gbow, groups, glens, stop
    
    def group_encode(self, key_val_encode_output, groups, glens, stop):
    
        group_bow = self.gather_nd_for_group_encode(key_val_encode_output, groups)
        group_mask = self.sequence_mask(glens, groups.size(2), dtype=torch.float32)
        expanded_group_mask = torch.unsqueeze(group_mask, 3)
        #print(group_bow.size(), expanded_group_mask.size())
        group_sum_bow = torch.sum(group_bow * expanded_group_mask, axis=2)
        safe_group_lens = glens + torch.eq(glens, 0).int()
        group_mean_bow = group_sum_bow / safe_group_lens.unsqueeze(2).float()
        
        group_mean_bow = group_mean_bow.permute(1, 0, 2)
        group_encoder_output, (group_encoder_h_state, group_encoder_c_state) = self.group_encoder_rnn(group_mean_bow)
        group_encoder_embed = torch.cat((group_encoder_h_state[-2, :, :], group_encoder_h_state[-1, :, :]), 1)
        #print(group_encoder_h_state.size())
        
        return group_bow, group_mean_bow.permute(1, 0, 2), group_encoder_embed.squeeze(0)
    
    def word_decode(self, groups, glens, dec_input, group_embed, group_bow, group_mean_bow):
        
        i = 0
        input_group_concat = torch.cat((dec_input, group_embed), 1)
        tile_input_group_concat = torch.tile(input_group_concat.unsqueeze(0), (2, 1, 1))
        plan_h_state = self.plan_init_h_state_fc(tile_input_group_concat)   # shape[1, batch_size, dim]
        plan_c_state = self.plan_init_c_state_fc(tile_input_group_concat)   # shape[1, batch_size, dim]
        #plan_h_state = self.plan_init_h_state_fc(input_group_concat).unsqueeze(0)
        #plan_c_state = self.plan_init_c_state_fc(input_group_concat).unsqueeze(0)
        sent_state = torch.zeros((self.batch_size, self.config.PHVM_decoder_output_dim), dtype=torch.float32, device=self.device)
        sent_z = torch.zeros((self.batch_size, self.config.PHVM_sent_latent_dim), dtype=torch.float32, device=self.device)
        #sent_idx = torch.zeros((self.batch_size, groups.size(1), self.config.PHVM_maximum_iterations), dtype=torch.float32, device=self.device)
        group_idx_list = []
        
        while i < groups.size(1):
            gbow = group_mean_bow[:, i, :]
            sent_group = group_bow[:, i, :, :]
            sent_glen = glens[:, i]
            
            plan_input = torch.cat((sent_state, sent_z), 1).unsqueeze(0)
            sent_cond_embed, (plan_h_state, plan_c_state) = self.latent_decoder_rnn(plan_input, (plan_h_state, plan_c_state))
            
            sent_cond_embed = torch.squeeze(sent_cond_embed, 0)
            sent_prior_input = torch.cat((sent_cond_embed, gbow), 1)
            sent_prior_fc = self.prior_sent_fc_layer(sent_prior_input)
            sent_prior_mu, sent_prior_logvar = torch.split(sent_prior_fc, self.config.PHVM_sent_latent_dim, 1)
            sent_z = self.sample_gaussian((sent_prior_input.size(0), self.config.PHVM_sent_latent_dim), sent_prior_mu, sent_prior_logvar)
            sent_cond_z_embed = torch.cat((sent_cond_embed, sent_z), 1)
            
            sent_dec_state = torch.cat((sent_cond_z_embed, gbow), 1)
            sent_dec_state = sent_dec_state.unsqueeze(0)
            tile_sent_dec_state = torch.tile(sent_dec_state, (self.config.PHVM_decoder_num_layer*2, 1, 1))
            word_idx = torch.zeros((self.batch_size), dtype=torch.int32, device=self.device)
            group_idx = torch.zeros((self.batch_size, self.config.PHVM_maximum_iterations), dtype=torch.int32, device=self.device)
            word_idx_list = []
            
            sent_dec_h_state = self.sent_dec_h_state_fc(tile_sent_dec_state)
            sent_dec_c_state = self.sent_dec_c_state_fc(tile_sent_dec_state)
            
            for t in range(self.config.PHVM_maximum_iterations):
                embedded = self.word_embedding(word_idx).unsqueeze(0)
                attn_weights  = self.attention(sent_dec_h_state[0], sent_group)
                context = attn_weights.bmm(sent_group)
                rnn_input = torch.cat((embedded, context), 2)
                output, (sent_dec_h_state, sent_dec_c_state) = self.decoder_rnn(rnn_input, (sent_dec_h_state, sent_dec_c_state))
                output = output.squeeze(0)  # (1,B,N) -> (B,N)
                context = context.squeeze(0)
                output = self.word_attention_fc(torch.cat([output, context], 1))  # [32, 512] cat [32, 512] => [32, 512*2]
                output = F.log_softmax(output, dim=1)
                
                word_idx = output.data.max(1)[1]
                #group_idx[:, t] = word_idx
                word_idx_list.append(word_idx.unsqueeze(1))
                
                t += 1
            group_idx = torch.cat(word_idx_list, 1)
            #sent_idx[:, i, :] = group_idx
            group_idx_list.append(group_idx)
            sent_state = sent_dec_h_state[-1]
            
            i += 1
        sent_idx = torch.cat(group_idx_list, 1)
        
        return sent_idx[:, 1:, :]
        
    def forward(self, key_input, val_input, input_lens, target_input, target_output, output_lens, groups, glens, group_cnt, text, cate_input):
        
        key_val_embed, cate_embed, text_embed = self.input_embedding(key_input, val_input, cate_input, text)
        key_val_encode_output, key_val_encode_embed = self.input_encode(key_val_embed)
        if self.train_flag:
            self.sent_KL_loss = torch.tensor(0, dtype=torch.float, device=self.device)
            self.bow_loss = torch.tensor(0, dtype=torch.float, device=self.device)
            self.group_dec_loss = torch.tensor(0, dtype=torch.float, device=self.device)
            self.sent_dec_loss = torch.tensor(0, dtype=torch.float, device=self.device)
            
            _, tgt_embed = self.text_encode(text_embed)
            dec_input, plan_KL_weight = self.input_sample_encode(cate_embed, key_val_encode_embed, tgt_embed)
            self.sent_word_decode_train(dec_input, key_val_encode_output, input_lens, output_lens, groups, glens, group_cnt, target_input, target_output)
            self.loss_computation(plan_KL_weight)
            
            self.global_step += 1
            
            return self.train_loss
        else:
            dec_input, plan_KL_weight = self.input_sample_encode(cate_embed, key_val_encode_embed)
            gbow, groups, glens, stop = self.group_decode(dec_input, key_val_encode_output.size(1), key_val_encode_output, input_lens)
            group_bow, group_mean_bow, group_embed = self.group_encode(key_val_encode_output, groups, glens, stop)
            sent_idx = self.word_decode(groups, glens, dec_input, group_embed, group_bow, group_mean_bow)
            
            return (sent_idx, stop)
            

