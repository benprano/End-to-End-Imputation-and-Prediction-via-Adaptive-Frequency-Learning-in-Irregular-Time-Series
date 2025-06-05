from typing import List
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor


class DyFAIP_Cell(torch.jit.ScriptModule):
    def __init__(self, input_size, hidden_size, seq_len, output_dim, batch_first=True, bidirectional=True):
        super(DyFAIP_Cell, self).__init__()
        self.input_size = input_size
        self.output_dim = output_dim
        self.initializer_range = 0.02
        self.hidden_size = hidden_size
        self.seq_len = seq_len
        self.batch_first = batch_first
        self.bidirectional = bidirectional
        self.c1 = torch.Tensor([1]).float()
        self.c2 = torch.Tensor([np.e]).float()
        self.ones = torch.ones([self.input_size, 1, self.hidden_size]).float()
        self.decay_features = torch.Tensor(torch.arange(self.input_size)).float()
        self.register_buffer('c1_const', self.c1)
        self.register_buffer('c2_const', self.c2)
        self.register_buffer("ones_const", self.ones)
        self.alpha = torch.FloatTensor([0.5])
        self.imp_weight = torch.FloatTensor([0.05])
        self.alpha_imp = torch.FloatTensor([0.5])
        self.register_buffer("factor", self.alpha)
        self.register_buffer("imp_weight_freq", self.imp_weight)
        self.register_buffer("features_decay", self.decay_features)
        self.register_buffer("factor_impu", self.alpha_imp)

        self.U_j = nn.Parameter(torch.normal(0.0, self.initializer_range, size=(self.input_size, 1, self.hidden_size)))
        self.U_i = nn.Parameter(torch.normal(0.0, self.initializer_range, size=(self.input_size, 1, self.hidden_size)))
        self.U_f = nn.Parameter(torch.normal(0.0, self.initializer_range, size=(self.input_size, 1, self.hidden_size)))
        self.U_o = nn.Parameter(torch.normal(0.0, self.initializer_range, size=(self.input_size, 1, self.hidden_size)))
        self.U_c = nn.Parameter(torch.normal(0.0, self.initializer_range, size=(self.input_size, 1, self.hidden_size)))
        self.U_last = nn.Parameter(torch.normal(0.0, self.initializer_range, size=(self.input_size, 1, self.hidden_size)))
        self.U_time = nn.Parameter(torch.normal(0.0, self.initializer_range, size=(self.input_size, 1, self.hidden_size)))
        self.Dw = nn.Parameter(torch.normal(0.0, self.initializer_range, size=(self.input_size, 1, self.hidden_size)))

        self.W_j = nn.Parameter(torch.normal(0.0, self.initializer_range, size=(self.input_size, self.hidden_size, self.hidden_size)))
        self.W_i = nn.Parameter(torch.normal(0.0, self.initializer_range, size=(self.input_size, self.hidden_size, self.hidden_size)))
        self.W_f = nn.Parameter(torch.normal(0.0, self.initializer_range, size=(self.input_size, self.hidden_size, self.hidden_size)))
        self.W_o = nn.Parameter(torch.normal(0.0, self.initializer_range, size=(self.input_size, self.hidden_size, self.hidden_size)))
        self.W_c = nn.Parameter(torch.normal(0.0, self.initializer_range, size=(self.input_size, self.hidden_size, self.hidden_size)))
        self.W_d = nn.Parameter(torch.normal(0.0, self.initializer_range, size=(self.input_size, self.hidden_size, self.hidden_size)))
        self.W_decomp = nn.Parameter(torch.normal(0.0, self.initializer_range, size=(self.input_size, self.hidden_size, self.hidden_size)))
        self.W_cell_i = nn.Parameter(torch.normal(0.0, self.initializer_range, size=(self.input_size, self.hidden_size)))
        self.W_cell_f = nn.Parameter(torch.normal(0.0, self.initializer_range, size=(self.input_size, self.hidden_size)))
        self.W_cell_o = nn.Parameter(torch.normal(0.0, self.initializer_range, size=(self.input_size, self.hidden_size)))

        self.b_decomp = nn.Parameter(torch.normal(0.0, self.initializer_range, size=(self.input_size, self.hidden_size)))
        self.b_j = nn.Parameter(torch.normal(0.0, self.initializer_range, size=(self.input_size, self.hidden_size)))
        self.b_i = nn.Parameter(torch.normal(0.0, self.initializer_range, size=(self.input_size, self.hidden_size)))
        self.b_f = nn.Parameter(torch.normal(0.0, self.initializer_range, size=(self.input_size, self.hidden_size)))
        self.b_o = nn.Parameter(torch.normal(0.0, self.initializer_range, size=(self.input_size, self.hidden_size)))
        self.b_c = nn.Parameter(torch.normal(0.0, self.initializer_range, size=(self.input_size, self.hidden_size)))
        self.b_last = nn.Parameter(torch.normal(0.0, self.initializer_range, size=(self.input_size, self.hidden_size)))
        self.b_time = nn.Parameter(torch.normal(0.0, self.initializer_range, size=(self.input_size, self.hidden_size)))
        self.b_d = nn.Parameter(torch.normal(0.0, self.initializer_range, size=(self.input_size, self.hidden_size)))
        # Interpolation
        self.W_ht_mask = nn.Parameter(torch.normal(0.0, self.initializer_range, size=(self.input_size, self.hidden_size, 1)))
        self.W_ct_mask = nn.Parameter(torch.normal(0.0, self.initializer_range, size=(self.input_size, self.hidden_size, 1)))
        self.b_j_mask = nn.Parameter(torch.normal(0.0, self.initializer_range, size=(self.input_size, self.hidden_size)))

        self.W_ht_last = nn.Parameter(torch.normal(0.0, self.initializer_range, size=(self.input_size, self.hidden_size, 1)))
        self.W_ct_last = nn.Parameter(torch.normal(0.0, self.initializer_range, size=(self.input_size, self.hidden_size, 1)))
        self.b_j_last = nn.Parameter(torch.normal(0.0, self.initializer_range, size=(self.input_size, self.hidden_size)))

        # Gate Linear Unit for last records
        self.activation_layer = nn.ELU()
        self.F_alpha = nn.Parameter(torch.normal(0.0, self.initializer_range, size=(self.input_size, self.hidden_size * 2, 1)))
        self.F_alpha_n_b = nn.Parameter(torch.normal(0.0, self.initializer_range, size=(self.input_size, 1)))
        self.F_beta = nn.Linear(self.seq_len, 4 * self.hidden_size)
        self.layer_norm1 = nn.LayerNorm([self.input_size, self.seq_len])
        self.layer_norm = nn.LayerNorm([self.input_size, 4 * self.hidden_size])
        self.Phi = nn.Linear(4 * self.hidden_size, self.output_dim)

    @torch.jit.script_method
    def dyfaip_unit(self, prev_hidden_memory, cell_hidden_memory, inputs, times, last_data, freq_list):
        h_tilda_t, c_tilda_t = prev_hidden_memory, cell_hidden_memory,
        x = inputs
        t = times
        l = last_data
        freq = freq_list
        # frequency weights for imputation of missing data based on frequencies of features
        # Imputation gate for inputs x last records
        x_last_hidden = torch.tanh(torch.einsum("bij,ijk->bik", self.freq_decay(freq, h_tilda_t), self.W_ht_last) + \
                                   torch.einsum("bij,ijk->bik", self.freq_decay(freq, c_tilda_t),
                                                self.W_ct_last) + self.b_j_last).permute(0, 2, 1)

        imputat_imputs = torch.tanh(torch.einsum("bij,ijk->bik", self.freq_decay(freq, h_tilda_t), self.W_ht_mask) + \
                                    torch.einsum("bij,ijk->bik", self.freq_decay(freq, c_tilda_t), self.W_ct_mask) + \
                                    self.b_j_mask).permute(0, 2, 1)
        
        # Replace nan data with the impuated value generated from DyFAIP memory from the frequency weighting
        _, x_last = self.impute_missing_data(l, freq, x_last_hidden)
        all_imputed_x, imputed_x = self.impute_missing_data(x, freq, imputat_imputs)

        # Ajust previous to incoporate the latest records for each feature
        last_tilda_t = self.activation_layer(torch.einsum("bij,jik->bjk", x_last, self.U_last) + self.b_last)
        h_tilda_t = h_tilda_t + last_tilda_t
        # Capturing Temporal Dependencies wrt to the previous hidden state
        j_tilda_t = torch.tanh(torch.einsum("bij,ijk->bik", h_tilda_t, self.W_j) + \
                               torch.einsum("bij,jik->bjk", imputed_x, self.U_j) + self.b_j)

        # Time Gate
        t_gate = torch.sigmoid(torch.einsum("bij,jik->bjk", imputed_x, self.U_time) +
                               torch.sigmoid(self.map_elapse_time(t)) + self.b_time)
        # Input Gate
        i = torch.sigmoid(torch.einsum("bij,jik->bjk", imputed_x, self.U_i) + \
                          torch.einsum("bij,ijk->bik", h_tilda_t, self.W_i) + \
                          c_tilda_t * self.W_cell_i + self.b_i * self.freq_decay(freq, j_tilda_t))
        # Forget Gate
        f = torch.sigmoid(torch.einsum("bij,jik->bjk", imputed_x, self.U_f) + \
                          torch.einsum("bij,ijk->bik", h_tilda_t, self.W_f) + \
                          c_tilda_t * self.W_cell_f)

        f_new = f * self.map_elapse_time(t) + (1 - f) * self.freq_decay(freq, j_tilda_t)
        # Candidate Memory Cell
        c = torch.tanh(torch.einsum("bij,jik->bjk", imputed_x, self.U_c) + \
                       torch.einsum("bij,ijk->bik", h_tilda_t, self.W_c) + self.b_c)
        # Current Memory Cell
        ct = (f_new + t_gate) * c_tilda_t + i * j_tilda_t * t_gate * c
        # Output Gate
        o = torch.sigmoid(torch.einsum("bij,jik->bjk", imputed_x, self.U_o) +
                          torch.einsum("bij,ijk->bik", h_tilda_t, self.W_o) +
                          ct * self.W_cell_o + self.b_o)
        # Current Hidden State
        h_tilda_t = o * torch.tanh(ct  + last_tilda_t)

        return h_tilda_t, ct, self.freq_decay(freq, j_tilda_t), f_new, all_imputed_x

    @torch.jit.script_method
    def impute_missing_data(self, x: torch.Tensor, freq_dict: torch.Tensor, x_hidden: torch.Tensor):
        # Calculate feature factor
        factor_feature = torch.div(
            torch.exp(-self.imp_weight_freq * freq_dict),
            torch.exp(-self.imp_weight_freq * freq_dict).max()).unsqueeze(1)

        # Calculate imputation factor
        factor_imp = torch.div(
            torch.exp(self.factor_impu * freq_dict),
            torch.exp(self.factor_impu * freq_dict).max()).unsqueeze(1)

        # Adjust frequencies
        frequencies = (self.seq_len - freq_dict) * torch.exp(-self.factor * self.features_decay)
        frequencies = torch.div(frequencies, frequencies.max()).unsqueeze(-1)
        
        # Compute imputed values %factor_imp >= threshold, or any value from grid search
        # any value from 0.93 to 0.99 give the similar results as more than 90% features are selected based on this threshold settings.
        # (0.99 ---> select almost all the features to be used)
        omega = 0.93  
        threshold = omega * factor_imp.max()

        # Compute imputed values
        imputed_missed_x = torch.where(
            factor_imp >= threshold,
            frequencies.permute(0, 2, 1) * x_hidden,
            factor_feature * x_hidden
        )

        # Replace missing values
        x_imputed = torch.where(torch.isnan(x.unsqueeze(1)), imputed_missed_x, x.unsqueeze(1))

        return imputed_missed_x, x_imputed

    @torch.jit.script_method
    def map_elapse_time(self, t):
        T = torch.div(self.c1_const, torch.log(t + self.c2_const))
        T = torch.einsum("bij,jik->bjk", T.unsqueeze(1), self.ones_const)
        return T

    @torch.jit.script_method
    def freq_decay(self, freq_dict: torch.Tensor, ht: torch.Tensor):
        freq_weight = torch.exp(-self.factor_impu * freq_dict)
        weights = torch.sigmoid(torch.einsum("bij,jik->bjk", freq_weight.unsqueeze(-1), self.Dw) + \
                                torch.einsum("bij,ijk->bik", ht, self.W_d) + self.b_d)
        return weights

    @torch.jit.script_method
    def forward(self, inputs, times, last_values, freqs):
        device = inputs.device
        if self.batch_first:
            batch_size = inputs.size()[0]
            inputs = inputs.permute(1, 0, 2)
            last_values = last_values.permute(1, 0, 2)
            freqs = freqs.permute(1, 0, 2)
            times = times.transpose(0, 1)
        else:
            batch_size = inputs.size()[1]
        prev_hidden = torch.zeros((batch_size, inputs.size()[2], self.hidden_size), device=device)
        prev_cell = torch.zeros((batch_size, inputs.size()[2], self.hidden_size), device=device)

        seq_len = inputs.size()[0]
        imputed_inputs = torch.jit.annotate(List[Tensor], [])
        hidden_his = torch.jit.annotate(List[Tensor], [])
        weights_decay = torch.jit.annotate(List[Tensor], [])
        weights_fgate = torch.jit.annotate(List[Tensor], [])
        for i in range(seq_len):
            prev_hidden, prev_cell, pre_we_decay, fgate_f, imputed_x = self.dyfaip_unit(prev_hidden, prev_cell, inputs[i],
                                                                                        times[i], last_values[i],
                                                                                        freqs[i])
            hidden_his += [prev_hidden]
            imputed_inputs += [imputed_x]
            weights_decay += [pre_we_decay]
            weights_fgate += [fgate_f]
        imputed_inputs = torch.stack(imputed_inputs)
        hidden_his = torch.stack(hidden_his)
        weights_decay = torch.stack(weights_decay)
        weights_fgate = torch.stack(weights_fgate)
        if self.bidirectional:
            second_hidden = torch.zeros((batch_size, inputs.size()[2], self.hidden_size), device=device)
            second_cell = torch.zeros((batch_size, inputs.size()[2], self.hidden_size), device=device)
            second_inputs = torch.flip(inputs, [0])
            second_times = torch.flip(times, [0])
            imputed_inputs_b = torch.jit.annotate(List[Tensor], [])
            second_hidden_his = torch.jit.annotate(List[Tensor], [])
            second_weights_decay = torch.jit.annotate(List[Tensor], [])
            second_weights_fgate = torch.jit.annotate(List[Tensor], [])
            for i in range(seq_len):
                if i == 0:
                    time = times[i]
                else:
                    time = second_times[i - 1]
                second_hidden, second_cell, b_we_decay, fgate_b, imputed_x_b = self.dyfaip_unit(second_hidden,second_cell,
                                                                                                second_inputs[i], time,
                                                                                                last_values[i], freqs[i])
                second_hidden_his += [second_hidden]
                second_weights_decay += [b_we_decay]
                second_weights_fgate += [fgate_b]
                imputed_inputs_b += [imputed_x_b]

            imputed_inputs_b = torch.stack(imputed_inputs_b)
            second_hidden_his = torch.stack(second_hidden_his)
            second_weights_fgate = torch.stack(second_weights_fgate)
            second_weights_decay = torch.stack(second_weights_decay)
            weights_decay = torch.cat((weights_decay, second_weights_decay), dim=-1)
            weights_fgate = torch.cat((weights_fgate, second_weights_fgate), dim=-1)
            hidden_his = torch.cat((hidden_his, second_hidden_his), dim=-1)
            imputed_inputs = torch.cat((imputed_inputs, imputed_inputs_b), dim=2)
            prev_hidden = torch.cat((prev_hidden, second_hidden), dim=-1)
            prev_cell = torch.cat((prev_cell, second_cell), dim=-1)
        if self.batch_first:
            hidden_his = hidden_his.permute(1, 0, 2, 3)
            imputed_inputs = imputed_inputs.permute(1, 0, 2, 3)
            weights_decay = weights_decay.permute(1, 0, 2, 3)
            weights_fgate = weights_fgate.permute(1, 0, 2, 3)

        alphas = torch.tanh(torch.einsum("btij,ijk->btik", hidden_his, self.F_alpha) + self.F_alpha_n_b)
        alphas = alphas.reshape(alphas.size(0), alphas.size(2), alphas.size(1) * alphas.size(-1))
        mu = self.Phi(self.layer_norm(self.F_beta(self.layer_norm1(alphas))))
        out = torch.max(mu, dim=1).values
        return out, weights_decay, weights_fgate, imputed_inputs

class DyFAIP_Aware(nn.Module):
    def __init__(self, input_dim, hidden_dim,seq_len, output_dim, dropout=0.2):
        super(DyFAIP_Aware, self).__init__()
        # hidden dimensions
        self.input_size = input_dim
        self.hidden_size = hidden_dim
        self.seq_len = seq_len
        self.output_dim= output_dim
        # Temporal embedding DyFAIP
        self.dyfaip = DyFAIP_Cell(self.input_size, self.hidden_size,
                                          self.seq_len, self.output_dim)
    def forward(self,historic_features,timestamp, last_features, features_freqs , is_test=False):
        # Temporal features embedding
        outputs, decay_weights, fgate, imputed_inputs = self.dyfaip(historic_features,timestamp,
                                                                    last_features, features_freqs)
        if is_test:
            return decay_weights, fgate, imputed_inputs.mean(axis=2), outputs
        else:
            return outputs
