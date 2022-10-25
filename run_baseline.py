# DUMP ALL SCRIPTS!

from models.rnn import RNN_BiGRU, RNN_BiLSTM, RNN_Config, default_bigru_config, default_bilstm_config


bilstm_model = RNN_BiLSTM(default_bilstm_config)
bigru_model = RNN_BiLSTM(default_bilstm_config)