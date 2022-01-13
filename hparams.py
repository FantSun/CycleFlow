from tfcompat.hparam import HParams

# NOTE: If you want full control for model architecture. please take a look
# at the code and change whatever you want. Some hyper parameters are hardcoded.

# Default hyperparameters:
hparams = HParams(
    # model   
    freq_c = 8,
    dim_neck_c = 8,
    freq_r = 8,
    dim_neck_r = 1,
    freq_f = 8,
    dim_neck_f = 32,
    freq_t = 192,
    
    dim_enc_c = 512,
    dim_enc_r = 128,
    dim_enc_f = 256,
    
    dim_freq = 80,
    dim_spk_emb = 32,
    dim_f0 = 257,
    dim_dec = 512,
    len_raw = 128,
    chs_grp = 16,
    
    # interp
    min_len_seg = 19,
    max_len_seg = 32,
    min_len_seq = 64,
    max_len_seq = 128,
    max_len_pad = 192,
    
    # data loader
    root_dir = 'data/training_set/spmel',
    feat_dir = 'data/training_set/raptf0',
    dvec_dir = 'data/training_set/dvec',
    demo_file = 'data/valid_set/demo.pkl',
    batch_size = 64,
    mode = 'train',
    shuffle = True,
    num_workers = 0,
    samplier = 8,
    
)


def hparams_debug_string():
    values = hparams.values()
    hp = ['  %s: %s' % (name, values[name]) for name in values]
    return 'Hyperparameters:\n' + '\n'.join(hp)
