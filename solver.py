from model import Encoder_cyc as Encoder
from model import Decoder_cyc as Decoder
from model import InterpLnr
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import numpy as np
import os
import time
import datetime
import pickle

from utils import pad_seq_to_2, quantize_f0_torch, quantize_f0_numpy


class Solver(object):
    """Solver for training"""

    def __init__(self, vcc_loader, config, hparams):
        """Initialize configurations."""

        # Data loader.
        self.vcc_loader = vcc_loader
        self.hparams = hparams

        # Training configurations.
        self.num_iters = config.num_iters
        self.g_lr = config.g_lr
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.resume_iters = config.resume_iters
        
        # Miscellaneous.
        self.use_tensorboard = config.use_tensorboard
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device('cuda:{}'.format(config.device_id) if self.use_cuda else 'cpu')

        # Directories.
        self.log_dir = config.log_dir
        self.sample_dir = config.sample_dir
        self.model_save_dir = config.model_save_dir

        # use demo data for simplicity
        # make your own validation set as needed
        self.validation_pt = pickle.load(open(self.hparams.demo_file, "rb"))

        # Step size.
        self.log_step = config.log_step
        self.valid_step = config.valid_step
        self.sample_step = config.sample_step
        self.model_save_step = config.model_save_step

        self.len_pad = self.hparams.max_len_pad

        # Build the model and tensorboard.
        self.build_model()
        if self.use_tensorboard:
            self.build_tensorboard()

            
    def build_model(self):
        self.E = Encoder(self.hparams)
        self.D = Decoder(self.hparams)
        
        self.Interp = InterpLnr(self.hparams)
            
        self.g_optimizer = torch.optim.Adam([{'params': self.E.parameters()}, {'params': self.D.parameters()}], self.g_lr, [self.beta1, self.beta2])
        self.print_network(self.E, 'E')
        self.print_network(self.D, 'D')

        self.E.to(self.device)
        self.D.to(self.device)
        self.Interp.to(self.device)

        
    def print_network(self, model, name):
        """Print out the network information."""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(model)
        print(name)
        print("The number of parameters: {}".format(num_params))
        
        
    def print_optimizer(self, opt, name):
        print(opt)
        print(name)
        
        
    def restore_model(self, resume_iters):
        print('Loading the trained models from step {}...'.format(resume_iters))
        G_path = os.path.join(self.model_save_dir, '{}-G.ckpt'.format(resume_iters))
        g_checkpoint = torch.load(G_path, map_location=lambda storage, loc: storage)
        self.E.load_state_dict(g_checkpoint['encoder'])
        self.D.load_state_dict(g_checkpoint['decoder'])
        self.g_optimizer.load_state_dict(g_checkpoint['optimizer'])
        self.g_lr = self.g_optimizer.param_groups[0]['lr']
        
        
    def build_tensorboard(self):
        """Build a tensorboard logger."""
        from torch.utils.tensorboard import SummaryWriter
        self.writer = SummaryWriter(self.log_dir)
        

    def reset_grad(self):
        """Reset the gradient buffers."""
        self.g_optimizer.zero_grad()
      
    
#=====================================================================================================================
    
    
                
    def train(self):
        # Set data loader.
        data_loader = self.vcc_loader
        
        # Fetch fixed inputs for debugging.
        data_iter = iter(data_loader)
        
        # Start training from scratch or resume training.
        start_iters = 0
        if self.resume_iters:
            print('Resuming ...')
            start_iters = self.resume_iters
            #self.num_iters += self.resume_iters
            self.restore_model(self.resume_iters)
            self.print_optimizer(self.g_optimizer, 'G_optimizer')
                        
        # Learning rate cache for decaying.
        g_lr = self.g_lr
        print ('Current learning rates, g_lr: {}.'.format(g_lr))
        
        # Print logs in specified order
        keys = ['G/loss', 'G/loss_x', 'G/loss_f0', 'G/loss_z']

        # Start training.
        print('Start training...')
        start_time = time.time()
        for i in range(start_iters, self.num_iters):

            # =================================================================================== #
            #                             1. Preprocess input data                                #
            # =================================================================================== #

            # Fetch real images and labels.
            try:
                x_real_org, emb_org, f0_org, len_org = next(data_iter)
            except:
                data_iter = iter(data_loader)
                x_real_org, emb_org, f0_org, len_org = next(data_iter)
            
            x_real_org = x_real_org.to(self.device)
            emb_org = emb_org.to(self.device)
            len_org = len_org.to(self.device)
            f0_org = f0_org.to(self.device)
            
                    
            # =================================================================================== #
            #                               2. Train the generator                                #
            # =================================================================================== #
            self.E = self.E.train()
            self.D = self.D.train()
                        
            # Identity mapping loss
            x_f0 = torch.cat((x_real_org, f0_org), dim=-1)
            f0_org_q = quantize_f0_torch(x_f0[:,:,-1])[0]
            x_f0_intrp = self.Interp(x_f0, len_org) 
            f0_org_intrp = quantize_f0_torch(x_f0_intrp[:,:,-1])[0]
            x_f0_intrp_org = torch.cat((x_f0_intrp[:,:,:-1], f0_org_intrp), dim=-1)

            codes_r, codes_c, codes_f = self.E(x_f0_intrp_org, x_real_org)
            x_identic, f0_identic = self.D(emb_org, codes_r, codes_c, codes_f, mode='train')

            z_select = torch.randint(0, 4, (1,))[0]
            ord_sfl = torch.randperm(x_real_org.shape[0])
            if z_select == 0:
                codes_r = codes_r[ord_sfl]
            elif z_select == 1:
                codes_c = codes_c[ord_sfl]
            elif z_select == 2:
                codes_f = codes_f[ord_sfl]
            else:
                emb_org = emb_org[ord_sfl]
            x_identic_cyc, f0_identic_cyc = self.D(emb_org, codes_r, codes_c, codes_f, mode='train')
            x_f0_cyc = torch.cat((x_identic_cyc, f0_identic_cyc), dim=-1)
            codes_cyc_r, codes_cyc_c, codes_cyc_f = self.E(x_f0_cyc, x_identic_cyc)

            codes_org = torch.cat((codes_r.reshape(x_real_org.shape[0], -1), codes_c.reshape(x_real_org.shape[0], -1), codes_f.reshape(x_real_org.shape[0], -1)), dim=-1)
            codes_cyc = torch.cat((codes_cyc_r.reshape(x_real_org.shape[0], -1), codes_cyc_c.reshape(x_real_org.shape[0], -1), codes_cyc_f.reshape(x_real_org.shape[0], -1)), dim=-1)

            g_loss_x = 0.2 * F.mse_loss(x_real_org, x_identic, reduction='mean') 
            g_loss_f0 = F.mse_loss(f0_org_q, f0_identic, reduction='mean') 
            g_loss_z = F.mse_loss(codes_org, codes_cyc, reduction='mean')

            # Backward and optimize.
            g_loss = g_loss_x + g_loss_f0 + g_loss_z
            self.reset_grad()
            g_loss.backward()
            self.g_optimizer.step()

            # Logging.
            loss = {}
            loss['G/loss'] = g_loss.item()
            loss['G/loss_x'] = g_loss_x.item()
            loss['G/loss_f0'] = g_loss_f0.item()
            loss['G/loss_z'] = g_loss_z.item()
            

            # =================================================================================== #
            #                                 4. Miscellaneous                                    #
            # =================================================================================== #

            # Print out training information.
            if (i+1) % self.log_step == 0:
                et = time.time() - start_time
                et = str(datetime.timedelta(seconds=et))[:-7]
                log = "Elapsed [{}], Iteration [{}/{}]".format(et, i+1, self.num_iters)
                for tag in keys:
                    log += ", {}: {:.8f}".format(tag, loss[tag])
                print(log)

                if self.use_tensorboard:
                    for tag, value in loss.items():
                        self.writer.add_scalar(tag, value, i+1)
                        
                        
            # Save model checkpoints.
            if (i+1) % self.model_save_step == 0:
                G_path = os.path.join(self.model_save_dir, '{}-G.ckpt'.format(i+1))
                torch.save({'encoder': self.E.state_dict(),
                            'decoder': self.D.state_dict(),
                            'optimizer': self.g_optimizer.state_dict()}, G_path)
                print('Saved model checkpoints into {}...'.format(self.model_save_dir))            
            

            # Validation.
            if (i+1) % self.valid_step == 0:
                self.E = self.E.eval()
                self.D = self.D.eval()
                with torch.no_grad():
                    loss_val = []
                    for val_sub in self.validation_pt:
                        emb_val = torch.from_numpy(val_sub[1]).to(self.device)
                        for k in range(2, 3):
                            x_real_pad, _ = pad_seq_to_2(val_sub[k][0][np.newaxis,:,:], self.len_pad)
                            len_val = torch.tensor([val_sub[k][2]]).to(self.device)
                            f0_val = np.pad(val_sub[k][1], (0, self.len_pad-val_sub[k][2]), 'constant', constant_values=(0, 0))
                            f0_quantized = quantize_f0_numpy(f0_val)[0]
                            f0_onehot = f0_quantized[np.newaxis, :, :]
                            f0_org_val = torch.from_numpy(f0_onehot).to(self.device)
                            x_real_pad = torch.from_numpy(x_real_pad).to(self.device)
                            x_f0 = torch.cat((x_real_pad, f0_org_val), dim=-1)
                            codes_val_r, codes_val_c, codes_val_f = self.E(x_f0, x_real_pad)
                            x_identic_val, f0_identic_val = self.D(emb_val, codes_val_r, codes_val_c, codes_val_f, mode='train')
                            z_select = torch.randint(0, 4, (1,))[0]

                            ord_sfl_val = torch.randperm(x_real_pad.shape[0])
                            if z_select == 0:
                                codes_val_r = codes_val_r[ord_sfl_val]
                            elif z_select == 1:
                                codes_val_c = codes_val_c[ord_sfl_val]
                            elif z_select == 2:
                                codes_val_f = codes_val_f[ord_sfl_val]
                            else:
                                emb_val = emb_val[ord_sfl_val]
                            x_identic_cyc_val, f0_identic_cyc_val = self.D(emb_val, codes_val_r, codes_val_c, codes_val_f, mode='train')
                            x_f0_cyc_val = torch.cat((x_identic_cyc_val, f0_identic_cyc_val), dim=-1)
                            codes_val_cyc_r, codes_val_cyc_c, codes_val_cyc_f = self.E(x_f0_cyc_val, x_identic_cyc_val)

                            codes_val_org = torch.cat((codes_val_r.reshape(x_real_pad.shape[0], -1), codes_val_c.reshape(x_real_pad.shape[0], -1), codes_val_f.reshape(x_real_pad.shape[0], -1)), dim=-1)
                            codes_val_cyc = torch.cat((codes_val_cyc_r.reshape(x_real_pad.shape[0], -1), codes_val_cyc_c.reshape(x_real_pad.shape[0], -1), codes_val_cyc_f.reshape(x_real_pad.shape[0], -1)), dim=-1)

                            g_loss_val_x = 0.2 * F.mse_loss(x_real_pad, x_identic_val, reduction='mean')
                            g_loss_val_f0 = F.mse_loss(f0_org_val, f0_identic_val, reduction='mean')
                            g_loss_val_z = F.mse_loss(codes_val_org, codes_val_cyc, reduction='mean')
                            g_loss_val = g_loss_val_x + g_loss_val_f0 + g_loss_val_z
                            loss_val.append(g_loss_val.item())
                val_loss = np.mean(loss_val)
                print('Validation loss: {}'.format(val_loss))
                if self.use_tensorboard:
                    self.writer.add_scalar('Validation_loss', val_loss, i+1)


            # plot test samples
            if (i+1) % self.sample_step == 0:
                self.E = self.E.eval()
                self.D = self.D.eval()
                with torch.no_grad():
                    for smp_sub in self.validation_pt:
                        emb_smp = torch.from_numpy(smp_sub[1]).to(self.device)         
                        for k in range(2, 3):
                            x_real_pad, _ = pad_seq_to_2(smp_sub[k][0][np.newaxis,:,:], self.len_pad)
                            len_smp = torch.tensor([smp_sub[k][2]]).to(self.device) 
                            f0_smp = np.pad(smp_sub[k][1], (0, self.len_pad-smp_sub[k][2]), 'constant', constant_values=(0, 0))
                            f0_quantized = quantize_f0_numpy(f0_smp)[0]
                            f0_onehot = f0_quantized[np.newaxis, :, :]
                            f0_org_smp = torch.from_numpy(f0_onehot).to(self.device) 
                            x_real_pad = torch.from_numpy(x_real_pad).to(self.device) 
                            x_f0 = torch.cat((x_real_pad, f0_org_smp), dim=-1)
                            x_f0_F = torch.cat((x_real_pad, torch.zeros_like(f0_org_smp)), dim=-1)
                            x_f0_C = torch.cat((torch.zeros_like(x_real_pad), f0_org_smp), dim=-1)
                            
                            codes_r, codes_c, codes_f = self.E(x_f0, x_real_pad)
                            x_identic_smp = self.D(emb_smp, codes_r, codes_c, codes_f, mode='test')
                            codes_r, codes_c, codes_f = self.E(x_f0_F, x_real_pad)
                            x_identic_woF = self.D(emb_smp, codes_r, codes_c, codes_f, mode='test')
                            codes_r, codes_c, codes_f = self.E(x_f0, torch.zeros_like(x_real_pad))
                            x_identic_woR = self.D(emb_smp, codes_r, codes_c, codes_f, mode='test')
                            codes_r, codes_c, codes_f = self.E(x_f0_C, x_real_pad)
                            x_identic_woC = self.D(emb_smp, codes_r, codes_c, codes_f, mode='test')

                            melsp_gd_pad = x_real_pad[0].cpu().numpy().T
                            melsp_out = x_identic_smp[0].cpu().numpy().T
                            melsp_woF = x_identic_woF[0].cpu().numpy().T
                            melsp_woR = x_identic_woR[0].cpu().numpy().T
                            melsp_woC = x_identic_woC[0].cpu().numpy().T
                            
                            min_value = np.min(np.hstack([melsp_gd_pad, melsp_out, melsp_woF, melsp_woR, melsp_woC]))
                            max_value = np.max(np.hstack([melsp_gd_pad, melsp_out, melsp_woF, melsp_woR, melsp_woC]))
                            
                            fig, (ax1,ax2,ax3,ax4,ax5) = plt.subplots(5, 1, sharex=True)
                            im1 = ax1.imshow(melsp_gd_pad, aspect='auto', vmin=min_value, vmax=max_value)
                            im2 = ax2.imshow(melsp_out, aspect='auto', vmin=min_value, vmax=max_value)
                            im3 = ax3.imshow(melsp_woC, aspect='auto', vmin=min_value, vmax=max_value)
                            im4 = ax4.imshow(melsp_woR, aspect='auto', vmin=min_value, vmax=max_value)
                            im5 = ax5.imshow(melsp_woF, aspect='auto', vmin=min_value, vmax=max_value)
                            plt.savefig(f'{self.sample_dir}/{i+1}_{smp_sub[0]}_{k}.png', dpi=150)
                            plt.close(fig) 
