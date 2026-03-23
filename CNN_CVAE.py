
import torch
import torch.nn as nn


def get_norm_layer(channels, norm_type="bn"):
    if norm_type == "bn":
        return nn.BatchNorm2d(channels, eps=1e-4)
    elif norm_type == "gn":
        return nn.GroupNorm(8, channels, eps=1e-4)
    else:
        ValueError("norm_type must be bn or gn")

class ResDown(nn.Module):
    """
    Residual down sampling block for the encoder
    """
    def __init__(self, channel_in, channel_out, kernel_size=3, norm_type="bn"):
        super(ResDown, self).__init__()   
        self.norm1 = get_norm_layer(channel_in, norm_type=norm_type)

        self.conv1 = nn.Conv2d(channel_in, (channel_out // 2) + channel_out, kernel_size, 2, kernel_size // 2)         

        self.norm2 = get_norm_layer(channel_out // 2, norm_type=norm_type)

        self.conv2 = nn.Conv2d(channel_out // 2, channel_out, kernel_size, 1, kernel_size // 2)     

        self.act_fnc = nn.ELU() 

        self.channel_out = channel_out

    def forward(self, x):    

        x = self.act_fnc(self.norm1(x))

        x_cat = self.conv1(x)

        skip = x_cat[:, :self.channel_out]

        x = x_cat[:, self.channel_out:]

        x = self.act_fnc(self.norm2(x))

        x = self.conv2(x)

        return x + skip





class ResUp(nn.Module):

    """

    Residual up sampling block for the decoder

    """



    def __init__(self, channel_in, channel_out, kernel_size=3, scale_factor=2, norm_type="bn"):

        super(ResUp, self).__init__()

        self.norm1 = get_norm_layer(channel_in, norm_type=norm_type)



        self.conv1 = nn.Conv2d(channel_in, (channel_in // 2) + channel_out, kernel_size, 1, kernel_size // 2)

        self.norm2 = get_norm_layer(channel_in // 2, norm_type=norm_type)



        self.conv2 = nn.Conv2d(channel_in // 2, channel_out, kernel_size, 1, kernel_size // 2)



        self.up_nn = nn.Upsample(scale_factor=scale_factor, mode="nearest")

        self.act_fnc = nn.ELU()

        self.channel_out = channel_out



    def forward(self, x_in):

        x = self.up_nn(self.act_fnc(self.norm1(x_in)))

        # Combine skip and first conv into one layer for speed

        x_cat = self.conv1(x)

        skip = x_cat[:, :self.channel_out]

        x = x_cat[:, self.channel_out:]

        x = self.act_fnc(self.norm2(x))

        x = self.conv2(x)

        return x + skip



class ResBlock(nn.Module):

    def __init__(self, channel_in, channel_out, kernel_size=3, norm_type="bn"):

        super(ResBlock, self).__init__()

        self.norm1 = get_norm_layer(channel_in, norm_type=norm_type)



        first_out = channel_in // 2 if channel_in == channel_out else (channel_in // 2) + channel_out

        self.conv1 = nn.Conv2d(channel_in, first_out, kernel_size, 1, kernel_size // 2)



        self.norm2 = get_norm_layer(channel_in // 2, norm_type=norm_type)



        self.conv2 = nn.Conv2d(channel_in // 2, channel_out, kernel_size, 1, kernel_size // 2)

        self.act_fnc = nn.ELU()

        self.skip = channel_in == channel_out

        self.bttl_nk = channel_in // 2



    def forward(self, x_in):

        x = self.act_fnc(self.norm1(x_in))



        x_cat = self.conv1(x)

        x = x_cat[:, :self.bttl_nk]



        # If channel_in == channel_out we do a simple identity skip

        if self.skip:

            skip = x_in

        else:

            skip = x_cat[:, self.bttl_nk:]



        x = self.act_fnc(self.norm2(x))

        x = self.conv2(x)



        return x + skip





class Encoder(nn.Module):

    def __init__(self, channels, ch=64, blocks=(1, 2, 4, 8), latent_channels=256, num_res_blocks=1, norm_type="bn",

                 deep_model=False, kernel_size=3):

        super(Encoder, self).__init__()

        self.conv_in = nn.Conv2d(channels, blocks[0]*ch,
                                 kernel_size, 1, kernel_size//2)

        widths_in = list(blocks)    # [1, 2, 4, 8]

        widths_out = list(blocks[1:]) + [2 * blocks[-1]]    

        self.layer_blocks = nn.ModuleList([])

        for w_in, w_out in zip(widths_in, widths_out):

            if deep_model:

                self.layer_blocks.append(ResBlock(w_in * ch, w_in * ch, kernel_size=kernel_size, norm_type=norm_type))

            self.layer_blocks.append(ResDown(w_in * ch, w_out * ch, kernel_size=kernel_size, norm_type=norm_type))

        for _ in range(num_res_blocks):

            self.layer_blocks.append(ResBlock(widths_out[-1] * ch, widths_out[-1] * ch, kernel_size=kernel_size, norm_type=norm_type)) 

        self.conv_mu = nn.Conv2d(widths_out[-1] * ch, latent_channels, 1, 1) # kernel_size=1, stride=1

        self.conv_log_var = nn.Conv2d(widths_out[-1] * ch, latent_channels, 1, 1)

        self.act_fnc = nn.ELU()

    def sample(self, mu, log_var):  

        std = torch.exp(0.5 * log_var)

        eps = torch.randn_like(std)   

        return mu + eps * std

    def forward(self, x, sample=False):
        x = self.conv_in(x)
        for block in self.layer_blocks:
            x = block(x)
        x = self.act_fnc(x)
        mu = self.conv_mu(x)
        log_var = self.conv_log_var(x)
        if self.training or sample:
            x = self.sample(mu, log_var)
        else:
            x = mu
        return x, mu, log_var
    

class Encoder_cond(nn.Module):

    def __init__(self, channels, ch=64, blocks=(1, 2, 4, 8), latent_channels=256, num_res_blocks=1, norm_type="bn",

                 deep_model=False, kernel_size=3, condition_dim=3):

        super(Encoder_cond, self).__init__()
        self.condition_dim = condition_dim
        self.fc_cond = nn.Linear(self.condition_dim, blocks[0]*ch)
        self.conv_in = nn.Conv2d(channels, blocks[0]*ch,
                                 kernel_size, 1, kernel_size//2)

        widths_in = list(blocks)    # [1, 2, 4, 8]

        widths_out = list(blocks[1:]) + [2 * blocks[-1]]    
      
        self.layer_blocks = nn.ModuleList([])

        for w_in, w_out in zip(widths_in, widths_out):

            if deep_model:

                self.layer_blocks.append(ResBlock(w_in * ch, w_in * ch, kernel_size=kernel_size, norm_type=norm_type))

            self.layer_blocks.append(ResDown(w_in * ch, w_out * ch, kernel_size=kernel_size, norm_type=norm_type))

        for _ in range(num_res_blocks):

            self.layer_blocks.append(ResBlock(widths_out[-1] * ch, widths_out[-1] * ch, kernel_size=kernel_size, norm_type=norm_type)) 

        self.act_fnc = nn.ELU()

    def forward(self, x, conditions):
        x = self.conv_in(x) # [B, C, H, W]
        c = self.fc_cond(conditions).unsqueeze(-1).unsqueeze(-1)  # [B, C, 1, 1]
        x = x + c # [B, C, H, W] + [B, C, 1, 1] = [B, C, H, W]
        for block in self.layer_blocks:
            x = block(x)
        x = self.act_fnc(x)
        return x

class Decoder(nn.Module):
    def __init__(self, channel_out=1, ch=64, blocks=(1, 2, 4, 8), latent_channels=256, num_res_blocks=1, norm_type="bn",
                 deep_model=False, kernel_size=3):
        super(Decoder, self).__init__()
        widths_out = list(blocks)[::-1]
        widths_in = (list(blocks[1:]) + [2 * blocks[-1]])[::-1]
        self.conv_in = nn.Conv2d(latent_channels, widths_in[0] * ch, 1, 1)
        self.layer_blocks = nn.ModuleList([])
        for _ in range(num_res_blocks):
            self.layer_blocks.append(ResBlock(widths_in[0] * ch, widths_in[0] * ch, kernel_size=kernel_size, norm_type=norm_type))
        for w_in, w_out in zip(widths_in, widths_out):
            self.layer_blocks.append(ResUp(w_in * ch, w_out * ch, norm_type=norm_type))
            if deep_model:
                # Add an additional non up-sampling block after up-sampling
                self.layer_blocks.append(ResBlock(w_out * ch, w_out * ch, kernel_size=kernel_size, norm_type=norm_type))
        self.conv_out = nn.Conv2d(blocks[0] * ch, channel_out, 5, 1, 2)
        self.act_fnc = nn.ELU()

    def forward(self, x):
        x = self.conv_in(x)
        for block in self.layer_blocks:
            x = block(x)
        x = self.act_fnc(x)
        return self.conv_out(x)

class VAE(nn.Module):

    def __init__(self, channel_in=3, ch=64, blocks=(1, 2, 4, 8), latent_channels=256, num_res_blocks=1, 
                 norm_type="bn", deep_model=False):
        super(VAE, self).__init__()
        self.encoder = Encoder(channel_in, ch=ch, blocks=blocks, latent_channels=latent_channels,

                               num_res_blocks=num_res_blocks, norm_type=norm_type, deep_model=deep_model)
        self.decoder = Decoder(channel_in, ch=ch, blocks=blocks, latent_channels=latent_channels,

                               num_res_blocks=num_res_blocks, norm_type=norm_type, deep_model=deep_model)
        self.latent_channels = latent_channels

    def forward(self, x):
        encoding, mu, log_var = self.encoder(x)  # x: [1, C, H, W] → mu/log_var: [1, C', h, w]
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + eps * std  # [1, latent_dim, h, w]
        recon_image = self.decoder(z)  # [1, C, H, W]
        return recon_image, mu, log_var


class CVAE_encoder(nn.Module):
    def __init__(self, channel_in=3, ch=64, blocks=(1, 2, 4, 8), latent_channels=256, num_res_blocks=1, 
                 norm_type="bn", deep_model=False, condition_dim=3, kernel_size=3):
        super(CVAE_encoder, self).__init__()
        self.channel_in = channel_in    
        
        self.encoder = Encoder(channel_in + condition_dim, ch=ch, blocks=blocks,
                            latent_channels=latent_channels, num_res_blocks=num_res_blocks,
                            norm_type=norm_type, deep_model=deep_model, kernel_size=kernel_size)

        self.decoder = Decoder(channel_in, ch=ch, blocks=blocks, latent_channels=latent_channels + condition_dim,
                               num_res_blocks=num_res_blocks, norm_type=norm_type, deep_model=deep_model, kernel_size=kernel_size)
        
        self.latent_channels = latent_channels
        self.condition_dim = condition_dim

    def forward(self, x, conditions):
        """
        Forward pass for the CVAE.
        
        Args:
        - x: Input image tensor of shape [batch_size, channels, height, width].
        - conditions: Minkowski Functionals (MFs) tensor of shape [batch_size, condition_dim].

        Returns:
        - recon_img: Reconstructed image tensor.
        - mu: Mean of the latent representation.
        - log_var: Log variance of the latent representation.
        """
        cond_map = conditions.unsqueeze(2).unsqueeze(3)           
        cond_map = cond_map.expand(-1, -1, x.size(2), x.size(3))  
        x_cond = torch.cat([x, cond_map], dim=1)
        # Encode image to latent representation
        _, mu, log_var = self.encoder(x_cond) # the shape of mu and log_var is [B, C_lat, H, W]        
        # Sample from latent space
        std = torch.exp(0.5 * log_var) # the shape of std is [B, C_lat, H, W]
        eps = torch.randn_like(std) # eps is a random noise tensor with the same shape as std
        z = mu + eps * std  # Reparameterization trick

        # Concatenate latent representation with conditions
        conditions_expanded = conditions.unsqueeze(2).unsqueeze(3)  # Expand dims for broadcasting
        conditions_expanded = conditions_expanded.expand(-1, -1, z.size(2), z.size(3))  # Match spatial dims
        z_cond = torch.cat([z, conditions_expanded], dim=1)

        # Decode conditioned latent representation to reconstruct image
        recon_img = self.decoder(z_cond)
        return recon_img, mu, log_var


class CVAE_Flat(nn.Module):
    def __init__(self, channel_in=1, condition_dim=3, ch=64, blocks=(1, 2, 4),
                 latent_channels=23, hw=(5, 5), num_res_blocks=1,
                 norm_type="bn", deep_model=False, kernel_size=3):
        super().__init__()

        # ---------- Encoder ----------
        self.encoder = Encoder(channel_in + condition_dim, ch=ch, blocks=blocks,
                               latent_channels=latent_channels, num_res_blocks=num_res_blocks,
                               norm_type=norm_type, deep_model=deep_model, kernel_size=kernel_size)

        self.H, self.W = hw
        self.latent_channels = latent_channels
        self.z_dim = latent_channels * self.H * self.W

        self.fc_mu      = nn.Linear(self.z_dim, self.z_dim)
        self.fc_log_var = nn.Linear(self.z_dim, self.z_dim)

        # ---------- Decoder ----------
        self.fc_dec = nn.Linear(self.z_dim + condition_dim, self.z_dim)
        self.decoder = Decoder(channel_in, ch=ch, blocks=blocks, latent_channels=latent_channels,
                               num_res_blocks=num_res_blocks, norm_type=norm_type,
                               deep_model=deep_model, kernel_size=kernel_size)

    @staticmethod
    def _sample(mu, log_var):
        std = (0.5 * log_var).exp()
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, cond):
        """
        x    : [B, C_img, H, W]
        cond : [B, condition_dim]
        """
        # 1) broadcast condition → feature-map
        cond_map = cond.unsqueeze(2).unsqueeze(3)           # [B, c_dim, 1, 1]
        cond_map = cond_map.expand(-1, -1, x.size(2), x.size(3))  # [B, c_dim, H, W]
        x_cond = torch.cat([x, cond_map], dim=1)            # [B, C_img+c_dim, H, W]

        # 2) Encoder
        h, mu_map, log_var_map = self.encoder(x_cond)       # [B, C_lat, 5, 5]
        z_flat_in = h.view(x.size(0), -1)                   # flatten

        # 3) Reparameterize
        mu      = self.fc_mu(z_flat_in)
        log_var = self.fc_log_var(z_flat_in)
        z       = self._sample(mu, log_var)

        z_cond = torch.cat([z, cond], dim=1)                # [B, z_dim + c_dim]

        # 4) fc_dec → reshape → Decoder
        z_map = self.fc_dec(z_cond).view(
            x.size(0), self.latent_channels, self.H, self.W
        )
        recon = self.decoder(z_map)
        return recon, mu, log_var


class CVAE_Flat_consistent(nn.Module):
    def __init__(self, channel_in=1, condition_dim=3, ch=64, blocks=(1, 2, 4),
                 latent_channels=23, hw=(5, 5), num_res_blocks=1,
                 norm_type="bn", deep_model=False, kernel_size=3):
        super().__init__()
        self.encoder = Encoder_cond(channel_in, ch=ch, blocks=blocks,
                               latent_channels=latent_channels, num_res_blocks=num_res_blocks,
                               norm_type=norm_type, deep_model=deep_model, kernel_size=kernel_size, condition_dim=condition_dim)

        self.H, self.W = hw
        self.latent_channels = latent_channels
        self.z_dim = latent_channels * self.H * self.W

        enc_out_ch = (2 * blocks[-1]) * ch    
        self.enc_feat_dim = enc_out_ch * self.H * self.W

        self.fc_mu      = nn.Linear(self.enc_feat_dim, self.z_dim)
        self.fc_log_var = nn.Linear(self.enc_feat_dim, self.z_dim)

        # ---------- Decoder ----------
        self.fc_dec = nn.Linear(self.z_dim + condition_dim, self.z_dim)
        self.decoder = Decoder(channel_in, ch=ch, blocks=blocks, latent_channels=latent_channels,
                               num_res_blocks=num_res_blocks, norm_type=norm_type,
                               deep_model=deep_model, kernel_size=kernel_size)

    @staticmethod
    def _sample(mu, log_var):
        std = (0.5 * log_var).exp()
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, cond):
        """
        x    : [B, C_img, H, W]
        cond : [B, condition_dim]
        """
        # Encoder
        h= self.encoder(x, cond)       # [B, C_lat, 5, 5]
        z_flat_in = h.view(x.size(0), -1)                   # flatten

        # Reparameterize
        mu      = self.fc_mu(z_flat_in)
        log_var = self.fc_log_var(z_flat_in)
        z       = self._sample(mu, log_var)

        z_cond = torch.cat([z, cond], dim=1)                # [B, z_dim + c_dim]

        # fc_dec → reshape → Decoder
        z_map = self.fc_dec(z_cond).view(
            x.size(0), self.latent_channels, self.H, self.W
        )
        recon = self.decoder(z_map)
        return recon, mu, log_var