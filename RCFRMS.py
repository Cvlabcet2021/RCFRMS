"""
## Residual Cell Feature Refined Deep Multi-Scale Network
## Chippy M. Manu and Sreeni K. G. 

import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as stx

##########################################################################
def conv(in_channels, out_channels, kernel_size, bias=False, stride = 1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias, stride = stride)


##########################################################################

class DRCLayer(nn.Module):
    def __init__(self, channel, reduction=16, bias=False):
        super(DRCLayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


##########################################################################
## Contextual Feature Module (CFM)
class CFM(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, bias, act):
        super(CFM, self).__init__()
        modules_body = []
        modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
        modules_body.append(act)
        modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))

        self.DRC = DRCLayer(n_feat, reduction, bias=bias)
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res = self.DRC(res)
        res += x
        return res

##########################################################################
## Supervised Attention Module
class SAM(nn.Module):
    def __init__(self, n_feat, kernel_size, bias):
        super(SAM, self).__init__()
        self.conv1 = conv(n_feat, n_feat, kernel_size, bias=bias)
        self.conv2 = conv(n_feat, 3, kernel_size, bias=bias)
        self.conv3 = conv(3, n_feat, kernel_size, bias=bias)

    def forward(self, x, x_img):
        x1 = self.conv1(x)
        img = self.conv2(x) + x_img
        x2 = torch.sigmoid(self.conv3(img))
        x1 = x1*x2
        x1 = x1+x
        return x1, img

##########################################################################
## U-Net

class Encoder(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, act, bias, scale_unetfeats, csff):
        super(Encoder, self).__init__()

        self.encoder_level1 = [CFM(n_feat,                     kernel_size, reduction, bias=bias, act=act) for _ in range(2)]
        self.encoder_level2 = [CFM(n_feat+scale_unetfeats,     kernel_size, reduction, bias=bias, act=act) for _ in range(2)]
        self.encoder_level3 = [CFM(n_feat+(scale_unetfeats*2), kernel_size, reduction, bias=bias, act=act) for _ in range(2)]

        self.encoder_level1 = nn.Sequential(*self.encoder_level1)
        self.encoder_level2 = nn.Sequential(*self.encoder_level2)
        self.encoder_level3 = nn.Sequential(*self.encoder_level3)

        self.down12  = DownSample(n_feat, scale_unetfeats)
        self.down23  = DownSample(n_feat+scale_unetfeats, scale_unetfeats)

        # Cross Stage Feature Fusion (CSFF)
        if csff:
            self.csff_enc1 = nn.Conv2d(n_feat,                     n_feat,                     kernel_size=1, bias=bias)
            self.csff_enc2 = nn.Conv2d(n_feat+scale_unetfeats,     n_feat+scale_unetfeats,     kernel_size=1, bias=bias)
            self.csff_enc3 = nn.Conv2d(n_feat+(scale_unetfeats*2), n_feat+(scale_unetfeats*2), kernel_size=1, bias=bias)

            self.csff_dec1 = nn.Conv2d(n_feat,                     n_feat,                     kernel_size=1, bias=bias)
            self.csff_dec2 = nn.Conv2d(n_feat+scale_unetfeats,     n_feat+scale_unetfeats,     kernel_size=1, bias=bias)
            self.csff_dec3 = nn.Conv2d(n_feat+(scale_unetfeats*2), n_feat+(scale_unetfeats*2), kernel_size=1, bias=bias)

    def forward(self, x, encoder_outs=None, decoder_outs=None):
        enc1 = self.encoder_level1(x)
        if (encoder_outs is not None) and (decoder_outs is not None):
            enc1 = enc1 + self.csff_enc1(encoder_outs[0]) + self.csff_dec1(decoder_outs[0])

        x = self.down12(enc1)

        enc2 = self.encoder_level2(x)
        if (encoder_outs is not None) and (decoder_outs is not None):
            enc2 = enc2 + self.csff_enc2(encoder_outs[1]) + self.csff_dec2(decoder_outs[1])

        x = self.down23(enc2)

        enc3 = self.encoder_level3(x)
        if (encoder_outs is not None) and (decoder_outs is not None):
            enc3 = enc3 + self.csff_enc3(encoder_outs[2]) + self.csff_dec3(decoder_outs[2])
        
        return [enc1, enc2, enc3]

class Decoder(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, act, bias, scale_unetfeats):
        super(Decoder, self).__init__()

        self.decoder_level1 = [CFM(n_feat,                     kernel_size, reduction, bias=bias, act=act) for _ in range(2)]
        self.decoder_level2 = [CFM(n_feat+scale_unetfeats,     kernel_size, reduction, bias=bias, act=act) for _ in range(2)]
        self.decoder_level3 = [CFM(n_feat+(scale_unetfeats*2), kernel_size, reduction, bias=bias, act=act) for _ in range(2)]

        self.decoder_level1 = nn.Sequential(*self.decoder_level1)
        self.decoder_level2 = nn.Sequential(*self.decoder_level2)
        self.decoder_level3 = nn.Sequential(*self.decoder_level3)

        self.skip_attn1 = CFM(n_feat,                 kernel_size, reduction, bias=bias, act=act)
        self.skip_attn2 = CFM(n_feat+scale_unetfeats, kernel_size, reduction, bias=bias, act=act)

        self.up21  = SkipUpSample(n_feat, scale_unetfeats)
        self.up32  = SkipUpSample(n_feat+scale_unetfeats, scale_unetfeats)

    def forward(self, outs):
        enc1, enc2, enc3 = outs
        dec3 = self.decoder_level3(enc3)

        x = self.up32(dec3, self.skip_attn2(enc2))
        dec2 = self.decoder_level2(x)

        x = self.up21(dec2, self.skip_attn1(enc1))
        dec1 = self.decoder_level1(x)

        return [dec1,dec2,dec3]

##########################################################################
##---------- Resizing Modules ----------    
class DownSample(nn.Module):
    def __init__(self, in_channels,s_factor):
        super(DownSample, self).__init__()
        self.down = nn.Sequential(nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=False),
                                  nn.Conv2d(in_channels, in_channels+s_factor, 1, stride=1, padding=0, bias=False))

    def forward(self, x):
        x = self.down(x)
        return x

class UpSample(nn.Module):
    def __init__(self, in_channels,s_factor):
        super(UpSample, self).__init__()
        self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                                nn.Conv2d(in_channels+s_factor, in_channels, 1, stride=1, padding=0, bias=False))

    def forward(self, x):
        x = self.up(x)
        return x

class SkipUpSample(nn.Module):
    def __init__(self, in_channels,s_factor):
        super(SkipUpSample, self).__init__()
        self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                                nn.Conv2d(in_channels+s_factor, in_channels, 1, stride=1, padding=0, bias=False))

    def forward(self, x, y):
        x = self.up(x)
        x = x + y
        return x

##########################################################################
##  (DRC)
class ARDN(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, act, bias, num_cfm):
        super(ARDN, self).__init__()
        modules_body = []
        modules_body1 = []
        modules_body2 = []
        modules_body3 = []
        modules_body4 = []
        modules_body5 = []
        modules_body1 = [DRC(n_feat, kernel_size, reduction, bias=bias, act=act) for _ in range(num_drc)]
        modules_body2 = [MAB(n_feat, kernel_size, reduction, bias=bias, act=act) for _ in range(num_mab)]
        modules_body3 = [DRC(n_feat, kernel_size, reduction, bias=bias, act=act) for _ in range(num_drc)]
        modules_body4 = [DFN(n_feat, kernel_size, reduction, bias=bias, act=act) for _ in range(num_dfn)]
        modules_body5 = [DRC(n_feat, kernel_size, reduction, bias=bias, act=act) for _ in range(num_drc)]
        modules_body = [ESIN(n_feat, kernel_size, reduction, bias=bias, act=act) for _ in range(num_esin)]

        modules_body.append(conv(n_feat, n_feat, kernel_size))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res

##########################################################################
class ARDNs(nn.Module):
    def __init__(self, n_feat, scale_orsnetfeats, kernel_size, reduction, act, bias, scale_unetfeats, num_cfm):
        super(ARDNs, self).__init__()

        self.ardn1 = ARDN(n_feat+scale_orsnetfeats, kernel_size, reduction, act, bias, num_cfm)
        self.ardn2 = ARDN(n_feat+scale_orsnetfeats, kernel_size, reduction, act, bias, num_cfm)
     
        self.up_enc1 = UpSample(n_feat, scale_unetfeats)
        self.up_dec1 = UpSample(n_feat, scale_unetfeats)

        self.up_enc2 = nn.Sequential(UpSample(n_feat+scale_unetfeats, scale_unetfeats), UpSample(n_feat, scale_unetfeats))
        self.up_dec2 = nn.Sequential(UpSample(n_feat+scale_unetfeats, scale_unetfeats), UpSample(n_feat, scale_unetfeats))

        self.conv_enc1 = nn.Conv2d(n_feat, n_feat+scale_orsnetfeats, kernel_size=1, bias=bias)
        self.conv_enc2 = nn.Conv2d(n_feat, n_feat+scale_orsnetfeats, kernel_size=1, bias=bias)
        self.conv_enc3 = nn.Conv2d(n_feat, n_feat+scale_orsnetfeats, kernel_size=1, bias=bias)

        self.conv_dec1 = nn.Conv2d(n_feat, n_feat+scale_orsnetfeats, kernel_size=1, bias=bias)
        self.conv_dec2 = nn.Conv2d(n_feat, n_feat+scale_orsnetfeats, kernel_size=1, bias=bias)
        self.conv_dec3 = nn.Conv2d(n_feat, n_feat+scale_orsnetfeats, kernel_size=1, bias=bias)

    def forward(self, x, encoder_outs, decoder_outs):
        x = self.ardn1(x)
        x = x + self.conv_enc1(encoder_outs[0]) + self.conv_dec1(decoder_outs[0])

        x = self.ardn2(x)
        x = x + self.conv_enc2(self.up_enc1(encoder_outs[1])) + self.conv_dec2(self.up_dec1(decoder_outs[1]))
        return x


##########################################################################
class RCFRMS(nn.Module):
    def __init__(self, in_c=3, out_c=3, n_feat=40, scale_unetfeats=20, scale_orsnetfeats=16, num_cfm=8, kernel_size=3, reduction=4, bias=False):
        super(RCFRMS, self).__init__()

        act=nn.PReLU()
        self.shallow_feat1 = nn.Sequential(conv(in_c, n_feat, kernel_size, bias=bias), CFM(n_feat,kernel_size, reduction, bias=bias, act=act))
        self.shallow_feat2 = nn.Sequential(conv(in_c, n_feat, kernel_size, bias=bias), CFM(n_feat,kernel_size, reduction, bias=bias, act=act))
        self.shallow_feat3 = nn.Sequential(conv(in_c, n_feat, kernel_size, bias=bias), CFM(n_feat,kernel_size, reduction, bias=bias, act=act))

        # Cross Stage Feature Fusion (CSFF)
        self.stage1_encoder = Encoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats, csff=False)
        self.stage1_decoder = Decoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats)

        self.stage2_encoder = Encoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats, csff=True)
        self.stage2_decoder = Decoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats)

        self.stage3_mrdns = MRDNs(n_feat, scale_orsnetfeats, kernel_size, reduction, act, bias, scale_unetfeats, num_cfm)

        self.mab12 = MAB(n_feat, kernel_size=1, bias=bias)
        self.mab23 = MAB(n_feat, kernel_size=1, bias=bias)
        
        self.concat12  = conv(n_feat*2, n_feat, kernel_size, bias=bias)
        self.concat23  = conv(n_feat*2, n_feat+scale_mrdnsfeats, kernel_size, bias=bias)
        self.tail     = conv(n_feat+scale_mrdnsfeats, out_c, kernel_size, bias=bias)

    def forward(self, x4_img):
        # Original-resolution Image for Stage 4
        H = x4_img.size(2)
        W = x4_img.size(3)

        # Multi-Patch Hierarchy: Split Image into four non-overlapping patches

        # Two Patches for Stage 3
        x3top_img  = x4_img[:,:,0:int(H/2),:]
        x3bot_img  = x4_img[:,:,int(H/2):H,:]

        # Four Patches for Stage 2
        x2ltop_img = x3top_img[:,:,:,0:int(W/2)]
        x2rtop_img = x3top_img[:,:,:,int(W/2):W]
        x2lbot_img = x3bot_img[:,:,:,0:int(W/2)]
        x2rbot_img = x3bot_img[:,:,:,int(W/2):W]

        ##-------------------------------------------
        ##-------------- Stage 1---------------------
        ##-------------------------------------------
        ## Compute Shallow Features
        x1ltop = self.shallow_feat1(x1ltop_img)
        x1rtop = self.shallow_feat1(x1rtop_img)
        x1ltop = self.shallow_feat1(x1ltop_img)
        x1rtop = self.shallow_feat1(x1rtop_img)
        x1lbot = self.shallow_feat1(x1lbot_img)
        x1rbot = self.shallow_feat1(x1rbot_img)
        x1lbot = self.shallow_feat1(x1lbot_img)
        x1rbot = self.shallow_feat1(x1rbot_img)


        # Eight Patches for Stage 1
        x1ltop_img = x2top_img[:,:,:,0:int(W/4)]
        x1rtop_img = x2top_img[:,:,:,int(W/4):W]
        x1ltop_img = x2top_img[:,:,:,0:int(H/4)]
        x1rtop_img = x2top_img[:,:,:,int(H/4):H]
        x1lbot_img = x2bot_img[:,:,:,0:int(W/4)]
        x1rbot_img = x2bot_img[:,:,:,int(W/4):W]
        x1lbot_img = x2bot_img[:,:,:,0:int(H/4)]
        x1rbot_img = x2bot_img[:,:,:,int(H/4):H]

 ##-------------------------------------------
        ##-------------- Stage 1---------------------
        ##-------------------------------------------
        ## Compute Shallow Features
        x1ltop = self.shallow_feat1(x1ltop_img)
        x1rtop = self.shallow_feat1(x1rtop_img)
        x1ltop = self.shallow_feat1(x1ltop_img)
        x1rtop = self.shallow_feat1(x1rtop_img)
        x1lbot = self.shallow_feat1(x1lbot_img)
        x1rbot = self.shallow_feat1(x1rbot_img)
        x1lbot = self.shallow_feat1(x1lbot_img)
        x1rbot = self.shallow_feat1(x1rbot_img)


        
        ## Process features of all 8 patches with Encoder of Stage 1
        feat1_ltop = self.stage1_encoder(x1ltop)
        feat1_rtop = self.stage1_encoder(x1rtop)
        feat1_ltop = self.stage1_encoder(x1ltop)
        feat1_rtop = self.stage1_encoder(x1rtop)
        feat1_lbot = self.stage1_encoder(x1lbot)
        feat1_rbot = self.stage1_encoder(x1rbot)
        feat1_lbot = self.stage1_encoder(x1lbot)
        feat1_rbot = self.stage1_encoder(x1rbot)
        
        ## Concat deep features
        feat1_top = [torch.cat((k,v), 3) for k,v in zip(feat1_ltop,feat1_rtop)]
        feat1_bot = [torch.cat((k,v), 3) for k,v in zip(feat1_lbot,feat1_rbot)]
        
        ## Pass features through Decoder of Stage 1
        res1_top = self.stage1_decoder(feat1_top)
        res1_bot = self.stage1_decoder(feat1_bot)

        ## Apply Supervised Attention Module (MAB)
        x2top_mabfeats, stage1_img_top = self.mab12(res1_top[0], x2top_img)
        x2bot_mabfeats, stage1_img_bot = self.mab12(res1_bot[0], x2bot_img)

        ## Output image at Stage 1
        stage1_img = torch.cat([stage1_img_top, stage1_img_bot],2) 
        ##-------------------------------------------
        ##-------------- Stage 2---------------------
        ##-------------------------------------------
        ## Compute Shallow Features
        x2top  = self.shallow_feat2(x2top_img)
        x2bot  = self.shallow_feat2(x2bot_img)

        ## Concatenate MAB features of Stage 1 with shallow features of Stage 2
        x2top_cat = self.concat12(torch.cat([x2top, x2top_mabfeats], 1))
        x2bot_cat = self.concat12(torch.cat([x2bot, x2bot_mabfeats], 1))

        ## Process features of both patches with Encoder of Stage 2
        feat2_top = self.stage2_encoder(x2top_cat, feat1_top, res1_top)
        feat2_bot = self.stage2_encoder(x2bot_cat, feat1_bot, res1_bot)

        ## Concat deep features
        feat2 = [torch.cat((k,v), 2) for k,v in zip(feat2_top,feat2_bot)]

        ## Pass features through Decoder of Stage 2
        res2 = self.stage2_decoder(feat2)

        ## Apply MAB
        x3_mabfeats, stage2_img = self.mab23(res2[0], x4_img)


        ##-------------------------------------------
        ##-------------- Stage 3---------------------
        ##-------------------------------------------
        ## Compute Shallow Features
        x3     = self.shallow_feat3(x4_img)

        ## Concatenate SAM features of Stage 2 with shallow features of Stage 3
        x3_cat = self.concat23(torch.cat([x3, x3_samfeats], 1))
        
        x3_cat = self.stage3_orsnet(x3_cat, feat2, res2)

        stage3_img = self.tail(x3_cat)
##-------------------------------------------
        ##-------------- Stage 4---------------------
        ##-------------------------------------------
        ## Compute Shallow Features
        x4     = self.shallow_feat3(x3_img)

        ## Concatenate SAM features of Stage 2 with shallow features of Stage 3
        x4_cat = self.concat23(torch.cat([x3, x3_samfeats], 1))
        
        x4_cat = self.stage3_orsnet(x4_cat, feat2, res2)

        stage4_img = self.tail(x4_cat)

        return [stage4_img+x4_img, stage3_img, stage2_img, stage1_img]

