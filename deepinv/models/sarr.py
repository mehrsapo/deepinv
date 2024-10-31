import torch

from deepinv.models.ridge_regularizer import RidgeRegularizer
from deepinv.physics import Denoising

from .utils import get_weights_url


class SARR(torch.nn.Module):
    r"""
    Weakly Convex Ridge Regularizer with Spatial Adaptivity

    Implementation of the `weakly convex ridge regularizer <>. The regularizer is defined as

    .. math::

        R(x)=\sum_{i=1}^N \psi_i(W_i x)

    where :math:`W_i` are some convolutions and :math:`\psi_i` are some weakly convex activation functions parameterized by splines as defined. In practice, the :math:`W_i` are realized by a concatenation of several convolutions without non-linearities, where the number of channels of these convolutions can be specified in the constructor.


    :param list of int channel_sequence: number of channels for the convolutions
    :param int kernel_size: kernel sizes for the convolutions
    :param float max_noise_level: maximum noise level where the model can be trained
    :param float rho_convex: modulus of weak convexity
    :param list of int spline_knots: spline_knots[0] is the number of knots of the scaling splines and spline_knots[1] is the number of knots for the potentials
    :param str mask_network: mask generation network name
    """

    def __init__(
        self,
        channel_sequence=[1, 4, 8, 80],
        kernel_size=5,
        max_noise_level=30.0 / 255.0,
        rho_wconvex=1.0,
        spline_knots=[11, 101],
        mask_network='RFDN',
        pretrained="download",
    ):
        super().__init__()

        print(pretrained)
        self.wcrr_init = RidgeRegularizer(channel_sequence=channel_sequence, kernel_size=kernel_size, max_noise_level=max_noise_level, rho_wconvex=rho_wconvex, spline_knots=spline_knots, pretrained=None)
        if mask_network == 'RFDN':
            self.mask_network = RFDN(in_nc=channel_sequence[0], out_nc=channel_sequence[-1], upscale=1, nf = 40)
        else:
            raise NotImplementedError
        self.wcrr_final = RidgeRegularizer(channel_sequence=channel_sequence, kernel_size=kernel_size, max_noise_level=max_noise_level, rho_wconvex=rho_wconvex, spline_knots=spline_knots, pretrained=None)

        if pretrained is not None:
            if pretrained == "download":
                url = get_weights_url(model_name="wcrr", file_name="wcrr_gray.pth")
                ckpt = torch.hub.load_state_dict_from_url(
                    url, map_location=lambda storage, loc: storage, file_name=name
                )
            else:
                ckpt = torch.load(pretrained, map_location=lambda storage, loc: storage)
            self.load_state_dict(ckpt)

    
    def load_state_dict(self, state_dict, **kwargs):
        r"""
        The load_state_dict method is overloaded to handle some internal parameters.
        """
        super().load_state_dict(state_dict, **kwargs)
        self.wcrr_init.potential.phi_plus.hyper_param_to_device()
        self.wcrr_init.potential.phi_minus.hyper_param_to_device()
        self.wcrr_init.W.spectral_norm(mode="power_method", n_steps=100)

        self.wcrr_final.potential.phi_plus.hyper_param_to_device()
        self.wcrr_final.potential.phi_minus.hyper_param_to_device()
        self.wcrr_final.W.spectral_norm(mode="power_method", n_steps=100)


    def forward(self, x, sigma, tol=1e-4, max_iter=500, mask=1.):
        r"""
        Solve the denoising problem for the Spatial Adaptive (Weakly) Convex Ridge Regularizer 

        via an accelerated gradient descent. When called without torch.no_grad() this might require a large amount of memory.
        """

        return self.reconstruct(
            Denoising(),
            x,
            sigma,
            1.0,
            tol=tol,
            max_iter=max_iter,
            physics_norm=1,
            init=x
        )

    def reconstruct(
        self,
        physics,
        y,
        sigma,
        lmbd,
        tol=1e-4,
        max_iter=500,
        physics_norm=None,
        init=None,
        ):

        rec_init = self.wcrr_init.reconstruct(physics=physics, y=y, sigma=sigma, lmbd=lmbd, tol=tol, max_iter=max_iter, physics_norm=physics_norm, init=init)
        mask =torch.sigmoid(self.mask_network(rec_init))
        rec_final = self.wcrr_final.reconstruct(physics=physics, y=y, sigma=sigma, lmbd=lmbd, tol=tol, max_iter=max_iter, physics_norm=physics_norm, init=rec_init, mask=mask)

        return [rec_init, rec_final]



def make_model(args, parent=False):
    model = RFDN()
    return model


class RFDN(torch.nn.Module):
    def __init__(self, in_nc=3, nf=50, num_modules=4, out_nc=3, upscale=4):
        super(RFDN, self).__init__()

        self.fea_conv = conv_layer(in_nc, nf, kernel_size=3)

        self.B1 = RFDB(in_channels=nf)
        self.B2 = RFDB(in_channels=nf)
        self.B3 = RFDB(in_channels=nf)
        self.B4 = RFDB(in_channels=nf)
        self.c = conv_block(nf * num_modules, nf, kernel_size=1, act_type='lrelu')

        self.LR_conv = conv_layer(nf, nf, kernel_size=3)

        upsample_block = pixelshuffle_block
        self.upsampler = upsample_block(nf, out_nc, upscale_factor=upscale)
        self.scale_idx = 0


    def forward(self, input):
        out_fea = self.fea_conv(input)
        out_B1 = self.B1(out_fea)
        out_B2 = self.B2(out_B1)
        out_B3 = self.B3(out_B2)
        out_B4 = self.B4(out_B3)

        out_B = self.c(torch.cat([out_B1, out_B2, out_B3, out_B4], dim=1))
        out_lr = self.LR_conv(out_B) + out_fea

        output = self.upsampler(out_lr)

        return output

    def set_scale(self, scale_idx):
        self.scale_idx = scale_idx

def conv_layer(in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1):
    padding = int((kernel_size - 1) / 2) * dilation
    return torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, bias=True, dilation=dilation,
                     groups=groups)


def norm(norm_type, nc):
    norm_type = norm_type.lower()
    if norm_type == 'batch':
        layer = torch.nn.BatchNorm2d(nc, affine=True)
    elif norm_type == 'instance':
        layer = torch.nn.InstanceNorm2d(nc, affine=False)
    else:
        raise NotImplementedError('normalization layer [{:s}] is not found'.format(norm_type))
    return layer


def pad(pad_type, padding):
    pad_type = pad_type.lower()
    if padding == 0:
        return None
    if pad_type == 'reflect':
        layer = torch.nn.ReflectionPad2d(padding)
    elif pad_type == 'replicate':
        layer = torch.nn.ReplicationPad2d(padding)
    else:
        raise NotImplementedError('padding layer [{:s}] is not implemented'.format(pad_type))
    return layer


def get_valid_padding(kernel_size, dilation):
    kernel_size = kernel_size + (kernel_size - 1) * (dilation - 1)
    padding = (kernel_size - 1) // 2
    return padding


def conv_block(in_nc, out_nc, kernel_size, stride=1, dilation=1, groups=1, bias=True,
               pad_type='zero', norm_type=None, act_type='relu'):
    padding = get_valid_padding(kernel_size, dilation)
    p = pad(pad_type, padding) if pad_type and pad_type != 'zero' else None
    padding = padding if pad_type == 'zero' else 0

    c = torch.nn.Conv2d(in_nc, out_nc, kernel_size=kernel_size, stride=stride, padding=padding,
                  dilation=dilation, bias=bias, groups=groups)
    a = activation(act_type) if act_type else None
    n = norm(norm_type, out_nc) if norm_type else None
    return sequential(p, c, n, a)


def activation(act_type, inplace=True, neg_slope=0.05, n_prelu=1):
    act_type = act_type.lower()
    if act_type == 'relu':
        layer = torch.nn.ReLU(inplace)
    elif act_type == 'lrelu':
        layer = torch.nn.LeakyReLU(neg_slope, inplace)
    elif act_type == 'prelu':
        layer = torch.nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    else:
        raise NotImplementedError('activation layer [{:s}] is not found'.format(act_type))
    return layer


class ShortcutBlock(torch.nn.Module):
    def __init__(self, submodule):
        super(ShortcutBlock, self).__init__()
        self.sub = submodule

    def forward(self, x):
        output = x + self.sub(x)
        return output

def mean_channels(F):
    assert(F.dim() == 4)
    spatial_sum = F.sum(3, keepdim=True).sum(2, keepdim=True)
    return spatial_sum / (F.size(2) * F.size(3))

def stdv_channels(F):
    assert(F.dim() == 4)
    F_mean = mean_channels(F)
    F_variance = (F - F_mean).pow(2).sum(3, keepdim=True).sum(2, keepdim=True) / (F.size(2) * F.size(3))
    return F_variance.pow(0.5)

def sequential(*args):
    if len(args) == 1:
        if isinstance(args[0], OrderedDict):
            raise NotImplementedError('sequential does not support OrderedDict input.')
        return args[0]
    modules = []
    for module in args:
        if isinstance(module, torch.nn.Sequential):
            for submodule in module.children():
                modules.append(submodule)
        elif isinstance(module, torch.nn.Module):
            modules.append(module)
    return torch.nn.Sequential(*modules)

class ESA(torch.nn.Module):
    def __init__(self, n_feats, conv):
        super(ESA, self).__init__()
        f = n_feats // 4
        self.conv1 = conv(n_feats, f, kernel_size=1)
        self.conv_f = conv(f, f, kernel_size=1)
        self.conv_max = conv(f, f, kernel_size=3, padding=1)
        self.conv2 = conv(f, f, kernel_size=3, stride=2, padding=0)
        self.conv3 = conv(f, f, kernel_size=3, padding=1)
        self.conv3_ = conv(f, f, kernel_size=3, padding=1)
        self.conv4 = conv(f, n_feats, kernel_size=1)
        self.sigmoid = torch.nn.Sigmoid()
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, x):
        c1_ = (self.conv1(x))
        c1 = self.conv2(c1_)
        v_max = torch.nn.functional.max_pool2d(c1, kernel_size=7, stride=3)
        v_range = self.relu(self.conv_max(v_max))
        c3 = self.relu(self.conv3(v_range))
        c3 = self.conv3_(c3)
        c3 = torch.nn.functional.interpolate(c3, (x.size(2), x.size(3)), mode='bilinear', align_corners=False) 
        cf = self.conv_f(c1_)
        c4 = self.conv4(c3+cf)
        m = self.sigmoid(c4)
        
        return x * m


class RFDB(torch.nn.Module):
    def __init__(self, in_channels, distillation_rate=0.25):
        super(RFDB, self).__init__()
        self.dc = self.distilled_channels = in_channels//2
        self.rc = self.remaining_channels = in_channels
        self.c1_d = conv_layer(in_channels, self.dc, 1)
        self.c1_r = conv_layer(in_channels, self.rc, 3)
        self.c2_d = conv_layer(self.remaining_channels, self.dc, 1)
        self.c2_r = conv_layer(self.remaining_channels, self.rc, 3)
        self.c3_d = conv_layer(self.remaining_channels, self.dc, 1)
        self.c3_r = conv_layer(self.remaining_channels, self.rc, 3)
        self.c4 = conv_layer(self.remaining_channels, self.dc, 3)
        self.act = activation('lrelu', neg_slope=0.05)
        self.c5 = conv_layer(self.dc*4, in_channels, 1)
        self.esa = ESA(in_channels, torch.nn.Conv2d)

    def forward(self, input):
        distilled_c1 = self.act(self.c1_d(input))
        r_c1 = (self.c1_r(input))
        r_c1 = self.act(r_c1+input)

        distilled_c2 = self.act(self.c2_d(r_c1))
        r_c2 = (self.c2_r(r_c1))
        r_c2 = self.act(r_c2+r_c1)

        distilled_c3 = self.act(self.c3_d(r_c2))
        r_c3 = (self.c3_r(r_c2))
        r_c3 = self.act(r_c3+r_c2)

        r_c4 = self.act(self.c4(r_c3))

        out = torch.cat([distilled_c1, distilled_c2, distilled_c3, r_c4], dim=1)
        out_fused = self.esa(self.c5(out)) 

        return out_fused



def pixelshuffle_block(in_channels, out_channels, upscale_factor=2, kernel_size=3, stride=1):
    conv = conv_layer(in_channels, out_channels * (upscale_factor ** 2), kernel_size, stride)
    pixel_shuffle = torch.nn.PixelShuffle(upscale_factor)
    return sequential(conv, pixel_shuffle)