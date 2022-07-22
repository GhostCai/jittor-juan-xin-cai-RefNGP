import torch
import torch.nn as nn
import torch.nn.functional as Func
from .sh import *
from encoding import get_encoder
from activation import trunc_exp
from .renderer import NeRFRenderer
import numpy as np
import raymarching
def apply_gamma(rgb, gamma="srgb"):
    """Linear to gamma rgb.
    Assume that rgb values are in the [0, 1] range (but values outside are tolerated).
    gamma can be "srgb", a real-valued exponent, or None.
    >>> apply_gamma(torch.tensor([0.5, 0.4, 0.1]).view([1, 3, 1, 1]), 0.5).view(-1)
    tensor([0.2500, 0.1600, 0.0100])
    """
    if gamma == "srgb":
        T = 0.0031308
        rgb1 = torch.max(rgb, rgb.new_tensor(T))
        return torch.where(rgb < T, 12.92 * rgb, (1.055 * torch.pow(torch.abs(rgb1), 1 / 2.4) - 0.055))
    elif gamma is None:
        return rgb
    else:
        return torch.pow(torch.max(rgb, rgb.new_tensor(0.0)), 1.0 / gamma)


def sample_pdf(bins, weights, n_samples, det=False):
    # This implementation is from NeRF
    # bins: [B, T], old_z_vals
    # weights: [B, T - 1], bin weights.
    # return: [B, n_samples], new_z_vals

    # Get pdf
    weights = weights + 1e-5  # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)
    # Take uniform samples
    if det:
        u = torch.linspace(0. + 0.5 / n_samples, 1. - 0.5 / n_samples, steps=n_samples).to(weights.device)
        u = u.expand(list(cdf.shape[:-1]) + [n_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [n_samples]).to(weights.device)

    # Invert CDF
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds - 1), inds - 1)
    above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (B, n_samples, 2)

    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[..., 1] - cdf_g[..., 0])
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples



def sample_spherical(npoints, ndim=3):
    vec = torch.randn(npoints, ndim,device='cuda:0')
   # print(vec.shape)
   # print(torch.linalg.norm(vec, dim=1).shape)
    vec /= (torch.linalg.norm(vec, dim=1).reshape(-1,1).repeat(1,3) + 1e-4)
    return vec
class NeRFNetwork(NeRFRenderer):
    def __init__(self,
                 encoding="hashgrid",
                 encoding_dir="sphere_harmonics",
                 encoding_bg="hashgrid",
                 num_layers=2,
                 hidden_dim=64,
                 geo_feat_dim=15,
                 num_layers_color=3,
                 hidden_dim_color=64,
                 num_layers_sh=5,
                 hidden_dim_sh=128,
                 num_layers_bg=2,
                 hidden_dim_bg=64,
                 bound=1,
                 **kwargs,
                 ):
        super().__init__(bound, **kwargs)
      #  self.sh_coeff = torch.nn.parameter.Parameter(data=torch.randn(1,3,9), requires_grad=True)
        # sigma network
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.geo_feat_dim = geo_feat_dim
        self.encoder, self.in_dim = get_encoder(encoding, desired_resolution=2048 * bound)

        sigma_net = []
        for l in range(num_layers):
            if l == 0:
                in_dim = self.in_dim
            else:
                in_dim = hidden_dim
            
            if l == num_layers - 1:
                out_dim = 1 + self.geo_feat_dim # 1 sigma + 15 SH features for color
            else:
                out_dim = hidden_dim
            
            sigma_net.append(nn.Linear(in_dim, out_dim, bias=True))

        self.sigma_net = nn.ModuleList(sigma_net)

        # color network
        self.num_layers_color = num_layers_color        
        self.hidden_dim_color = hidden_dim_color
        self.num_layers_sh = num_layers_sh
        self.hidden_dim_sh = hidden_dim_sh
        self.encoder_dir, self.in_dim_dir = get_encoder(encoding_dir)
        
        color_net =  [] #albedo
        for l in range(num_layers_color):
            if l == 0:
                in_dim = self.in_dim# + self.geo_feat_dim
            else:
                in_dim = hidden_dim
            
            if l == num_layers_color - 1:
                out_dim = 3 # 3 rgb
            else:
                out_dim = hidden_dim
            
            color_net.append(nn.Linear(in_dim, out_dim, bias=True))

        self.color_net = nn.ModuleList(color_net)

        specular_net =  []
        for l in range(num_layers_color):
            if l == 0:
                in_dim = self.in_dim_dir + self.in_dim
            else:
                in_dim = hidden_dim
            
            if l == num_layers_color - 1:
                out_dim = 3 # 3 rgb
            else:
                out_dim = hidden_dim
            
            specular_net.append(nn.Linear(in_dim, out_dim, bias=True))

        self.specular_net = nn.ModuleList(specular_net)

        # mat_net = [] #
        # for l in range(num_layers_color):
        #     if l == 0:
        #         in_dim = self.in_dim #+  self.geo_feat_dim
        #     else:
        #         in_dim = hidden_dim
            
        #     if l == num_layers_color - 1:
        #         out_dim = 1 # 1 roughness
        #     else:
        #         out_dim = hidden_dim
            
        #     mat_net.append(nn.Linear(in_dim, out_dim, bias=True))

        # self.mat_net = nn.ModuleList(mat_net)

        normal_net = [] #
        for l in range(num_layers_color):
            if l == 0:
                in_dim = self.in_dim # +  self.geo_feat_dim
            else:
                in_dim = hidden_dim
            
            if l == num_layers_color - 1:
                out_dim = 2 # 2 phi and theta
            else:
                out_dim = hidden_dim
            
            normal_net.append(nn.Linear(in_dim, out_dim, bias=True))

        self.normal_net = nn.ModuleList(normal_net)

        # background network
        if self.bg_radius > 0:
            self.num_layers_bg = num_layers_bg        
            self.hidden_dim_bg = hidden_dim_bg
            self.encoder_bg, self.in_dim_bg = get_encoder(encoding_bg, input_dim=2, num_levels=4, log2_hashmap_size=19, desired_resolution=2048) # much smaller hashgrid 
            
            bg_net = []
            for l in range(num_layers_bg):
                if l == 0:
                    in_dim = self.in_dim_bg + self.in_dim_dir
                else:
                    in_dim = hidden_dim_bg
                
                if l == num_layers_bg - 1:
                    out_dim = 3 # 3 rgb
                else:
                    out_dim = hidden_dim_bg
                
                bg_net.append(nn.Linear(in_dim, out_dim, bias=False))

            self.bg_net = nn.ModuleList(bg_net)
        else:
            self.bg_net = None


    def forward(self, x, d):
        # x: [N, 3], in [-bound, bound]
        # d: [N, 3], nomalized in [-1, 1]

        # sigma
        x = self.encoder(x, bound=self.bound)

        h = x
        for l in range(self.num_layers):
            h = self.sigma_net[l](h)
            if l != self.num_layers - 1:
                h = Func.relu(h, inplace=True)

        #sigma = Func.relu(h[..., 0])
        sigma = trunc_exp(h[..., 0])
        geo_feat = h[..., 1:]

        # color
        
       # d = self.encoder_dir(d)
        h = torch.cat([x, geo_feat], dim=-1)
        for l in range(self.num_layers_color):
            h = self.color_net[l](h)
            if l != self.num_layers_color - 1:
                h = Func.relu(h, inplace=True)
        
        # sigmoid activation for rgb
        color = torch.sigmoid(h)


        h = torch.cat([x, geo_feat], dim=-1)
        for l in range(self.num_layers_color):
            h = self.shadow_net[l](h)
            if l != self.num_layers_color - 1:
                h = Func.relu(h, inplace=True)
        
        # sigmoid activation for rgb
        shadow = torch.sigmoid(h)

        h = torch.cat([x, geo_feat], dim=-1)
        for l in range(self.num_layers_color):
            h = self.sh_net[l](h)
            if l != self.num_layers_color - 1:
                h = Func.relu(h, inplace=True)
        
        # sigmoid activation for rgb
        sh = h

        return sigma, color, shadow, sh

    def density(self, x):
        # x: [N, 3], in [-bound, bound]

        x = self.encoder(x, bound=self.bound)
        h = x
        for l in range(self.num_layers):
            h = self.sigma_net[l](h)
            if l != self.num_layers - 1:
                h = Func.relu(h, inplace=True)

        #sigma = Func.relu(h[..., 0])
        sigma = trunc_exp(h[..., 0])
        geo_feat = h[..., 1:]
        return torch.cat([sigma.reshape(-1,1),geo_feat],dim=-1)
        # return {
        #     'sigma': sigma,
        #     'geo_feat': geo_feat,
        # }

    def background(self, x, d):
        # x: [N, 2], in [-1, 1]

        h = self.encoder_bg(x) # [N, C]
        d = self.encoder_dir(d)

        h = torch.cat([d, h], dim=-1)
        for l in range(self.num_layers_bg):
            h = self.bg_net[l](h)
            if l != self.num_layers_bg - 1:
                h = Func.relu(h, inplace=True)
        
        # sigmoid activation for rgb
        rgbs = torch.sigmoid(h)

        return rgbs

    # allow masked inference
    def color(self, x, d, mask=None, geo_feat=None, **kwargs):
        # x: [N, 3] in [-bound, bound]
        # mask: [N,], bool, indicates where we actually needs to compute rgb.
        x = self.encoder(x, bound=self.bound)
        if mask is not None:
            rgbs = torch.zeros(mask.shape[0], 3, dtype=x.dtype, device=x.device) # [N, 3]
            # in case of empty mask
            if not mask.any():
                return rgbs
            x = x[mask]
        #    d = d[mask]
            geo_feat = kwargs['density_outputs'][mask,1:]
       # print(geo_feat.shape)
      #  d = self.encoder_dir(d)
        h = x
        for l in range(self.num_layers_color):
            h = self.color_net[l](h)
            if l != self.num_layers_color - 1:
                h = Func.relu(h, inplace=True)
        
        # sigmoid activation for rgb
        h = torch.sigmoid(h)

        if mask is not None:
            rgbs[mask] = h.to(rgbs.dtype) # fp16 --> fp32
        else:
            rgbs = h

        return rgbs     

    def spec(self, x, d, mask=None, geo_feat=None, **kwargs):
        # x: [N, 3] in [-bound, bound]
        # mask: [N,], bool, indicates where we actually needs to compute rgb.
        x = self.encoder(x, bound=self.bound)
        if mask is not None:
            rgbs = torch.zeros(mask.shape[0], 3, dtype=x.dtype, device=x.device) # [N, 3]
            # in case of empty mask
            if not mask.any():
                return rgbs
            x = x[mask]
            d = d[mask]
            geo_feat = kwargs['density_outputs'][mask,1:]
       # print(geo_feat.shape)
        d = self.encoder_dir(d)
        h = torch.cat([d, x], dim=-1)
        for l in range(self.num_layers_color):
            h = self.specular_net[l](h)
            if l != self.num_layers_color - 1:
                h = Func.relu(h, inplace=True)
        
        # sigmoid activation for rgb
        h = torch.sigmoid(h)

        if mask is not None:
            rgbs[mask] = h.to(rgbs.dtype) # fp16 --> fp32
        else:
            rgbs = h

        return rgbs

    def normal(self, x, mask=None, geo_feat=None, **kwargs):
        
        # x: [N, 3] in [-bound, bound]
        # mask: [N,], bool, indicates where we actually needs to compute rgb.
        x = self.encoder(x, bound=self.bound)
        #print(x.shape)
        geo_feat = kwargs['density_outputs'][:,1:]
       # print(geo_feat.shape)
       # print(x.shape)
       # print(geo_feat.shape)
       # d = self.encoder_dir(d)
        #sh_coeff = sh_coeff.flatten().reshape(1,9).repeat(x.shape[0],1)
        h = x
        for l in range(self.num_layers_color):
            h = self.normal_net[l](h)
            if l != self.num_layers_color - 1:
                h = Func.relu(h, inplace=True)
        
        
        # sigmoid activation for rgb
        h = torch.sigmoid(h)
        theta = h[:,0] * 3.1415926 # theta
        phi = h[:,1] * 3.1415926 * 2 # phi
        sintheta = torch.sin(theta)
        costheta = torch.cos(theta)
        sinphi = torch.sin(phi)
        cosphi = torch.cos(phi)


        ret = torch.zeros((h.shape[0],3),device='cuda:0')
        ret[:,0] = cosphi * sintheta
        ret[:,1] = sinphi * sintheta
        ret[:,2] = costheta
        #print(h.shape)
        if mask is not None:
            rgbs[mask] = ret.to(rgbs.dtype) # fp16 --> fp32
        else:
            rgbs = ret

        return rgbs


   # def sh(self, idx, mask=None, geo_feat=None, **kwargs):
        # x: [N, 3] in [-bound, bound]
        # mask: [N,], bool, indicates where we actually needs to compute rgb.

 
        # h = idx
        # for l in range(self.num_layers_sh):
        #     h = self.sh_net[l](h)
        #     if l != self.num_layers_sh - 1:
        #         h = Func.relu(h, inplace=True)
    

        #return h

    def render(self, rays_o, rays_d, idx, staged=False, max_ray_batch=4096, **kwargs):
        # rays_o, rays_d: [B, N, 3], assumes B == 1
        # return: pred_rgb: [B, N, 3]

        if self.cuda_ray:
            _run = self.run_cuda
        else:
            _run = self.run

        B, N = rays_o.shape[:2]
        device = rays_o.device
        idx = torch.tensor(idx,device='cuda:0').float().view(-1,1)
        # never stage when cuda_ray
        if staged and not self.cuda_ray:
            depth = torch.empty((B, N), device=device)
            image = torch.empty((B, N, 3), device=device)
            image2 = torch.empty((B, N, 3), device=device)
            normal = torch.empty((B, N, 3), device=device)
            normal_pred = torch.empty((B, N, 3), device=device)
            shadow = torch.empty((B, N,3), device=device)

            # 'depth': depth,
            # 'image': image,
            # 'normal': normal,
            # 'normal_pred' : normal_pred,
            # 'normal_diff': normal_diff,
            # 'normal_dot2': normal_dot2,
            # 'shadow': Lo,
            # 'image2': image2

            for b in range(B):
                head = 0
                while head < N:
                    tail = min(head + max_ray_batch, N)
                    
                    results_ = _run(rays_o[b:b+1, head:tail], rays_d[b:b+1, head:tail],idx, **kwargs)
                    depth[b:b+1, head:tail] = results_['depth']
                    image[b:b+1, head:tail] = results_['image']
                    image2[b:b+1, head:tail] = results_['image2']
                    normal[b:b+1, head:tail] = results_['normal']
                    normal_pred[b:b+1, head:tail] = results_['normal_pred']
                    shadow[b:b+1, head:tail] = results_['shadow']
                    head += max_ray_batch
            
            results = {}
            results['depth'] = depth
            results['image'] = image
            results['image2'] = image2
            results['normal'] = normal
            results['normal_pred'] = normal_pred
            results['shadow'] = shadow

        else:
            results = _run(rays_o, rays_d, idx,**kwargs)

        return results
    def gradient(self, x): 
        with torch.enable_grad():
            x.requires_grad_(True)
            y = self.density(x)
            y = y[:,0].view(-1,1)
            d_output = torch.ones_like(y, requires_grad=True, device=y.device)
            #print(x.shape)
        # print(y.shape)
            gradients = torch.autograd.grad(
                outputs=y,
                inputs=x,
                grad_outputs=d_output,
                create_graph=True,
                retain_graph=True,
                only_inputs=True)[0]
        return gradients
    def run(self, rays_o, rays_d, image_idx, num_steps=128, upsample_steps=128, bg_color=None, perturb=False, **kwargs):
        # rays_o, rays_d: [B, N, 3], assumes B == 1
        # bg_color: [3] in range [0, 1]
        # return: image: [B, N, 3], depth: [B, N]
       # print(rays_o.shape)
        prefix = rays_o.shape[:-1]
        rays_o = rays_o.contiguous().view(-1, 3)
        rays_d = rays_d.contiguous().view(-1, 3)

        N = rays_o.shape[0] # N = B * N, in fact
        device = rays_o.device

        # choose aabb
        aabb = self.aabb_train if self.training else self.aabb_infer

        # sample steps
        nears, fars = raymarching.near_far_from_aabb(rays_o, rays_d, aabb, self.min_near)
        nears.unsqueeze_(-1)
        fars.unsqueeze_(-1)

        #print(f'nears = {nears.min().item()} ~ {nears.max().item()}, fars = {fars.min().item()} ~ {fars.max().item()}')

        z_vals = torch.linspace(0.0, 1.0, num_steps, device=device).unsqueeze(0) # [1, T]
        z_vals = z_vals.expand((N, num_steps)) # [N, T]
        z_vals = nears + (fars - nears) * z_vals # [N, T], in [nears, fars]

        # perturb z_vals
        sample_dist = (fars - nears) / num_steps
        if perturb:
            z_vals = z_vals + (torch.rand(z_vals.shape, device=device) - 0.5) * sample_dist
            #z_vals = z_vals.clamp(nears, fars) # avoid out of bounds xyzs.

        # generate xyzs
        xyzs = rays_o.unsqueeze(-2) + rays_d.unsqueeze(-2) * z_vals.unsqueeze(-1) # [N, 1, 3] * [N, T, 1] -> [N, T, 3]
        xyzs = torch.min(torch.max(xyzs, aabb[:3]), aabb[3:]) # a manual clip.

        #plot_pointcloud(xyzs.reshape(-1, 3).detach().cpu().numpy())

        # query SDF and RGB
        density_outputs = self.density(xyzs.reshape(-1, 3))

        #sigmas = density_outputs['sigma'].view(N, num_steps) # [N, T]
        # for k, v in density_outputs.items():
        #     density_outputs[k] = v.view(N, num_steps, -1)

        density_outputs = density_outputs.view(N, num_steps, -1)
        
        # upsample z_vals (nerf-like)
        if upsample_steps > 0:
            with torch.no_grad():

                deltas = z_vals[..., 1:] - z_vals[..., :-1] # [N, T-1]
                deltas = torch.cat([deltas, sample_dist * torch.ones_like(deltas[..., :1])], dim=-1)

                alphas = 1 - torch.exp(-deltas * self.density_scale * density_outputs[:,:,0].squeeze(-1)) # [N, T]
                alphas_shifted = torch.cat([torch.ones_like(alphas[..., :1]), 1 - alphas + 1e-15], dim=-1) # [N, T+1]
                weights = alphas * torch.cumprod(alphas_shifted, dim=-1)[..., :-1] # [N, T]

                # sample new z_vals
                z_vals_mid = (z_vals[..., :-1] + 0.5 * deltas[..., :-1]) # [N, T-1]
                new_z_vals = sample_pdf(z_vals_mid, weights[:, 1:-1], upsample_steps, det=not self.training).detach() # [N, t]

                new_xyzs = rays_o.unsqueeze(-2) + rays_d.unsqueeze(-2) * new_z_vals.unsqueeze(-1) # [N, 1, 3] * [N, t, 1] -> [N, t, 3]
                new_xyzs = torch.min(torch.max(new_xyzs, aabb[:3]), aabb[3:]) # a manual clip.

            # only forward new points to save computation
            new_density_outputs = self.density(new_xyzs.reshape(-1, 3))
           

            #new_sigmas = new_density_outputs['sigma'].view(N, upsample_steps) # [N, t]
            # for k, v in new_density_outputs.items():
            #     new_density_outputs[k] = v.view(N, upsample_steps, -1)
            new_density_outputs = new_density_outputs.view(N, num_steps, -1)

            # re-order
            z_vals = torch.cat([z_vals, new_z_vals], dim=1) # [N, T+t]
            z_vals, z_index = torch.sort(z_vals, dim=1)
        
            xyzs = torch.cat([xyzs, new_xyzs], dim=1) # [N, T+t, 3]
            xyzs = torch.gather(xyzs, dim=1, index=z_index.unsqueeze(-1).expand_as(xyzs))

          #  for k in density_outputs:
                # tmp_output = torch.cat([density_outputs[k], new_density_outputs[k]], dim=1)
                # density_outputs[k] = torch.gather(tmp_output, dim=1, index=z_index.unsqueeze(-1).expand_as(tmp_output))
            tmp_output = torch.cat([density_outputs, new_density_outputs], dim=1)
            density_outputs = torch.gather(tmp_output, dim=1, index=z_index.unsqueeze(-1).expand_as(tmp_output))
    
        deltas = z_vals[..., 1:] - z_vals[..., :-1] # [N, T+t-1]
        deltas = torch.cat([deltas, sample_dist * torch.ones_like(deltas[..., :1])], dim=-1)
        alphas = 1 - torch.exp(-deltas * self.density_scale * density_outputs[:,:,0].squeeze(-1)) # [N, T+t]
        alphas_shifted = torch.cat([torch.ones_like(alphas[..., :1]), 1 - alphas + 1e-15], dim=-1) # [N, T+t+1]
        weights = alphas * torch.cumprod(alphas_shifted, dim=-1)[..., :-1] # [N, T+t]

       # jacobian = torch.autograd.functional.jacobian(self.density,xyzs.reshape(-1, 3))
       # idx_jac = torch.linspace(0,jacobian.shape(0) - 1, jacobian.shape(0), device=device).long()
        density_outputs = density_outputs.view(-1, density_outputs.shape[-1])
        
        normal_pred = self.normal(xyzs.reshape(-1, 3), mask=None, density_outputs=density_outputs) #[N,T+t,3]
        normal_pred = normal_pred.view(N,-1,3)

        dirs_r = rays_d.view(-1, 1, 3).expand_as(xyzs)
        mask = weights > 1e-4 # hard coded
        rgbs = self.color(xyzs.reshape(-1, 3), dirs_r.reshape(-1, 3), mask=mask.reshape(-1), density_outputs=density_outputs)
        dirs_v = -dirs_r

        VdotN = (dirs_v * normal_pred).sum(dim=-1).reshape(N,-1,1).repeat(1,1,3)

        dirs_ref = 2 * VdotN * normal_pred - dirs_v



        spec = self.spec(xyzs.reshape(-1, 3), dirs_ref.reshape(-1, 3), mask=mask.reshape(-1), density_outputs=density_outputs)

       # print(normal_pred)
        
        normal_dot2 = (rays_d.reshape(N,1,3).repeat(1,normal_pred.shape[1],1) * normal_pred).sum(dim=-1).reshape(N,-1,1)

        normal_dot2 = torch.maximum(normal_dot2,torch.tensor(0,device='cuda:0')) # [N,T+t,1]
        normal_dot2 = normal_dot2 * normal_dot2
        normal_dot2 = torch.sum(weights.view(N,-1,1) * normal_dot2, dim=1).mean()

        jacobian = -self.gradient(xyzs.reshape(-1, 3))
        jacobian = jacobian / (torch.linalg.norm(jacobian,axis=-1).view(jacobian.shape[0],1)+ 1e-5) # [N,3]
        jacobian = jacobian.view(N, -1, 3)
        normal_diff = (jacobian.detach() - normal_pred)
        normal_diff = (normal_diff * normal_diff)  
        normal_diff = torch.sum(weights.view(N,-1,1) * normal_diff, dim=1).mean()
       
        


       # print(jacobian.shape)
       # jacobian = jacobian[idx_jac,idx_jac,:] # [N,T+t,3]
        
        normal = torch.sum(weights.view(N,-1,1) * jacobian, dim=1)
        normal = normal / (torch.linalg.norm(normal,axis=-1).view(normal.shape[0],1)+ 1e-5) # [N,3]


        normal_pred = torch.sum(weights.view(N,-1,1) * normal_pred, dim=1)
        #print(normal)
        normal_pred = normal_pred / (torch.linalg.norm(normal_pred,axis=-1).view(normal_pred.shape[0],1)+ 1e-5) # [N,3]
       
        #print(normal)
        

        
        
        # for k, v in density_outputs.items():
        #     density_outputs[k] = v.view(-1, v.shape[-1])
        


       
        
       # shadows = self.shadow(xyzs.reshape(-1, 3), self.sh_coeff.sum(dim=1), mask=mask.reshape(-1), density_outputs=density_outputs)# [N,1]
        #print(shadows.shape)
       
        rgbs = rgbs.view(N, -1, 3) # [N, T+t, 3]
        spec = spec.view(N,-1,3)

        


        # calculate color
        image = torch.sum(weights.unsqueeze(-1) * rgbs, dim=-2) # [N, 3], in [0, 1]
        spec = torch.sum(weights.unsqueeze(-1) * spec, dim=-2) # [N, 3], in [0, 1]
       # print(rgbs.shape)
       # print(image_idx)
       # sh = self.sh(image_idx).reshape(-1,3,9)

   

        # calculate weight_sum (mask)
        weights_sum = weights.sum(dim=-1) # [N]
        
        # calculate depth 
        ori_z_vals = ((z_vals - nears) / (fars - nears)).clamp(0, 1)
        depth = torch.sum(weights * ori_z_vals, dim=-1)
       

        

       
       
       
       # image = (kD *  image.reshape(N,1,3).repeat(1,64,1) * sh_res * NdotL).sum(dim=1).reshape(N,3)

        
       # print(image.sum(dim=0))
       # print(rgbs.shape)
        #print(shadows.shape)
      #  shadows = torch.sum(weights.unsqueeze(-1) * shadows, dim=-2) # [N, 1], in [0, 1]
       # print(image.shape)
      #  print(sh_res.shape)
       # print(shadows.shape)
        # print('------')
        # print(torch.any(torch.isnan(image)))
        # print(torch.any(torch.isnan(sh_res)))
        # print(torch.any(torch.isnan(shadows)))
      #  print('------')
      #  print(shadows)
       
        
       # print(sh_res)
       # print(image.shape)
       # print(sh_res)
       # print(shadows)
     #   print(normal)
        image2 = image
        image = (image ) +spec 
       # print(sh_res)
        #print(image)
        #print(sh_res.sum(dim=0))
      #  np.savetxt('image.txt', image.detach().cpu().numpy())
     #   np.savetxt('shres.txt', sh_res.detach().cpu().numpy())
      #  exit(1)
        # mix background color
        if self.bg_radius > 0:
            # use the bg model to calculate bg_color
            polar = raymarching.polar_from_ray(rays_o, rays_d, self.bg_radius) # [N, 2] in [-1, 1]
            bg_color = self.background(polar, rays_d.reshape(-1, 3)) # [N, 3]
        elif bg_color is None:
            bg_color = 1
        
        image = image + (1 - weights_sum).unsqueeze(-1) * bg_color


       # image = apply_gamma(image)
        
        
        

        image = image.view(*prefix, 3)
        depth = depth.view(*prefix)

        # tmp: reg loss in mip-nerf 360
        # z_vals_shifted = torch.cat([z_vals[..., 1:], sample_dist * torch.ones_like(z_vals[..., :1])], dim=-1)
        # mid_zs = (z_vals + z_vals_shifted) / 2 # [N, T]
        # loss_dist = (torch.abs(mid_zs.unsqueeze(1) - mid_zs.unsqueeze(2)) * (weights.unsqueeze(1) * weights.unsqueeze(2))).sum() + 1/3 * ((z_vals_shifted - z_vals_shifted) * (weights ** 2)).sum()

        return {
            'depth': depth,
            'image': image,
            'normal': normal,
            'normal_pred' : normal_pred,
            'normal_diff': normal_diff,
            'normal_dot2': normal_dot2,
            'shadow': spec,
            'image2': image2
        }

    # optimizer utils
    def get_params(self, lr):

        params = [
            {'params': self.encoder.parameters(), 'lr': lr},
            {'params': self.sigma_net.parameters(), 'lr': lr},
            {'params': self.encoder_dir.parameters(), 'lr': lr},
            {'params': self.color_net.parameters(), 'lr': lr}, 
            #{'params': self.sh_coeff, 'lr': lr}, 
            {'params': self.specular_net.parameters(), 'lr': lr}, 
            {'params': self.normal_net.parameters(), 'lr': lr}, 
        ]
        if self.bg_radius > 0:
            params.append({'params': self.encoder_bg.parameters(), 'lr': lr})
            params.append({'params': self.bg_net.parameters(), 'lr': lr})
        
        return params
