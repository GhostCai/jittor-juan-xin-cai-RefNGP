from turtle import pos, position
import jittor as jt
from jittor import nn, init
import os
from jnerf.utils.config import get_cfg
from jnerf.utils.registry import build_from_cfg, NETWORKS, ENCODERS
from jnerf.ops.code_ops.fully_fused_mlp import FullyFusedMlp_weight

class FMLP(nn.Module):
    def __init__(self, weight_shapes, weights=None):
        super(FMLP, self).__init__()
        if weights == None:                   
            assert len(weight_shapes) > 2
            self.output_shape1 = weight_shapes[-1]
            dweights = []
            for i in range(len(weight_shapes) - 1):
                dweights.append(init.invariant_uniform((weight_shapes[i], weight_shapes[i+1]), "float16").float16())
        else:
            assert len(weights) >= 2
            self.output_shape1 = weights[-1].shape[-1]
            dweights = weights
        self.func = FullyFusedMlp_weight(dweights)
        con_weights = []
        for i in range(len(dweights)):
            if i == len(dweights) - 1:
                if dweights[i].shape[1] < 16: 
                    dweights[i] = jt.concat([dweights[i], jt.zeros((dweights[i].shape[0], 16 - dweights[i].shape[1]))], -1).float16()
            con_weights.append(dweights[i].transpose(1,0).reshape(-1))
        jt_con_weights = jt.concat(con_weights, -1)
        self.con_weights = jt_con_weights

    def execute(self, x):
        if x.shape[0] == 0:
            return jt.empty([0, self.output_shape1]).float16()
        ret = self.func(x, self.con_weights)
        if self.output_shape1 != ret.shape[1]:
            ret = ret[:,:self.output_shape1]
        return ret


@NETWORKS.register_module()
class RefNetworks(nn.Module):
    def __init__(self, use_fully=True, density_hidden_layer=1, density_n_neurons=64, rgb_hidden_layer=2, rgb_n_neurons=64):
        super(RefNetworks, self).__init__()
        self.use_fully = use_fully
        self.cfg = get_cfg()
        self.using_fp16 = self.cfg.fp16
        self.pos_encoder = build_from_cfg(self.cfg.encoder.pos_encoder, ENCODERS)
        self.dir_encoder = build_from_cfg(self.cfg.encoder.dir_encoder, ENCODERS)
        self.IDE_encoder = build_from_cfg(self.cfg.encoder.dir_encoder, ENCODERS)
        if self.use_fully and jt.flags.cuda_archs[0] >= 75:
            assert self.pos_encoder.out_dim%16==0
            assert self.dir_encoder.out_dim%16==0
            # self.spatial_mlp = FMLP([
            #     self.pos_encoder.out_dim,
            #     256,256,256,256,256,256,
            #     16
            # ])
            # self.directional_mlp = FMLP([
            #     3+self.IDE_encoder.out_dim+1,
            #     256,256,256,256,256,256,
            #     3
            # ])
            self.density_mlp = FMLP([self.pos_encoder.out_dim,density_n_neurons,density_n_neurons, 16])
            self.rgb_mlp = FMLP([self.dir_encoder.out_dim+16,rgb_n_neurons, rgb_n_neurons, 3])
        else:
            print("has to use fully!")
            if self.use_fully:
                print("Warning: Sm arch is lower than sm_75, FFMLPs is not supported. Automatically use original MLPs instead.")
            # self.density_mlp = nn.Sequential(
            #     nn.Linear(self.pos_encoder.out_dim, density_n_neurons, bias=False), 
            #     nn.ReLU(), 
            #     nn.Linear(density_n_neurons, 16, bias=False))
            # self.rgb_mlp = nn.Sequential(nn.Linear(self.dir_encoder.out_dim+16, rgb_n_neurons, bias=False),
                            # nn.ReLU(),
                            # nn.Linear(rgb_n_neurons, rgb_n_neurons, bias=False),
                            # nn.ReLU(),
                            # nn.Linear(rgb_n_neurons, 3, bias=False))
        self.set_fp16()

    def execute(self, pos_input, dir_input):  
        if self.using_fp16:
            with jt.flag_scope(auto_mixed_precision_level=5):
                return self.execute_(pos_input, dir_input)
        else:
            return self.execute_(pos_input, dir_input)

    def execute_(self, x, d):  
        # print("NEW")
        # x=self.pos_encoder(x)
        # res = self.spatial_mlp(x)
        # #got density
        # density = res[0:1]
        # cd = res[1:4]
        # s = res[4:7]
        # b = res[7:10]
        # rho = res[10:13]
        # n = res[13:16]
        # # print(n,d)
        # print(n.shape,d.shape)
        # dot_product = jt.sum(jt.matmul(d,n))
        # reflect = 2*jt.sum(jt.matmul(d,n))*n-d
        # IDE_output = jt.concat([rho,reflect],-1)
        # IDE_output = self.IDE_encoder(IDE_input)
        # rgb = jt.concat([b,IDE_output,cdot],-1)
        # rgb = self.directional_mlp(rgb)
        # rgb = rgb * s
        # rgb = rgb + b
        # rgb = tune_map(rgb)
        # outputs = jt.concat([rgb, density[..., :1]], -1)  # batchsize 4: rgbd
        # return outputs
        
        #d batchsize,3
        d = self.dir_encoder(d)

        #encoded_e batchsize,16
        x = self.pos_encoder(x)
        density = self.density_mlp(x)
        rgb = jt.concat([density, d], -1)
        rgb = self.rgb_mlp(rgb)
        # [batchsize,4]
        outputs = jt.concat([rgb, density[..., :1]], -1)  # batchsize 4: rgbd
        return outputs

    def density(self, pos_input):  # batchsize,3
        density = self.pos_encoder(pos_input)
        density = self.density_mlp(density)[:,:1]
        return density

    def set_fp16(self):
        if self.using_fp16:
            self.density_mlp.float16()
            self.rgb_mlp.float16()
            # self.spatial_mlp.float16()
            # self.directional_mlp.float16()
            self.pos_encoder.float16()
            self.dir_encoder.float16()
            self.IDE_encoder.float16()