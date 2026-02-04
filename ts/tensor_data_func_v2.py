import math
import typing
from dataclasses import dataclass, field, fields
from typing import Any, Dict, List, Optional, OrderedDict, Tuple, TypedDict
from torch import nn, Tensor
import torch
from torch.nn.utils import clip_grad_norm_
import time
import copy

from ts.core.arm_func import (Arm, _get_neighbor,  _get_non_zero_pixel_ctx_index,  _laplace_cdf,)
from ts.core.quantizer import (POSSIBLE_QUANTIZATION_NOISE_TYPE, POSSIBLE_QUANTIZER_TYPE,  quantize,)
from ts.core.synthesis_func import Synthesis
from ts.core.upsampling_func import Upsampling
from ts.core.manager import FrameEncoderManager

@dataclass
class CoolChicEncoderParameter:
    layers_synthesis: List[str] = field(default_factory=lambda: ['40-1-linear-relu','3-1-linear-none','3-3-residual-relu','3-3-residual-none'])
    n_ft_per_res: List[int] = field(default_factory=lambda:[1,1,1,1,1,1])
    dim_arm: int = 24
    n_hidden_layers_arm: int = 2
    upsampling_kernel_size: int = 8
    static_upsampling_kernel: bool = False
    encoder_gain: int = 16
    latent_n_grids: int = field(init=False)
    img_size: Optional[Tuple[int, int]] = field(init=False, default=None)
    device:torch.DeviceObjType = 'cuda:0'
    bitdepth:int = 8

    def __post_init__(self):
        self.latent_n_grids = len(self.n_ft_per_res)

    def set_image_size(self, img_size: Tuple[int, int]) -> None:
        self.img_size = img_size
        
    def set_deivce(self,device):
        self.device = device

    def pretty_string(self) -> str:
        """Return a pretty string formatting the data within the class"""
        ATTRIBUTE_WIDTH = 25
        VALUE_WIDTH = 80
        s = "CoolChicEncoderParameter value:\n"
        s += "-------------------------------\n"
        for k in fields(self):
            s += f"{k.name:<{ATTRIBUTE_WIDTH}}: {str(getattr(self, k.name)):<{VALUE_WIDTH}}\n"
        s += "\n"
        return s

def recursive_to_cpu(data):
    if isinstance(data, torch.Tensor):
        return data.cpu().clone().detach()
    elif isinstance(data, dict):
        return {key: recursive_to_cpu(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [recursive_to_cpu(item) for item in data]
    elif isinstance(data, tuple):
        return tuple(recursive_to_cpu(item) for item in data)
    else:
        return data
    
def recursive_to_device(data,device):
    #print('here')
    if isinstance(data, torch.Tensor):
        data = data.to(device)
    elif isinstance(data, dict):
        for _, value in data.items(): recursive_to_device(value,device)
    elif isinstance(data, list):
        for item in data: recursive_to_device(item, device)
    elif isinstance(data, tuple):
        for item in data: recursive_to_device(item, device)
    else:
        pass

class MParams(nn.Module):
    
    def __init__(self)->None:
        self.iter = 0
        self.solver = None
        self.pool = {}
        self.all_parameters = None
        self.lr_schedule = None
        self.need_opt = False
        self.current_device = None
        
    def init_solver(self,needs_opt:list[str],lr,max_itr,freq_valid,schedule=False):
        self.need_opt = False
        self.best_loss = 1e9
        self.best_param = None
        self.best_opt = None
        self.all_parameters =  []
        parameters_to_optimize = []
        for op in needs_opt:
            parameters_to_optimize += [*self.pool[op]]
        if len(parameters_to_optimize) == 0: return
        self.solver = torch.optim.Adam(parameters_to_optimize, lr=lr)
        if schedule:
            self.lr_schedule = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.solver, T_max= max_itr / freq_valid,
                eta_min=0.00001,  last_epoch=-1,
            )
        else:
            self.lr_schedule = None
        
        for pk in self.pool.keys(): self.all_parameters += [*self.pool[pk]]
        self.need_opt = True
        
    def empty_grad(self):
        for param in self.all_parameters:  param.grad = None
    
    def beat(self, loss):
        if loss < self.best_loss:
            self.best_loss = loss
            self.best_opt = self.get_solver()
            self.best_param = self.get_params()
            return True
        else:
            return False
    
    def load_best(self, device):
        self.set_params(self.best_param,device)
        self.set_solver(self.best_opt,device)
        current_lr = self.lr_schedule.state_dict()["_last_lr"][0]
        for g in self.solver.param_groups: g["lr"] = current_lr
        
    def load_best_param(self,device):
        if not self.best_param is None:  self.set_params(self.best_param,device)
        self.best_param = None
        self.best_opt = None
        self.solver = None
        self.all_parameters = None
        
    def step(self):
        if not self.need_opt: return
        clip_grad_norm_(self.all_parameters, 1e-1, norm_type=2.0, error_if_nonfinite=False)
        self.solver.step()
        
        
    def get_params(self):
        snap = {}
        for k,v in self.pool.items():
            snap[k] = OrderedDict({tk:tv.cpu().clone().detach() for tk,tv in v.state_dict().items()})
        return snap
    
    def set_params(self,params_snap,device):
        self.current_device = device
        for k,v in params_snap.items():
            if self.pool[k] is None:
                self.pool[k] = nn.ParameterList([nn.Parameter(v[f'{i}'],requires_grad=True) for i in range(len(v))])
            else:
                self.pool[k].load_state_dict(v)
            self.pool[k] = self.pool[k].to(device)
                 
    def get_solver(self):
        if self.solver is None: return None
        return recursive_to_cpu(self.solver.state_dict())
    
    def set_solver(self,solver_snap,device):
        if not self.need_opt: return
        self.solver.load_state_dict(solver_snap)
        recursive_to_device(self.solver.state_dict(), device)
        #self.solver.to(device)
        
    def to(self,device):
        #print(device)
        if device == self.current_device: return
        self.current_device = device
        for k,v in self.pool.items():
            self.pool[k] = v.to(device)
        if not self.solver is None:  recursive_to_device(self.solver.state_dict(),device)
        
class DGrids(MParams):
    
    def __init__(self, latent_grids:nn.ParameterList=None):
        super(DGrids,self).__init__()
        self.pool['grids'] = latent_grids
        self.bpp = 24


class DParams(MParams):
    
    def __init__(self, arm_param:nn.ParameterList=None, upsampling_param:nn.ParameterList=None, syn_param:nn.ParameterList=None) -> None:
        super().__init__()
        self.pool['ap'] = arm_param
        self.pool['up'] = upsampling_param
        self.pool['sp'] = syn_param


class CoolChicEncoder(nn.Module):

    def __init__(self, param: CoolChicEncoderParameter):
        super().__init__()
        torch.set_printoptions(threshold=10000000)
        self.param = param
        assert self.param.img_size is not None, ( "Image size is needed for initilization parameters.")
        self.encoder_gains = param.encoder_gain
        self.init_shapes()
        max_mask_size = 9
        max_context_pixel = int((max_mask_size**2 - 1) / 2)
        assert self.param.dim_arm <= max_context_pixel, ( f"You can not have more context pixels  than {max_context_pixel}. Found {self.param.dim_arm}")
        self.mask_size = max_mask_size
        self.non_zero_pixel_ctx_index =  _get_non_zero_pixel_ctx_index(self.param.dim_arm)
        #self.register_buffer( "non_zero_pixel_ctx_index",  _get_non_zero_pixel_ctx_index(self.param.dim_arm),  persistent=False,)
        self.arm = Arm(self.param.dim_arm, self.param.n_hidden_layers_arm)
        self.synthesis = Synthesis( sum([latent_size[1] for latent_size in self.size_per_latent]),   self.param.layers_synthesis, )
        self.upsampling = Upsampling(self.param.upsampling_kernel_size, self.param.static_upsampling_kernel)
        
    def forward(self,
        latent_grids,  arm_param:nn.ParameterList,  upsampling_param:nn.ParameterList,  syn_param:nn.ParameterList,
        quantizer_noise_type: POSSIBLE_QUANTIZATION_NOISE_TYPE = "kumaraswamy",
        quantizer_type: POSSIBLE_QUANTIZER_TYPE = "softround",
        soft_round_temperature: Optional[float] = 0.3,
        noise_parameter: Optional[float] = 1.0,
        AC_MAX_VAL: int = -1,
    ) -> Tuple[Tensor,Tensor]:
        
        size_per_latent_flat = [latent_i.numel() for latent_i in latent_grids]
        size_per_latent = [latent_i.shape for latent_i in latent_grids]
        encoder_side_flat_latent = torch.cat([latent_i.view(-1) for latent_i in latent_grids]).contiguous()
        flat_decoder_side_latent = quantize(encoder_side_flat_latent * self.encoder_gains,   quantizer_noise_type if self.training else "none",  quantizer_type if self.training else "hardround", soft_round_temperature, noise_parameter, )
        if AC_MAX_VAL != -1:  flat_decoder_side_latent = torch.clamp(flat_decoder_side_latent, -AC_MAX_VAL, AC_MAX_VAL + 1)
        decoder_side_latent = [ts.view(sz) for ts,sz in zip(torch.split(flat_decoder_side_latent,size_per_latent_flat),size_per_latent)]
        self.non_zero_pixel_ctx_index = self.non_zero_pixel_ctx_index.to(flat_decoder_side_latent.device)
        flat_context = torch.cat([ _get_neighbor(spatial_latent_i, self.mask_size, self.non_zero_pixel_ctx_index)   for spatial_latent_i in decoder_side_latent ],  dim=0)
        flat_latent = flat_decoder_side_latent
        flat_mu, flat_scale, _ = self.arm(flat_context,arm_param)
        proba = torch.clamp_min( _laplace_cdf(flat_latent + 0.5, flat_mu, flat_scale) - _laplace_cdf(flat_latent - 0.5, flat_mu, flat_scale),   min=2**-16, )
        flat_rate = -torch.log2(proba)
        synthesis_output = self.synthesis(self.upsampling(decoder_side_latent,upsampling_param),syn_param)
        
        return synthesis_output,  flat_rate
    
    def forward_per_sample(self,
        latent_grids,  arm_param:nn.ParameterList,  upsampling_param:nn.ParameterList,  syn_param:nn.ParameterList,
        noise:Tensor,quantizer_noise_type: POSSIBLE_QUANTIZATION_NOISE_TYPE = "kumaraswamy",
        quantizer_type: POSSIBLE_QUANTIZER_TYPE = "softround",
        soft_round_temperature: Optional[float] = 0.3,
        noise_parameter: Optional[float] = 1.0,AC_MAX_VAL: int = -1,
    ) -> Tuple[Tensor,Tensor,Tensor]:
        
        size_per_latent_flat = [latent_i.numel() for latent_i in latent_grids]
        size_per_latent = [latent_i.shape for latent_i in latent_grids]
        encoder_side_flat_latent = torch.cat([latent_i.view(-1) for latent_i in latent_grids]).contiguous()
        flat_decoder_side_latent = quantize(encoder_side_flat_latent * self.encoder_gains,   quantizer_noise_type,  quantizer_type, soft_round_temperature, noise_parameter, noise_out=noise)
        if AC_MAX_VAL != -1:  flat_decoder_side_latent = torch.clamp(flat_decoder_side_latent, -AC_MAX_VAL, AC_MAX_VAL + 1)
        decoder_side_latent = [ts.view(sz) for ts,sz in zip(torch.split(flat_decoder_side_latent,size_per_latent_flat),size_per_latent)]
        self.non_zero_pixel_ctx_index = self.non_zero_pixel_ctx_index.to(flat_decoder_side_latent.device)
        flat_context = torch.cat([ _get_neighbor(spatial_latent_i, self.mask_size, self.non_zero_pixel_ctx_index)   for spatial_latent_i in decoder_side_latent ],  dim=0)
        flat_latent = flat_decoder_side_latent
        flat_mu, flat_scale, _ = self.arm(flat_context,arm_param)
        proba = torch.clamp_min( _laplace_cdf(flat_latent + 0.5, flat_mu, flat_scale) - _laplace_cdf(flat_latent - 0.5, flat_mu, flat_scale),   min=2**-16, )
        flat_rate = -torch.log2(proba)
        with torch.no_grad():
            batch = latent_grids[0].shape[0]
            rate_per_grids = [torch.sum(ts.view(batch,-1),dim=1).unsqueeze(0) for ts in torch.split(flat_rate,size_per_latent_flat)]
            rate_per_sample = torch.sum(torch.cat(rate_per_grids),dim=0)
        synthesis_output = self.synthesis(self.upsampling(decoder_side_latent,upsampling_param),syn_param)
        
        return synthesis_output,  flat_rate, rate_per_sample
    
    @torch.no_grad()
    def forward_data(self,  latent_grids:nn.ParameterList, upsampling_param:nn.ParameterList,  syn_param:nn.ParameterList,
                    quantizer_noise_type: POSSIBLE_QUANTIZATION_NOISE_TYPE = "kumaraswamy",
                    quantizer_type: POSSIBLE_QUANTIZER_TYPE = "softround",
                    soft_round_temperature: Optional[float] = 0.3,
                    noise_parameter: Optional[float] = 1.0,AC_MAX_VAL: int = -1,) -> list[Tensor,Tensor]:
        
        size_per_latent_flat = [latent_i.numel() for latent_i in latent_grids]
        size_per_latent = [latent_i.shape for latent_i in latent_grids]
        encoder_side_flat_latent = torch.cat([latent_i.view(-1) for latent_i in latent_grids])
        flat_decoder_side_latent,noise = quantize(encoder_side_flat_latent * self.encoder_gains,  quantizer_noise_type, quantizer_type, soft_round_temperature, noise_parameter, output_noise=True)
        if AC_MAX_VAL != -1:  flat_decoder_side_latent = torch.clamp(flat_decoder_side_latent, -AC_MAX_VAL, AC_MAX_VAL + 1)
        decoder_side_latent = [ts.view(sz) for ts,sz in zip(torch.split(flat_decoder_side_latent,size_per_latent_flat),size_per_latent)]
        synthesis_output = self.synthesis(self.upsampling(decoder_side_latent,upsampling_param),syn_param)
        
        return synthesis_output,noise


    def init_shapes(self)->None:
        self.size_per_latent_flat = []
        self.size_per_latent = []
        self.grids_nparameters = 0
        for i in range(self.param.latent_n_grids):
            h_grid, w_grid = [int(math.ceil(x / (2**i))) for x in self.param.img_size]
            c_grid = self.param.n_ft_per_res[i]
            cur_size = (1, c_grid, h_grid, w_grid)
            self.size_per_latent.append(cur_size)
            cur_total = 1 * c_grid * h_grid * w_grid
            self.size_per_latent_flat.append(cur_total)
            self.grids_nparameters += cur_total

    def initialize_latent_grids(self) -> DGrids:
        latent_grids = nn.ParameterList()
        for cur_size in self.size_per_latent: 
            latent_grids.append(nn.Parameter(torch.rand(cur_size)*0.1-0.05, requires_grad=True))
        return DGrids(latent_grids)

    def initialize_parameters(self) -> DParams:
        arm_param = self.arm.initialize_parameters()
        upsampling_param = self.upsampling.initialize_parameters()
        synthesis_param = self.synthesis.initialize_parameters()
        res = DParams(arm_param=arm_param,upsampling_param=upsampling_param,syn_param=synthesis_param)
        return res
    
    def initialize_parameters_map(self) -> None:
        self.arm.initialize_parameters_map()
        self.upsampling.initialize_parameters_map()
        self.synthesis.initialize_parameters_map()
    
    def to_device(self, device) -> None:
        self.non_zero_pixel_ctx_index = self.non_zero_pixel_ctx_index.to(device)
        #self.arm.to(device)
        #self.upsampling.to(device)
        #self.synthesis.to(device)

def _linear_schedule(initial_value: float, final_value: float, cur_itr: float, max_itr: float) -> float:
    assert cur_itr >= 0 and cur_itr <= max_itr, (
        f"Linear scheduling from 0 to {max_itr} iterations"
        " except to have a current iterations between those two values."
        f" Found cur_itr = {cur_itr}."
    )
    return cur_itr * (final_value - initial_value) / max_itr + initial_value

class DataOptim:
    
    def __init__(self, manager,phase_idx, warmup=False) -> None:
        if warmup:
            assert phase_idx == 0 or phase_idx ==1, ("only two phases availbale for warmup")
            self.phase = manager.preset.warmup.phases[phase_idx].training_phase
            self.candidates = manager.preset.warmup.phases[phase_idx].candidates
        else:
            assert phase_idx == 0 or phase_idx ==1 or phase_idx == 2, ("only three phases availbale for training")
            self.phase = manager.preset.all_phases[phase_idx]

    def init_solver_dp(self, dp:DParams)->None:
        op_module = ["ap","up","sp"] if "all" in self.phase.optimized_module else []
        if "arm" in self.phase.optimized_module: op_module.append("ap")
        if "upsampling" in self.phase.optimized_module: op_module.append("up")
        if "synthesis" in self.phase.optimized_module:op_module.append("sp")
        dp.init_solver(op_module,self.phase.lr,self.phase.max_itr,self.phase.freq_valid,self.phase.schedule_lr)
    
    def init_solver_dg(self,dg:DGrids|list[DGrids])->None:
        op_module = ["grids"] if "all" in self.phase.optimized_module or "latent" in self.phase.optimized_module else []
        if isinstance(dg,DGrids):
            dg.init_solver(op_module,self.phase.lr,self.phase.max_itr,self.phase.freq_valid,self.phase.schedule_lr)
        else:
            for dgo in dg:
                dgo.init_solver(op_module,self.phase.lr,self.phase.max_itr,self.phase.freq_valid,self.phase.schedule_lr)
        
    def produce_quant_param(self,iter):
        cur_softround_temperature = _linear_schedule(self.phase.softround_temperature[0],  self.phase.softround_temperature[1],  iter,  self.phase.max_itr, )
        cur_noise_parameter = _linear_schedule(self.phase.noise_parameter[0], self.phase.noise_parameter[1], iter, self.phase.max_itr)
        return cur_softround_temperature, cur_noise_parameter
    

class TrainingHelper:
    
    def __init__(self,dg:DGrids|list[DGrids], dp:DParams, gen:CoolChicEncoder) -> None:
        self.dg = dg if isinstance(dg,list) else [dg]
        self.dp = dp
        self.gen = gen
        self.iter = 0

    def to(self, device)->None:
        self.dp.to(device)
        self.gen.non_zero_pixel_ctx_index= self.gen.non_zero_pixel_ctx_index.to(device)
        #print(self.gen.non_zero_pixel_ctx_index.device)
        for dg in self.dg: dg.to(device)
            
    def empty_grad(self)->None:
        self.dp.empty_grad()
        for dg in self.dg: dg.empty_grad()
    
    def step(self)->None:
        self.iter += 1
        self.dp.step()
        for dg in self.dg: dg.step()
    
    def beat(self,loss):
        for dg in self.dg: dg.beat(loss)
        return self.dp.beat(loss)
       
    def load_best(self,device):
        self.dp.load_best(device)
        for dg in self.dg: dg.load_best(device)
        
    def lr_schedule(self):
        self.dp.lr_schedule.step()
        for dg in self.dg: dg.lr_schedule.step()
        
    def load_best_param(self,device):
        self.dp.load_best_param(device)
        for dg in self.dg: dg.load_best_param(device)
    
        
class TensorData(nn.Module):
    
    def __init__(self, lmbda=0.001, n_itr=20000, image_size=(512,512), device='cpu') -> None:
        super().__init__()
        self.param = CoolChicEncoderParameter(device=device)
        self.param.set_image_size(image_size)
        self.gen = CoolChicEncoder(self.param)
        self.manager = FrameEncoderManager(preset_name='c3x', start_lr=0.01, lmbda=lmbda,  n_loops=1,  n_itr=n_itr)
        self.init_parameter_map()
        
    
    def to(self,device):
        super().to(device)
        self.gen.to_device(device)
        return self
    
    def merge_Dgrids(self,dg:DGrids|list[DGrids]):
        if isinstance(dg,DGrids): grids = dg.pool['grids']
        else: grids = [torch.cat([dg[gid].pool['grids'][pid] for gid in range(len(dg))],dim=0) for pid in range(dg[0].pool['grids'].__len__())]
        return grids
    
    @torch.no_grad()
    def mimic_forward(self,dg:DGrids|list[DGrids], dp:DParams, quantizer_noise_type,quantizer_type, qa, qb):
        #print(quantizer_noise_type,quantizer_type,qa,qb)
        grids = self.merge_Dgrids(dg)
        output,noise = self.gen.forward_data(grids, dp.pool['up'], dp.pool['sp'],
            quantizer_noise_type, quantizer_type, qa,qb)      
        #print(noise)
        return output,noise
    
    @torch.no_grad()
    def forward_for_test(self,dg:DGrids|list[DGrids], dp:DParams):
        self.set_to_eval()
        grids = self.merge_Dgrids(dg)
        output,rate = self.gen.forward(grids,dp.pool['ap'], dp.pool['up'], dp.pool['sp'],
                                  quantizer_noise_type="none", quantizer_type="hardround", )
        max_dynamic = 2 ** (self.param.bitdepth) - 1
        decoded_image = (torch.round(output * max_dynamic)/ max_dynamic)
        return decoded_image,rate
    
    def forward(self,dg:DGrids|list[DGrids], dp:DParams,quantizer_noise_type,quantizer_type, qa, qb):
        grids = self.merge_Dgrids(dg)
        output = self.gen.forward(grids,dp.pool['ap'], dp.pool['up'], dp.pool['sp'],
            quantizer_noise_type, quantizer_type, qa, qb)      
        return output
    
    def forward_per_sample(self,dg:DGrids|list[DGrids], dp:DParams,noise,quantizer_noise_type,quantizer_type, qa, qb):
        grids = self.merge_Dgrids(dg)
        output = self.gen.forward_per_sample(grids,dp.pool['ap'], dp.pool['up'], dp.pool['sp'], noise, quantizer_noise_type,quantizer_type, qa, qb)      
        return output
    
    @torch.no_grad()
    def test(self, dg:DGrids|list[DGrids], dp:DParams, ref:torch.Tensor, loss_func):
        decoded_image, rate = self.forward_for_test(dg,dp)
        loss_fn_output = loss_func(decoded_image, rate, ref, lmbda=self.manager.lmbda)
        self.set_to_train()
        return loss_fn_output
    
    def set_to_train(self) -> None:
        self = self.train()
        self.coolchic_encoder = self.gen.train()

    def set_to_eval(self) -> None:
        self = self.eval()
        self.coolchic_encoder = self.gen.eval()
        
    def set_training_phase(self, dg:DGrids|list[DGrids], dp:DParams, phase_idx, warmup=False):
        dop = DataOptim(self.manager,phase_idx,warmup)
        dop.init_solver_dg(dg)
        dop.init_solver_dp(dp)
        return dop
            
    def pretrain(self, dg:DGrids|list[DGrids], dp:DParams, loss_func, ref):
    
        helper = TrainingHelper(dg,dp,self.gen)
        device = ref.device
        helper.to(device)
        self.set_to_train()
        start_time = time.time()
        cnt_record = 0
        encoder_logs_best = initial_encoder_logs = self.test(dg,dp,ref,loss_func)
        helper.beat(encoder_logs_best.loss)
        show_col = True
        qa,qb = self.dop.produce_quant_param(helper.iter)
        for cnt in range(self.dop.phase.max_itr):
            if cnt - cnt_record > self.dop.phase.patience:
                if self.dop.phase.schedule_lr:
                    helper.load_best(device)
                    cnt_record = cnt
                else:
                    break
            helper.empty_grad()
            #'''
            dt,noise = self.mimic_forward(dg,dp,self.dop.phase.quantizer_noise_type,self.dop.phase.quantizer_type,qa,qb)
            nout,nrate,_ = self.forward_per_sample(dg,dp,noise,self.dop.phase.quantizer_noise_type,self.dop.phase.quantizer_type,qa,qb)
            #print(torch.mean(torch.abs(nout-dt)))
            out = dt.clone().detach()
            out.requires_grad_()
            out.retain_grad()
            loss = torch.mean((out-ref)**2)
            loss.backward()
            rate = nrate.clone().detach()
            rate.requires_grad_()
            rate.retain_grad()
            rate_loss = torch.sum(rate) / (ref.shape[0]*ref.shape[2]*ref.shape[3]) * self.manager.lmbda
            rate_loss.backward()
            torch.autograd.backward([nout,nrate],[out.grad,rate.grad]) 
            #out,rate = self.forward(dg,dp,self.dop.phase.quantizer_noise_type,self.dop.phase.quantizer_type,qa,qb)
            #loss_function_output = loss_func(out,rate,ref, lmbda=self.manager.lmbda)
            #loss_function_output.loss.backward()
            helper.step()
            if ((cnt + 1) % self.dop.phase.freq_valid == 0) or (cnt + 1 == self.dop.phase.max_itr):
                self.manager.total_training_time_sec += time.time() - start_time
                start_time = time.time()
                encoder_logs = self.test(dg, dp, ref, loss_func)
                flag_new_record = False
                if helper.beat(encoder_logs.loss):
                    delta_psnr = encoder_logs.psnr_db - encoder_logs_best.psnr_db
                    delta_bpp = (encoder_logs.rate_latent_bpp - encoder_logs_best.rate_latent_bpp)
                    flag_new_record = delta_bpp < 0.001 or delta_psnr > 0.001
                if flag_new_record:
                    this_phase_psnr_gain = ( encoder_logs.psnr_db - initial_encoder_logs.psnr_db )
                    this_phase_bpp_gain = ( encoder_logs.rate_latent_bpp - initial_encoder_logs.rate_latent_bpp  )
                    log_new_record = f"{this_phase_bpp_gain:+6.3f} bpp " + f"{this_phase_psnr_gain:+6.3f} db"
                    encoder_logs_best = encoder_logs
                    cnt_record = cnt
                else:
                    log_new_record = ""
                additional_data = {"iter": f"{cnt}", "time": f"{self.manager.total_training_time_sec:.1f}",  "record": log_new_record,}
                print(
                    encoder_logs.pretty_string(
                        show_col_name=show_col,
                        additional_data=additional_data,
                    )
                )
                show_col=False
                qa,qb = self.dop.produce_quant_param(helper.iter)
                if self.dop.phase.schedule_lr:  
                    helper.lr_schedule()
                self.set_to_train()
        #exit()
        helper.load_best_param(device)
        return dg,dp, encoder_logs_best
    
    def init_parameter_map(self):
        self.gen.initialize_parameters_map()
    
    def produce_parameters(self,ngrids):
        return [self.gen.initialize_latent_grids() for _ in range(ngrids)], self.gen.initialize_parameters()
    
    
    def warmup(self, loss_fn, ref):
        device = 'cuda:0'
        _col_width = 14
        all_candidates = [ ]
        for id_candidate in range(self.manager.preset.warmup.phases[0].candidates):
            dg,dp = self.produce_parameters(ref.shape[0])
            all_candidates.append({"metrics": None, "id": id_candidate, "dp": dp, "dg":dg})
    
        for phase_idx in range(2):
            print(f'{"-" * 30}  Warm-up phase: {phase_idx:>2} {"-" * 30}')
            candidates = self.manager.preset.warmup.phases[phase_idx].candidates
            if phase_idx != 0:
                n_elements_to_remove = len(all_candidates) - candidates
                for _ in range(n_elements_to_remove):  all_candidates.pop()
            for i in range(candidates):
                cur_candidate = all_candidates[i]
                cur_id = cur_candidate.get("id")
                dp = cur_candidate.get("dp")
                dg = cur_candidate.get("dg")
                #self.to(device)
                self.dop = self.set_training_phase(dg,dp,phase_idx,True)
                print(f"\nCandidate n° {i:<2}, ID = {cur_id:<2}:"   + "\n-------------------------\n")
                dg_new,dp_new,loss_new = self.pretrain(dg,dp,loss_fn,ref)
                cur_candidate["dg"] = dg_new
                cur_candidate["dp"]  = dp_new
                cur_candidate["metrics"] = loss_new
                all_candidates[i] = cur_candidate
            all_candidates = sorted(all_candidates, key=lambda x: x.get("metrics").loss)
            s = "\n\nPerformance at the end of the warm-up phase:\n\n"
            s += f'{"ID":^{6}}|{"loss":^{_col_width}}|{"rate_bpp":^{_col_width}}|{"psnr_db":^{_col_width}}|\n'
            s += f'------|{"-" * _col_width}|{"-" * _col_width}|{"-" * _col_width}|\n'
            for candidate in all_candidates:
                s = s + f'{candidate.get("id"):^{6}}|' +  f'{candidate.get("metrics").loss.item() * 1e3:^{_col_width}.4f}|'
                s = s + f'{candidate.get("metrics").rate_latent_bpp:^{_col_width}.4f}|' +  f'{candidate.get("metrics").psnr_db:^{_col_width}.4f}|'
                s += "\n"
            print(s)
        print("Warm-up is done!")
        print(f'Winner ID : {all_candidates[0].get("id")}\n')
        return all_candidates[0].get("dg"),all_candidates[0].get("dp")
        
    def pretrain_whole(self,ref,loss_fn):
        dg,dp = self.warmup(loss_fn,ref)
        return dg,dp
        for phase_idx in range(3):
            self.dop = self.set_training_phase(dg,dp,phase_idx)
            dg,dp,_ = self.pretrain(dg,dp,loss_fn,ref)
        return dg,dp
    
    def test_pretrain_whole(self,ref,loss_fn,dg,dp):
        for phase_idx in range(3):
            self.dop = self.set_training_phase(dg,dp,phase_idx)
            dg,dp,_ = self.pretrain(dg,dp,loss_fn,ref)
        return dg,dp
        
    

