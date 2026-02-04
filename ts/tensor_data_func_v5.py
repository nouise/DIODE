import math
import typing
from dataclasses import dataclass, field, fields
from typing import Any, Dict, List, Optional, OrderedDict, Tuple, TypedDict
from torch import nn, Tensor
import torch
from torch.nn.utils import clip_grad_norm_
import time
import copy
from ts.core.presets import TrainerPhase, WarmupPhase
from ts.core.arm_func import (Arm, _get_neighbor,  _get_non_zero_pixel_ctx_index,  _laplace_cdf,)
from ts.core.quantizer import (POSSIBLE_QUANTIZATION_NOISE_TYPE, POSSIBLE_QUANTIZER_TYPE,  quantize,)
from ts.core.synthesis_func import Synthesis
from ts.core.upsampling_func import Upsampling
from ts.core.manager import FrameEncoderManager
from ts.core.parameters import CoolChicEncoderParameter, MParams

        
class DPParams(MParams):
    
    def __init__(self, latent_grids:nn.ParameterList=None,
                 arm_param:nn.ParameterList=None, 
                 upsampling_param:nn.ParameterList=None, 
                 syn_param:nn.ParameterList=None) -> None:
        super(DPParams,self).__init__()
        self.pool['grids'] = latent_grids
        self.pool['ap'] = arm_param
        self.pool['up'] = upsampling_param
        self.pool['sp'] = syn_param
        self.bpp = 24





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
        #self.non_zero_pixel_ctx_index = self.non_zero_pixel_ctx_index.to(flat_decoder_side_latent.device)
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
    ) -> Tuple[Tensor,Tensor]:
        #print('here forward')
        size_per_latent_flat = [latent_i.numel() for latent_i in latent_grids]
        size_per_latent = [latent_i.shape for latent_i in latent_grids]
        encoder_side_flat_latent = torch.cat([latent_i.view(-1) for latent_i in latent_grids]).contiguous()
        flat_decoder_side_latent = quantize(encoder_side_flat_latent * self.encoder_gains,   quantizer_noise_type,  quantizer_type, soft_round_temperature, noise_parameter, noise_out=noise)
        if AC_MAX_VAL != -1:  flat_decoder_side_latent = torch.clamp(flat_decoder_side_latent, -AC_MAX_VAL, AC_MAX_VAL + 1)
        decoder_side_latent = [ts.view(sz) for ts,sz in zip(torch.split(flat_decoder_side_latent,size_per_latent_flat),size_per_latent)]
        #self.non_zero_pixel_ctx_index = self.non_zero_pixel_ctx_index.to(flat_decoder_side_latent.device)
        flat_context = torch.cat([ _get_neighbor(spatial_latent_i, self.mask_size, self.non_zero_pixel_ctx_index)   for spatial_latent_i in decoder_side_latent ],  dim=0)
        flat_latent = flat_decoder_side_latent
        flat_mu, flat_scale, _ = self.arm(flat_context,arm_param)
        proba = torch.clamp_min( _laplace_cdf(flat_latent + 0.5, flat_mu, flat_scale) - _laplace_cdf(flat_latent - 0.5, flat_mu, flat_scale),   min=2**-16, )
        flat_rate = -torch.log2(proba)
        synthesis_output = self.synthesis(self.upsampling(decoder_side_latent,upsampling_param),syn_param)
        return synthesis_output,  flat_rate
    
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
            cur_size = (self.param.batch_size, c_grid, h_grid, w_grid)
            self.size_per_latent.append(cur_size)
            cur_total = self.param.batch_size * c_grid * h_grid * w_grid
            self.size_per_latent_flat.append(cur_total)
            self.grids_nparameters += cur_total

    def initialize_latent_grids(self) -> nn.ParameterList:
        latent_grids = nn.ParameterList()
        for cur_size in self.size_per_latent: 
            latent_grids.append(nn.Parameter(torch.rand(cur_size)*0.1-0.05, requires_grad=True))
        return latent_grids

    def initialize_parameters(self) -> DPParams:
        arm_param = self.arm.initialize_parameters()
        upsampling_param = self.upsampling.initialize_parameters()
        synthesis_param = self.synthesis.initialize_parameters()
        grids = self.initialize_latent_grids()
        res = DPParams(latent_grids=grids,arm_param=arm_param,upsampling_param=upsampling_param,syn_param=synthesis_param)
        return res
    
    def initialize_parameters_map(self) -> None:
        self.arm.initialize_parameters_map()
        self.upsampling.initialize_parameters_map()
        self.synthesis.initialize_parameters_map()
    
    def to_device(self, device) -> None:
        self.non_zero_pixel_ctx_index = self.non_zero_pixel_ctx_index.to(device)

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
            self.phase:WarmupPhase  = manager.preset.warmup.phases[phase_idx].training_phase
            self.candidates = manager.preset.warmup.phases[phase_idx].candidates
        else:
            assert phase_idx == 0 or phase_idx ==1 or phase_idx == 2, ("only three phases availbale for training")
            self.phase:TrainerPhase = manager.preset.all_phases[phase_idx]

    def init_solver(self, dp:DPParams)->None:
        op_module = ["ap","up","sp","grids"] if "all" in self.phase.optimized_module else []
        if "arm" in self.phase.optimized_module: op_module.append("ap")
        if "upsampling" in self.phase.optimized_module: op_module.append("up")
        if "synthesis" in self.phase.optimized_module:op_module.append("sp")
        if "latent" in self.phase.optimized_module:op_module.append("grids")
        dp.init_solver(op_module,self.phase.lr,self.phase.max_itr,self.phase.freq_valid,self.phase.schedule_lr)
        
    def produce_quant_param(self,iter):
        cur_softround_temperature = _linear_schedule(self.phase.softround_temperature[0],  self.phase.softround_temperature[1],  iter,  self.phase.max_itr, )
        cur_noise_parameter = _linear_schedule(self.phase.noise_parameter[0], self.phase.noise_parameter[1], iter, self.phase.max_itr)
        return cur_softround_temperature, cur_noise_parameter
 
class TrainParam:
    
    def __init__(self, phase,ldb) -> None:
        self.qa = None
        self.qb = None
        self.phase = phase
        self.noise_type = self.phase.quantizer_noise_type
        self.quant_type = self.phase.quantizer_type
        self.ldb = ldb
        self.step(0)
        
    def start(self):
        self.record = 0
        
    def load_best(self,current)->bool:
        if current-self.record > self.phase.patience and self.phase.schedule_lr:
            self.record = current
            print('reload current best')
            return True
        return False
    
    def record_new(self,current):
        self.record = current 
        
    def step(self, iter)->None:
        self.qa = _linear_schedule(self.phase.softround_temperature[0],  self.phase.softround_temperature[1],  iter,  self.phase.max_itr, )
        self.qb = _linear_schedule(self.phase.noise_parameter[0], self.phase.noise_parameter[1], iter, self.phase.max_itr)
       

class TensorData(nn.Module):
    
    def __init__(self, channel = 1, image_size=(512,512), batch_size=1, device='cpu') -> None:
        super().__init__()
        layers = ['40-1-linear-relu',f'{channel}-1-linear-none',f'{channel}-3-residual-relu',f'{channel}-3-residual-none']
        min_size = min(image_size[0],image_size[1])
        nft = int(math.log2(min_size))
        n_ft_per_grid = [1,1,1,1,1,1] if min_size>=64 else [1 for _ in range(nft+1)]
        self.param = CoolChicEncoderParameter(layers_synthesis=layers,n_ft_per_res=n_ft_per_grid,device=device,batch_size=batch_size)
        self.param.set_image_size(image_size)
        self.gen = CoolChicEncoder(self.param)
        self.init_parameter_map()
        self.device = device
        self.gen.to_device(device)
        
    def set_param(self,dp:DPParams):
        self.dp = dp
        
    def to(self, device)->None:
        self.dp.to(device)
        
    def to_run(self)->None:
        self.dp.to(self.device)
    
    def empty_grad(self)->None:
        self.dp.empty_grad()
    
    def step(self)->None:
        self.dp.step()
    
    def record(self):
        self.dp.record()
       
    def load_best(self,device):
        self.dp.load_best(device)
        
    def lr_schedule(self):
        self.dp.lr_schedule.step()
        
    def load_best_param(self,device):
        self.dp.load_best_param(device)
    
    @torch.no_grad()
    def mimic_forward(self, qp:TrainParam):
        output,noise = self.gen.forward_data(self.dp.pool['grids'], self.dp.pool['up'], self.dp.pool['sp'],
            qp.noise_type,qp.quant_type,qp.qa,qp.qb)      
        return output,noise
    
    @torch.no_grad()
    def forward_for_test(self):
        self.set_to_eval()
        output,rate = self.gen.forward(self.dp.pool['grids'],self.dp.pool['ap'], self.dp.pool['up'], self.dp.pool['sp'],
                                  quantizer_noise_type="none", quantizer_type="hardround", )
        max_dynamic = 2 ** (self.param.bitdepth) - 1
        decoded_image = (torch.round(output * max_dynamic)/ max_dynamic)
        return decoded_image,rate
    
    def forward(self,qp:TrainParam):
        output = self.gen.forward(self.dp.pool['grids'],self.dp.pool['ap'], self.dp.pool['up'], self.dp.pool['sp'],
            qp.noise_type,qp.quant_type,qp.qa,qp.qb)      
        return output
    
    def forward_per_sample(self,qp:TrainParam,noise:Tensor):
        output = self.gen.forward_per_sample(self.dp.pool['grids'],self.dp.pool['ap'], self.dp.pool['up'], self.dp.pool['sp'], noise,
                                            qp.noise_type,qp.quant_type,qp.qa,qp.qb)      
        return output
    
    @torch.no_grad()
    def test(self, qp:TrainParam,ref:torch.Tensor, loss_func):
        decoded_image, rate = self.forward_for_test()
        loss_fn_output = loss_func(decoded_image, rate, ref, lmbda=qp.ldb)
        self.set_to_train()
        return loss_fn_output
    
    def set_to_train(self) -> None:
        self.gen = self.gen.train()

    def set_to_eval(self) -> None:
        #self = self.eval()
        self.gen = self.gen.eval()
        
    def init_parameter_map(self):
        self.gen.initialize_parameters_map()

    def produce_parameters(self):
        return self.gen.initialize_parameters()


class TrainingHelper:
    
    def __init__(self, lmbda, max_iters) -> None:
        self.manager = FrameEncoderManager(preset_name='c3x', start_lr=0.01, lmbda=lmbda,  n_loops=1,  n_itr=max_iters)
        self.loss = None
        self.lmdba = lmbda
        
    def set_training_phase(self, phase_idx, warmup=False):
        self.dop = DataOptim(self.manager,phase_idx,warmup)
        self.qp = TrainParam(self.dop.phase,self.manager.lmbda)
        self.qp.start()
        self.loss = 1e9
     
    def init_solver(self, dp:DPParams):
        self.dop.init_solver(dp)
     
    def beat(self,loss,cnt):
        if (self.loss - loss) / self.loss > 0.0001:
            print(f'new record: {loss:.5} in epoch {cnt}, last record: {self.loss:.5} in epoch {self.qp.record}')
            self.loss = loss
            self.qp.record_new(cnt)
            return True
        return False
    
    def validate(self,cnt):
        if ((cnt + 1) % self.dop.phase.freq_valid == 0) or (cnt + 1 == self.dop.phase.max_itr):
            self.qp.step(cnt)
            return True
        return False
        
        

    
def pretrain(op:TensorData, dp:DPParams, loss_func, ref, helper:TrainingHelper):
    op.set_param(dp)
    op.set_to_train()
    op.to_run()
    ref = ref.to(op.device)
    start_time = time.time()
    encoder_logs_best = initial_encoder_logs = op.test(helper.qp,ref,loss_func)
    helper.beat(encoder_logs_best.loss,0)
    show_col = True
    for cnt in range(helper.dop.phase.max_itr):
        if helper.qp.load_best(cnt):
            if helper.dop.phase.schedule_lr:
                op.load_best(op.device)
            else:
                break
        op.empty_grad()
        out,rate = op.forward(helper.qp)
        loss_function_output = loss_func(out,rate,ref, lmbda=helper.qp.ldb)
        loss_function_output.loss.backward()
        op.step()
        if helper.validate(cnt):
            helper.manager.total_training_time_sec += time.time() - start_time
            start_time = time.time()
            encoder_logs = op.test(helper.qp, ref, loss_func)
            if helper.beat(encoder_logs.loss,cnt):
                op.record()
                this_phase_psnr_gain = ( encoder_logs.psnr_db - initial_encoder_logs.psnr_db )
                this_phase_bpp_gain = ( encoder_logs.rate_latent_bpp - initial_encoder_logs.rate_latent_bpp  )
                log_new_record = f"{this_phase_bpp_gain:+6.3f} bpp " + f"{this_phase_psnr_gain:+6.3f} db"
                encoder_logs_best = encoder_logs
            else:
                log_new_record = ""
            additional_data = {"iter": f"{cnt}", "time": f"{helper.manager.total_training_time_sec:.1f}",  "record": log_new_record,}
            print(
                encoder_logs.pretty_string(
                    show_col_name=show_col,
                    additional_data=additional_data,
                )
            )
            show_col=False
            if helper.dop.phase.schedule_lr:  op.lr_schedule()
            op.set_to_train()
    op.load_best_param(op.device)
    return op.dp, encoder_logs_best

def warmup(loss_fn, ref, op:TensorData, helper:TrainingHelper):
    
    _col_width = 14
    all_candidates = [ ]
    for id_candidate in range(helper.manager.preset.warmup.phases[0].candidates):
        dp = op.produce_parameters()
        all_candidates.append({"metrics": None, "id": id_candidate, "dp": dp})

    for phase_idx in range(2):
        print(f'{"-" * 30}  Warm-up phase: {phase_idx:>2} {"-" * 30}')
        candidates = helper.manager.preset.warmup.phases[phase_idx].candidates
        if phase_idx != 0:
            n_elements_to_remove = len(all_candidates) - candidates
            for _ in range(n_elements_to_remove):  all_candidates.pop()
        for i in range(candidates):
            cur_candidate = all_candidates[i]
            cur_id = cur_candidate.get("id")
            dp = cur_candidate.get("dp")
            helper.set_training_phase(phase_idx,True)
            helper.init_solver(dp)
            print(f"\nCandidate n° {i:<2}, ID = {cur_id:<2}:"   + "\n-------------------------\n")
            dp_new,loss_new = pretrain(op,dp,loss_fn,ref,helper)
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
    return all_candidates[0].get("dp")
    
def pretrain_whole(op,ref,loss_fn,lmb,max_iter,only_warmup=True):
    helper = TrainingHelper(lmb, max_iter)
    dp = warmup(loss_fn,ref,op,helper)
    #return dp
    if only_warmup: return dp
    for phase_idx in range(3):
        helper.set_training_phase(phase_idx)
        helper.init_solver(dp)
        dp,_ = pretrain(op,dp,loss_fn,ref,helper)
    return dp

def test_pretrain_whole(op, ref, loss_fn, dp,lmb,max_iter):
    helper = TrainingHelper(lmb, max_iter)
    for phase_idx in range(3):
        helper.set_training_phase(phase_idx)
        dp,_ = pretrain(op,dp,loss_fn,ref,helper)
    return dp
        
    

