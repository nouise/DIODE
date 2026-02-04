import torch
import random
from ts.tensor_data_func_v6 import TensorData, DPParams,TrainingHelper,_linear_schedule, TrainParam, pretrain_whole
import concurrent.futures
import time
from torch import nn
from ts.training import loss_function

def all_to(param:DPParams,device):
    param = param.to(device)
    
def all_record(param:DPParams):
    return param.record()
    
def all_load_best(param:DPParams,device):
    param.load_best(device)
    
def all_load_best_param(param:DPParams, device):
    param.load_best_param(device)
    
def all_schedul_lr(param:DPParams):
    param.lr_schedule.step()

def run_model(model:TensorData, qp:TrainParam, param_device, target_device):
    #print(model.dp.pool['ap'][0].device,model.device)
    model.to_run()
    y,noise,rt = model.mimic_forward(qp)
    npiexls = y.shape[0] * y.shape[-1]* y.shape[-2]
    rt = rt.sum().item() / npiexls
    y = y.to(target_device)
    noise = noise.to(target_device)
    model.to(param_device)    
    return y,noise,rt

def run_model_test(model:TensorData, param_device, target_device):
    model.to_run()
    try:
        y,rt = model.forward_for_test()
    except Exception as e:
        print("run_error:run_model_test")
        print(str(e))
    npiexls = y.shape[0] * y.shape[-1]*y.shape[-2]
    #print("why:")
    rt = torch.sum(rt).item() / npiexls

    y = y.to(target_device)
    model.to(param_device)
    return y,rt

def run_model_backward(model:TensorData, qp:TrainParam, data_grad, noise, target_device,lr_scale,ldb_scale,epoch,cnt_param=0):
    running_device = model.device
    noise = noise.to(running_device)
    data_grad = data_grad.to(running_device)
    model.empty_grad()
    y,rt = model.forward_per_sample(qp,noise)
    drt = rt.clone().detach().requires_grad_()
    drt.retain_grad()
    npiexls = data_grad.shape[0] * data_grad.shape[-1]*data_grad.shape[-2]
    bpp = drt.sum() / npiexls
    rt_loss = bpp * qp.ldb 
    rt_loss.backward()
    if epoch%500 == 0 and cnt_param==0:
        print(f"sum of grad:{torch.sum(torch.abs(data_grad))},{torch.sum(torch.abs(drt.grad))},scale factor are:{lr_scale},{ldb_scale}")
        if model.dp.lr_schedule:
                print(f"step current lr:{model.dp.lr_schedule.state_dict()['_last_lr'][0]}")
    try:
        torch.autograd.backward([y,rt],[data_grad*lr_scale,drt.grad*ldb_scale])
        if epoch%50==0 and cnt_param==0:
            model.step(True)
        else:
            model.step(False)
    except Exception as e:
        print(f"An error occurred: {e}")
    model.to(target_device)
    return True

def run_warmup(model,ref,loss_fn,lmdb,max_iter,warmup_only):
    running_device = model.device 
    ref = ref.to(running_device)
    param = pretrain_whole(model,ref,loss_fn,lmdb,max_iter,warmup_only)
    return param,ref

def run_model_qat(model: TensorData, qp: TrainParam, quant_config, quantizer_type, soft_round_temperature, param_device, target_device):
    """
    量化版本的 run_model（用于前向生成合成数据 - train 模式）
    
    Returns:
        y: 解码图像
        noise: latent 噪声
        rt: latent 码率（bpp）
        nn_int_all: NN 参数整数符号
    """
    model.to_run()
    y, noise, rt, nn_int_all = model.mimic_forward_qat(qp, quant_config, quantizer_type, soft_round_temperature)
    
    npixels = y.shape[0] * y.shape[-1] * y.shape[-2]
    rt = rt.sum().item() / npixels
    
    y = y.to(target_device)
    if noise is not None:
         noise = noise.to(target_device)
    if nn_int_all is not None:
        nn_int_all = nn_int_all.to(target_device)
    
    model.to(param_device)
    return y, noise, rt, nn_int_all


def run_model_test_qat(model: TensorData, quant_config, quantizer_type, param_device, target_device):
    """
    量化版本的 run_model_test（用于测试/评估 - test 模式）
    
    Args:
        model: TensorData 模型
        quant_config: 量化配置
        quantizer_type: 量化器类型（test 模式固定用 'hardround'）
        param_device: 参数存储设备
        target_device: 目标输出设备
    
    Returns:
        y: 解码图像
        rt: latent 码率（bpp）
        nn_int_all: NN 参数整数符号
    """
    model.to_run()
    try:
        y, rt, nn_int_all = model.forward_for_test_qat(quant_config, quantizer_type='hardround', soft_round_temperature=0.0)
    except Exception as e:
        print("run_error: run_model_test_qat")
        print(str(e))
        raise
    
    npixels = y.shape[0] * y.shape[-1] * y.shape[-2]
    rt = torch.sum(rt).item() / npixels
    
    y = y.to(target_device)
    if nn_int_all is not None:
        nn_int_all = nn_int_all.to(target_device)
    
    model.to(param_device)
    return y, rt, nn_int_all


def run_model_backward_qat(
    model: TensorData, 
    qp: TrainParam, 
    quant_config,
    quantizer_type,
    soft_round_temperature,
    data_grad, 
    noise, 
    nn_int_all,
    target_device,
    lr_scale, 
    ldb_scale, 
    ldb_nn,
    alpha_nn,
    epoch, 
    cnt_param=0
):
    """
    量化版本的 run_model_backward（支持 NN 参数码率优化）
    
    Args:
        ldb_scale: latent 码率权重缩放
        ldb_nn: NN 参数码率权重
        alpha_nn: NN proxy 自动缩放系数
        nn_int_all: NN 参数的整数符号（来自 forward）
    """
    running_device = model.device
    if noise is not None:
        noise = noise.to(running_device)
    data_grad = data_grad.to(running_device)
    if nn_int_all is not None:
        nn_int_all = nn_int_all.to(running_device)
    
    model.empty_grad()
    
    # 前向传播（带量化，使用保存的 noise，有梯度）
    y, rt, nn_int_new = model.forward_per_sample_qat(qp, noise, quant_config, quantizer_type, soft_round_temperature)
    
    # latent 码率损失（使用中间变量 drt）
    drt = rt.clone().detach().requires_grad_()
    drt.retain_grad()
    npixels = data_grad.shape[0] * data_grad.shape[-1] * data_grad.shape[-2]
    bpp = drt.sum() / npixels
    rt_loss = bpp * qp.ldb
    
    # NN 参数码率代理（方案B：三次 backward）
    if nn_int_new is not None and ldb_nn > 0:
        # 原始 nn_proxy（用于最终反传）
        nn_proxy_raw = torch.mean(torch.log1p(torch.abs(nn_int_new)))
        
        # 创建中间变量 d_nn_proxy（类似 drt 的处理方式）
        d_nn_proxy = nn_proxy_raw.clone().detach().requires_grad_()
        d_nn_proxy.retain_grad()
        
        # 通过中间变量计算缩放和损失
        nn_proxy_scaled = d_nn_proxy * alpha_nn
        nn_loss = nn_proxy_scaled * ldb_nn
        
        # 调试：检查初始状态
        if epoch % 10 == 0 and cnt_param == 0:
            print(f"[DEBUG-BEFORE] y.requires_grad={y.requires_grad}, y.grad_fn={y.grad_fn is not None}")
            print(f"[DEBUG-BEFORE] rt.requires_grad={rt.requires_grad}, rt.grad_fn={rt.grad_fn is not None}")
            print(f"[DEBUG-BEFORE] nn_proxy_raw.requires_grad={nn_proxy_raw.requires_grad}, grad_fn={nn_proxy_raw.grad_fn is not None}")
        
        # 第一次 backward：填充 drt.grad（保留计算图）
        rt_loss.backward(retain_graph=True)
        
        # 第二次 backward：填充 d_nn_proxy.grad（保留计算图）
        nn_loss.backward(retain_graph=True)
        
        # 调试：检查中间变量状态
        if epoch % 10 == 0 and cnt_param == 0:
            print(f"[DEBUG-AFTER] drt.grad exists={drt.grad is not None}, sum={torch.sum(torch.abs(drt.grad)).item() if drt.grad is not None else 0:.4g}")
            print(f"[DEBUG-AFTER] d_nn_proxy.grad exists={d_nn_proxy.grad is not None}, value={d_nn_proxy.grad.item() if d_nn_proxy.grad is not None else 0:.4g}")
            print(f"[DEBUG-AFTER] y.requires_grad={y.requires_grad}, y.grad_fn={y.grad_fn is not None}")
            print(f"[DEBUG-AFTER] rt.requires_grad={rt.requires_grad}, rt.grad_fn={rt.grad_fn is not None}")
            print(f"[DEBUG-AFTER] nn_proxy_raw.requires_grad={nn_proxy_raw.requires_grad}, grad_fn={nn_proxy_raw.grad_fn is not None}")
        
        # 监控日志
        if epoch % 10 == 0 and cnt_param == 0:
            print(f"[QAT] epoch={epoch} | data_grad_sum={torch.sum(torch.abs(data_grad)):.4g} | "
                  f"drt_grad_sum={torch.sum(torch.abs(drt.grad)):.4g} | "
                  f"lr_scale={lr_scale:.4g} | ldb_scale={ldb_scale:.4g}")
            print(f"      nn_proxy={nn_proxy_raw.item():.4g} | nn_proxy_scaled={nn_proxy_scaled.item():.4g} | "
                  f"alpha_nn={alpha_nn:.4g} | ldb_nn={ldb_nn:.4g}")
            print(f"      d_nn_proxy_grad={d_nn_proxy.grad.item():.4g}")
            if model.dp.lr_schedule:
                print(f"      current_lr={model.dp.lr_schedule.state_dict()['_last_lr'][0]}")
        
        # 第三次 backward：一次性注入三个梯度
        try:
            torch.autograd.backward([y, rt, nn_proxy_raw], 
                                  [data_grad * lr_scale, drt.grad * ldb_scale, d_nn_proxy.grad])
            if epoch % 50 == 0 and cnt_param == 0:
                model.step(True)
            else:
                model.step(False)
        except Exception as e:
            print(f"[QAT-ERROR] backward failed: {e}")
    else:
        # 没有 QAT 时的原始逻辑
        rt_loss.backward()
        
        if epoch % 500 == 0 and cnt_param == 0:
            print(f"[QAT] epoch={epoch} | data_grad_sum={torch.sum(torch.abs(data_grad)):.4g} | "
                  f"drt_grad_sum={torch.sum(torch.abs(drt.grad)):.4g} | "
                  f"lr_scale={lr_scale:.4g} | ldb_scale={ldb_scale:.4g}")
            if model.dp.lr_schedule:
                print(f"      current_lr={model.dp.lr_schedule.state_dict()['_last_lr'][0]}")
        
        try:
            torch.autograd.backward([y, rt], [data_grad * lr_scale, drt.grad * ldb_scale])
            if epoch % 50 == 0 and cnt_param == 0:
                model.step(True)
            else:
                model.step(False)
        except Exception as e:
            print(f"[QAT-ERROR] backward failed: {e}")
    
    model.to(target_device)
    return True

class ThreadPoolManager:
    def __init__(self, max_workers=3):
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        self.futures = {}
        print(f"Thread pool created with {max_workers} workers.")

    def clear_futures(self):
        self.futures = {}
        
    def __del__(self):
        self.shutdown()

    def submit_task(self, thread_key, func, *args, **kwargs):
        future = self.executor.submit(func, *args, **kwargs)
        self.futures[future] = thread_key 
        return future

    def all_tasks_done(self):
        all_done = all(future.done() for future in self.futures.keys())
        while not all_done:
            time.sleep(0.001)
            all_done = all(future.done() for future in self.futures.keys())
    def shutdown(self, wait=True):
        self.executor.shutdown(wait=wait)
        print("Thread pool has been shut down.")

class TensorPool:
    
    def __init__(self,nclass,slice_size,sample_per_class:list[int], cuda_devices_idxs:list[int], nthread = 4,
                    ldb=0.001,img_size=(512,512),max_iter=50000,channel=3,lr=0.01,layers_v="v5",arm=32,dim=4,freeze_modules=None)->None:
        print(f"welcome to tensor pool, lr:{lr}, ldb:{ldb},channel:{channel},img_size:{img_size},cuda_devices_idxs:{cuda_devices_idxs}")
        if freeze_modules:
            print(f"Freeze modules: {freeze_modules}")
        self.freeze_modules = freeze_modules if freeze_modules else []
        self.nclass = nclass
        self.max_iter = max_iter
        self.slice_size = slice_size
        self.slice_per_class = [(ln+slice_size-1)//slice_size for ln in sample_per_class]
        self.cuda_devices_idx = cuda_devices_idxs
        self.sample_per_class = sample_per_class
        self.nsample = sum(self.sample_per_class)
        self.slice_pool = {f'{cidx}_{sidx}':{'param':DPParams()} for cidx in range(nclass) for sidx in range(self.slice_per_class[cidx]) }
        self.key_list = [f'{cidx}_{sidx}' for cidx in range(nclass) for sidx in range(self.slice_per_class[cidx])]
        map_func = lambda sidx,cidx: slice_size if sidx<self.slice_per_class[cidx] -1 else self.sample_per_class[cidx] % slice_size
        self.slice_nums = [map_func(sidx,cidx) for cidx in range(nclass) for sidx in range(self.slice_per_class[cidx])]
        self.label = torch.Tensor([k for k,nsample in enumerate(sample_per_class) for _ in range(nsample)]).type(torch.long)
        self.target_device= f'cuda:{(cuda_devices_idxs[0])}'
        self.nthread = nthread
        self.worker_pool = {tidx: TensorData(image_size=img_size,channel=channel,device='cuda:{}'.format(cuda_devices_idxs[tidx%len(cuda_devices_idxs)]),version=layers_v,arm=arm,dim=dim) for tidx in range(sum(self.slice_per_class))}
        self.helper = TrainingHelper(ldb,max_iter,lr)
        self.executor = ThreadPoolManager(nthread)
        self.ldb = ldb
        self.bidx = 0
        self.epoch = 0
        self.testing = False
        self.data_pool = None
        self.testing_data_redeay = False
        self.gain = len(self.key_list)

    
    def test(self):
        self.testing = True
        
    def train(self):
        self.testing = False
        self.testing_data_redeay = False
        
    def free_model(self):
        self.pool_idx = -1
    
    def get_model(self):
        self.pool_idx += 1
        return  self.worker_pool[self.pool_idx]
    
    def record(self):
        for pk in self.slice_pool.keys(): all_record(self.slice_pool[pk]['param'])
    
    def load_best(self):
        if not self.helper.dop.phase.schedule_lr: return
        if not self.helper.exist_best: return
        for pk in self.slice_pool.keys(): all_load_best(self.slice_pool[pk]['param'],'cpu')
    
    def load_best_param(self):
        for pk in self.slice_pool.keys(): all_load_best_param(self.slice_pool[pk]['param'],'cpu')
        
    def schedule_lr(self):
        for pk in self.slice_pool.keys(): all_schedul_lr(self.slice_pool[pk]['param'])
                
    def set_training_phase(self, phase, freeze_latent=None):
        """
        设置训练阶段
        
        Args:
            phase: 阶段索引
            freeze_latent: 是否冻结latent（None时自动从self.freeze_modules推断）
        """
        self.helper.set_training_phase(
            phase, 
            freeze_latent=freeze_latent,
            freeze_modules=self.freeze_modules
        )
        self.helper.qp.start()
        
    def init_solvers(self):
        for pk in self.slice_pool.keys(): 
            self.helper.init_solver(self.slice_pool[pk]['param'], freeze_modules=self.freeze_modules)
    
    def get_data(self):
        with torch.no_grad():
            if self.testing: self.forward_test()
            else: self.forward_data()
            #print(f"self.data_pool{list(self.data_pool.keys())}")
            data = [self.data_pool[pk] for pk in self.key_list]
            rt = [self.rate_pool[pk] for pk in self.key_list]
        self.data = torch.nn.Parameter(torch.cat(data,dim=0),requires_grad=True)
        return self.data, self.label, sum(rt)/len(rt)
    
    def get_data_v2(self, quant_config_map, quantizer_type='hardround', soft_round_temperature=0.0):
        """
        QAT版本的get_data，包装forward并负责数据拼接和码率计算
        
        Args:
            quant_config_map: 量化配置字典
            quantizer_type: 量化器类型
            soft_round_temperature: soft-round温度
        
        Returns:
            data: 拼接后的合成数据
            label: 标签
            latent_bpp: latent平均码率
            nn_bpp: NN参数平均码率
            total_bpp: 总平均码率
        """
        from ts.core.quantizemodel import _estimate_nn_bits_expgolomb
        
        with torch.no_grad():
            # 根据模式选择前向传播方法
            if self.testing:
                self.forward_test_qat(quant_config_map, quantizer_type='hardround', soft_round_temperature=0.0)
            else:
                self.forward_data_qat(quant_config_map, quantizer_type, soft_round_temperature)
            
            data = [self.data_pool[pk] for pk in self.key_list]
            latent_rt = [self.rate_pool[pk] for pk in self.key_list]
            
            # 计算 NN 参数码率（使用 exp-golomb 估计）
            nn_bpp_list = []
            for pk in self.key_list:
                param = self.slice_pool[pk]['param']
                quant_config = quant_config_map.get(pk, None)
                
                if quant_config is not None:
                    # 获取当前 pk 对应的 npixels
                    npixels = data[self.key_list.index(pk)].shape[0] * data[self.key_list.index(pk)].shape[-1] * data[self.key_list.index(pk)].shape[-2]
                    
                    nn_bits = 0.0
                    for pool_key in ['ap', 'up', 'sp']:
                        nn_bits += _estimate_nn_bits_expgolomb(
                            param.pool[pool_key],
                            quant_config[pool_key]['q_step'],
                            quant_config[pool_key]['expgol_cnt']
                        )
                    nn_bpp_list.append(nn_bits / npixels)
                else:
                    nn_bpp_list.append(0.0)
        
        self.data = torch.nn.Parameter(torch.cat(data, dim=0), requires_grad=True)
        latent_bpp = sum(latent_rt) / len(latent_rt)
        nn_bpp = sum(nn_bpp_list) / len(nn_bpp_list)
        total_bpp = latent_bpp + nn_bpp
        
        return self.data, self.label, latent_bpp, nn_bpp, total_bpp
    
    
    @torch.no_grad()
    def fill_data_diff(self):
        grads = torch.split(self.data.grad,self.slice_nums)
        #print(grads[0][0])
        for id, pk in enumerate(self.key_list):
            #print(id,pk)
            self.data_pool[pk].grad = grads[id]
    
    def init_from_data(self,ref_dict):
        '''ref_dict is a dict with a key for each class and the value is the tensor for the reference image in that class'''
        self.free_model()
        try:
            self.executor.clear_futures()
            for pk in ref_dict:
                nslice = self.slice_per_class[pk]
                for sidx in range(nslice):
                    ref = ref_dict[pk][sidx*self.slice_size:sidx*self.slice_size+self.slice_size]
                    current_model = self.get_model()
                    skey = f'{pk}_{sidx}'
                    self.executor.submit_task(skey,run_warmup,current_model,ref,loss_function,self.ldb,5000,True)
            self.executor.all_tasks_done()
            self.free_model()
            for future in self.executor.futures.keys():
                skey = self.executor.futures[future]
                param,ref = future.result()
                self.slice_pool[skey]['param'].set_params(param.get_params(),'cpu')
                #torch.save({'param':param.get_params(),'ref':ref},f'./mnist/m2_{skey}.pt')
                #mdict = torch.load(f'./mnist/m2_{skey}.pt',map_location='cpu',weights_only=False)
                #self.slice_pool[skey]['param'].set_params(mdict['param'],'cpu')
                #current_model = self.get_model()
                #current_model.set_param(self.slice_pool[skey]['param'])
                #y,rt = run_model_test(current_model,current_model.device,current_model.device)
                #ref = ref.to(current_model.device)
                #print(torch.mean((y-ref)**2),rt)
                #self.slice_pool[skey]['param'].set_params(param.to('cpu'),'cpu')
        except:
            print(f'fails:{future.exception()}')
        
                 
    def save_slice_pool(self,pool_path,load_best=False):
        if load_best: self.load_best_param()
        for pk in self.key_list: self.slice_pool[pk]['noise'] = None
        torch.save(self.slice_pool,pool_path)
        
    def load_slice_pool(self,pool_path):
        #self.slice_pool = torch.load(pool_path,map_location='cpu',weights_only=False) 版本不支持weights_only
        self.slice_pool = torch.load(pool_path,map_location='cpu',weights_only=False)
        for skey in self.key_list: self.slice_pool[skey]['param'].load_reset()
        
    def validate(self,cnt):
        if self.helper.validate(cnt):
            print('-'*20+f'validate epoch {cnt}'+'-'*20)
            self.test()
            self.schedule_lr()
            return True
        return False
    
    def forward_test(self):
        if self.testing_data_redeay: return
        self.rate_pool = {}
        self.data_pool = {}
        self.free_model()
        try:
            self.executor.clear_futures()
            for pk in self.key_list:
                #print(f"pk is {pk}")
                current_model = self.get_model()
                current_model.set_param(self.slice_pool[pk]['param'])
               # print(f"interesting:{pk},{self.slice_pool[pk]['param'].pool['sp'][0][0]}")
                self.executor.submit_task(pk,run_model_test,current_model,current_model.device,self.target_device)
            self.executor.all_tasks_done()
            for future in self.executor.futures.keys():
                y,rt = future.result()
                pk = self.executor.futures[future]
                self.data_pool[pk] = nn.Parameter(y,requires_grad=False)
                self.rate_pool[pk] = rt
        except Exception as e:
            print(f'fails:{future.exception()}')
            print(str(e))
        self.testing_data_redeay = True
    
    def forward_test_qat(self, quant_config_map, quantizer_type='hardround', soft_round_temperature=0.0):
        """
        量化版本的 forward_test（用于测试时的硬量化前向传播，无噪声）
        
        Args:
            quant_config_map: {pk: quant_config}
            quantizer_type: 量化器类型（测试时通常用 'hardround'）
            soft_round_temperature: softround 温度（测试时通常为 0）
        """
        if self.testing_data_redeay:
            return
        
        self.rate_pool = {}
        self.data_pool = {}
        self.free_model()
        
        try:
            self.executor.clear_futures()
            for pk in self.key_list:
                current_model = self.get_model()
                current_model.set_param(self.slice_pool[pk]['param'])
                quant_config = quant_config_map.get(pk, None)
                self.executor.submit_task(
                    pk, run_model_test_qat,
                    current_model, quant_config,
                    quantizer_type,
                    current_model.device, self.target_device
                )
            self.executor.all_tasks_done()
            for future in self.executor.futures.keys():
                pk = self.executor.futures[future]
                y, rt, _ = future.result()  # 测试时忽略 nn_int_all
                self.data_pool[pk] = nn.Parameter(y, requires_grad=False)
                self.rate_pool[pk] = rt
        except Exception as e:
            print(f'[QAT] forward_test_qat fails: {e}')
        
        self.testing_data_redeay = True
        
    
    def forward_data(self):
        self.rate_pool = {}
        self.data_pool = {}
        self.free_model()
        try:
            self.executor.clear_futures()
            for pk in self.key_list:
                current_model = self.get_model()
                current_model.set_param(self.slice_pool[pk]['param'])
                self.executor.submit_task(pk,run_model,current_model,self.helper.qp,current_model.device,self.target_device)
            self.executor.all_tasks_done()
            for future in self.executor.futures.keys():
                pk = self.executor.futures[future]
                y,noise,rt = future.result()
                self.data_pool[pk] = nn.Parameter(y,requires_grad=True)
                self.data_pool[pk].grad = torch.zeros_like(self.data_pool[pk].data)
                self.slice_pool[pk]['noise'] = noise
                self.rate_pool[pk] = rt
        except:
            print(f'fails:{future.exception()}')
    
    def backward(self,lr,ldb):
        self.free_model()
        try:
            
            self.executor.clear_futures()
            cnt=0
            for pk in self.key_list:
                current_model = self.get_model()
                current_model.set_param(self.slice_pool[pk]['param'])
                self.executor.submit_task(pk,run_model_backward,current_model,self.helper.qp,
                                          self.data_pool[pk].grad,self.slice_pool[pk]['noise'],current_model.device,lr,ldb,self.epoch,cnt)
                cnt=cnt+1
                
            self.executor.all_tasks_done()
            self.epoch+=1
            for future in self.executor.futures.keys():
                pass
        except:
            print(f'fails:{future.exception()}')
            pass
    
    def forward_data_qat(self, quant_config_map, quantizer_type='hardround', soft_round_temperature=0.0):
        """
        量化版本的 forward_data（生成合成数据集）
        
        Args:
            quant_config_map: {pk: quant_config}
            quantizer_type: NN 参数量化器类型
            soft_round_temperature: softround 温度
        
        Returns:
            保存到 self.data_pool, self.rate_pool, self.nn_int_all_map
        """
        self.rate_pool = {}
        self.data_pool = {}
        self.nn_int_all_map = {}  # 新增：存储 NN 参数整数符号
        self.free_model()
        try:
            self.executor.clear_futures()
            for pk in self.key_list:
                current_model = self.get_model()
                current_model.set_param(self.slice_pool[pk]['param'])
                quant_config = quant_config_map.get(pk, None)
                self.executor.submit_task(
                    pk, run_model_qat,
                    current_model, self.helper.qp, quant_config, 
                    quantizer_type, soft_round_temperature,
                    current_model.device, self.target_device
                )
            self.executor.all_tasks_done()
            for future in self.executor.futures.keys():
                pk = self.executor.futures[future]
                y, noise, rt, nn_int_all = future.result()
                self.data_pool[pk] = nn.Parameter(y, requires_grad=True)
                self.data_pool[pk].grad = torch.zeros_like(self.data_pool[pk].data)
                self.slice_pool[pk]['noise'] = noise
                self.rate_pool[pk] = rt
                self.nn_int_all_map[pk] = nn_int_all  # 保存用于 backward
        except Exception as e:
            print(f'[QAT] forward_data_qat fails: {e}')
    
    def backward_qat(self, lr, ldb, ldb_nn, alpha_nn_map, quant_config_map, quantizer_type='ste', soft_round_temperature=0.0):
        """
        量化版本的 backward（反传梯度）
        
        Args:
            ldb_nn: NN 参数码率权重
            alpha_nn_map: {pk: alpha_nn}
            quant_config_map: {pk: quant_config}
            quantizer_type: NN 参数量化器类型
            soft_round_temperature: softround 温度
        """
        self.free_model()
        try:
            self.executor.clear_futures()
            cnt = 0
            for pk in self.key_list:
                current_model = self.get_model()
                current_model.set_param(self.slice_pool[pk]['param'])
                quant_config = quant_config_map.get(pk, None)
                alpha_nn = alpha_nn_map.get(pk, 1.0)
                nn_int_all = self.nn_int_all_map.get(pk, None)
                
                self.executor.submit_task(
                    pk, run_model_backward_qat,
                    current_model, self.helper.qp,
                    quant_config, quantizer_type, soft_round_temperature,
                    self.data_pool[pk].grad, self.slice_pool[pk]['noise'], nn_int_all,
                    current_model.device, lr, ldb, ldb_nn, alpha_nn,
                    self.epoch, cnt
                )
                cnt += 1
            
            self.executor.all_tasks_done()
            self.epoch += 1
            for future in self.executor.futures.keys():
                pass
        except Exception as e:
            print(f'[QAT] backward_qat fails: {e}')
