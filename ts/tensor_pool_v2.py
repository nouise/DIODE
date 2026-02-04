import torch
import random
from ts.tensor_data_func import TensorData,DGrids,DParams,DataOptim,_linear_schedule
import concurrent.futures
import time
from torch import nn

class TrainParam:
    
    def __init__(self, phase,ldb) -> None:
        self.qa = None
        self.qb = None
        self.phase = phase
        self.noise_type = self.phase.quantizer_noise_type
        self.quant_type = self.phase.quantizer_type
        self.ldb = ldb
        self.npixels = None
        self.step(0)
        
    def set_npixels(self,npixels):
        self.npixels = npixels
        
    def start(self):
        self.record = 0
        
    def load_best(self,current)->bool:
        if current-self.record > self.phase.patience and self.phase.schedule_lr:
            self.record = current
            print('reload current best')
            return True
        return False
        
    def step(self, iter)->None:
        self.qa = _linear_schedule(self.phase.softround_temperature[0],  self.phase.softround_temperature[1],  iter,  self.phase.max_itr, )
        self.qb = _linear_schedule(self.phase.noise_parameter[0], self.phase.noise_parameter[1], iter, self.phase.max_itr)

def all_to(param:DParams, grids:DGrids,device):
    param = param.to(device)
    grids = grids.to(device)

def all_empty_grad(param:DParams, grids:DGrids):
    param.empty_grad()
    grids.empty_grad()
    
def all_step(param:DParams, grids:DGrids,device):
    param.step()
    grids.step()
    
def all_beat(param:DParams, grids:DGrids, loss):
    grids.beat(loss)
    return param.beat(loss)
    
def all_load_best(param:DParams, grids:DGrids,device):
    param.load_best(device)
    grids.load_best(device)
    
def all_load_best_param(param:DParams, grids:DGrids, device):
    param.load_best_param(device)
    grids.load_best_param(device)
    
def all_schedul_lr(param:DParams, grids:DGrids):
    param.lr_schedule.step()
    grids.lr_schedule.step()

def run_model(model:TensorData, param:DParams, grids:DGrids, qp:TrainParam, running_device, target_device):
    all_to(param,grids,running_device)
    y,noise = model.mimic_forward(grids,param,qp.noise_type,qp.quant_type,qp.qa,qp.qb)
    y = y.to(target_device)
    noise = noise.to(target_device)
    return y,noise

def run_model_test(model:TensorData, param:DParams, grids:DGrids, running_device, target_device):
    all_to(param,grids,running_device)
    y,rt = model.forward_for_test(grids,param)
    rt = torch.mean(rt).item()
    y = y.to(target_device)
    return y,rt

def run_model_backward(model:TensorData, param:DParams, grids:DGrids, qp:TrainParam, data_grad, noise, running_device, target_device):
    #print('here backward',data_grad.device,running_device)
    #print(qp.pool['ap'][0].device,running_device)
    all_to(param,grids,running_device)
    #print(param.pool['ap'][0].device,running_device)
    noise = noise.to(running_device)
    data_grad = data_grad.to(running_device)
    all_empty_grad(param,grids)
    y,rt = model.forward_per_sample(grids,param,noise,qp.noise_type,qp.quant_type,qp.qa,qp.qb)
    drt = rt.clone().detach().requires_grad_()
    drt.retain_grad()
    npiexls = data_grad.shape[0] * data_grad.shape[-1]*data_grad.shape[-2]
    bpp = drt.sum() / npiexls
    rt_loss = bpp * qp.ldb
    rt_loss.backward()
    torch.autograd.backward([y,rt],[data_grad,drt.grad])
    grids.bpp = bpp.item()
    #print('backward',data_grad.device,running_device)
    all_step(param,grids,running_device)
    all_to(param,grids,target_device)
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
            time.sleep(0.01)
            all_done = all(future.done() for future in self.futures.keys())

    def shutdown(self, wait=True):
        self.executor.shutdown(wait=wait)
        print("Thread pool has been shut down.")

class TensorPool:
    
    def __init__(self,nclass,slice_in_class,sample_per_slice, cuda_devices_idxs:list[int], nthread = 4,
                    batch_size = 16, ldb=0.001,img_size=(512,512),max_iter=50000)->None:
        self.nclass = nclass
        self.slice_in_class = slice_in_class
        self.sample_per_slice = sample_per_slice
        self.nslice = slice_in_class * sample_per_slice
        self.cuda_devices_idx = cuda_devices_idxs
        self.nsample = nclass*slice_in_class*sample_per_slice
        self.batch_size = batch_size
        self.nbatch = (self.nsample + batch_size - 1) // batch_size
        self.slice_pool = {f'{cidx}_{sidx}':{'param':DParams(),'grids':DGrids()} for sidx in range(slice_in_class) for cidx in range(nclass)}
        self.key_list = self.slice_pool.keys()
        self.target_device= f'cuda:{(cuda_devices_idxs[-1])}'
        self.nthread = nthread
        self.worker_pool = {tidx: TensorData(ldb,n_itr=max_iter,image_size=img_size,batch_size=self.sample_per_slice) for tidx in range(nthread)}
        self.device_pool = {tidx: 'cuda:{}'.format(cuda_devices_idxs[tidx%len(cuda_devices_idxs)]) for tidx in range(nthread)}
        self.executor = ThreadPoolManager(nthread)
        self.schedule = [] 
        self.ldb = ldb
        self.manager = None
        self.qp = None
        self.bidx = 0
        self.epoch = 0
        self.testing = False
        self.data_pool = None
        
    def __iter__(self):
        self.bidx = 0
        self.shuffle_idxs()
        if not self.testing:
            self.forward_data()
        else:
            self.forward_test()
        return self
    
    def __next__(self):
        if self.bidx < self.nbatch:
            data,rate,idxs = self.get_data()
            self.bidx += 1
            return data,rate, idxs
        else:
            raise StopIteration
    
    def get_data(self):
        batch = self.schedule[self.bidx]
        self.batch = batch
        data_list = [self.data_pool[pk][idx:idx+1] for pk in batch.keys() for idx in batch[pk]['data']]
        data = nn.Parameter(torch.cat(data_list,dim=0),requires_grad=True)
        rate_list = [self.rate_pool[pk] for  pk in batch.keys() for idx in batch[pk]]
        rate = sum(rate_list) / len(rate_list)
        idx_list = [idx for pk in batch.keys() for idx in batch[pk]['idx']]
        return data,rate,idx_list
    
    @torch.no_grad()
    def fill_data_diff(self,data):
        batch = self.batch
        grads = data.grad
        gid = 0
        for pk in batch.keys():
            for idx in batch[pk]['data']:
                self.data_pool[pk].grad[idx:idx+1] = grads[gid:gid+1]
                gid+=1
     
    def beat(self,loss):
        for pk in self.slice_pool.keys(): all_beat(self.slice_pool[pk]['param'],self.slice_pool[pk]['grids'],loss)
    
    def load_best(self):
        for pk in self.slice_pool.keys(): all_load_best(self.slice_pool[pk]['param'],self.slice_pool[pk]['grids'],'cpu')
    
    def load_best_param(self):
        for pk in self.slice_pool.keys(): all_load_best_param(self.slice_pool[pk]['param'],self.slice_pool[pk]['grids'],'cpu')
        
    def schedule_lr(self):
        for pk in self.slice_pool.keys(): all_schedul_lr(self.slice_pool[pk]['param'],self.slice_pool[pk]['grids'])
                
    def set_training_phase(self, phase):
        self.manager = DataOptim(self.worker_pool[0].manager, phase)
        self.qp = TrainParam(self.manager.phase,self.ldb)
        self.qp.start()
        
    def init_solvers(self):
        for pk in self.slice_pool.keys():
            self.manager.init_solver_dp(self.slice_pool[pk]['param'])
            self.manager.init_solver_dg(self.slice_pool[pk]['grids'])
                
    def load_slice_pool(self,pool_path):
        self.slice_pool = torch.load(pool_path,map_location='cpu',weights_only=False)
    
    def load_from_files(self,root):
        for cidx in range(self.nclass):
            for sidx in range(self.slice_in_class):
                mdict = torch.load(f'{root}/{cidx}_{sidx}.pt',map_location='cpu',weights_only=False)
                self.slice_pool[f'{cidx}_{sidx}']['param'].set_params(mdict['param'],'cpu')
                self.slice_pool[f'{cidx}_{sidx}']['grids'].set_params(mdict['grids'],'cpu')
                 
    def save_slice_pool(self,pool_path):
        self.slice_pool = torch.save(self.slice_pool,pool_path)
        
    def split_idx(self,idx):
        cidx = idx % self.sample_per_slice
        idx = idx // self.sample_per_slice
        bidx = idx % self.slice_in_class
        aidx = idx // self.slice_in_class
        return aidx,bidx,cidx
        
    def validate(self,cnt):
        if ((cnt + 1) % self.qp.phase.freq_valid == 0) or (cnt + 1 == self.qp.phase.max_itr):
            self.test()
            self.schedule_lr()
            self.qp.step(cnt)
            return True
        return False
            
    def shuffle_idxs(self):
        self.bidx = 0
        self.epoch += 1
        idxs_pool = [i for i in range(self.nsample)]
        if not self.testing: random.shuffle(idxs_pool)
        self.schedule = []
        for idx in range(self.nbatch):
            batch = {}
            for idy in range(self.batch_size):
                pid = idx * self.batch_size + idy 
                if pid >= self.nsample: break
                aidx,bidx,cidx = self.split_idx(idxs_pool[pid])
                pk = f'{aidx}_{bidx}'
                if not pk in batch.keys(): batch[pk] = {'data':[],'idx':[]}
                batch[pk]['data'].append(cidx)
                batch[pk]['idx'].append(idxs_pool[pid])
            self.schedule.append(batch)
    
    def forward_test(self):
        print('forward testing')
        self.rate_pool = {}
        self.data_pool = {}
        pool_idx = -1
        try:
            self.executor.clear_futures()
            for pk in self.key_list:
                pool_idx =  (pool_idx + 1) % self.nthread
                self.executor.submit_task(pk,run_model_test,self.worker_pool[pool_idx],self.slice_pool[pk]['param'],self.slice_pool[pk]['grids'],self.device_pool[pool_idx],self.target_device)
            self.executor.all_tasks_done()
            for future in self.executor.futures.keys():
                y,rt = future.result()
                pk = self.executor.futures[future]
                self.data_pool[pk] = nn.Parameter(y,requires_grad=False)
                self.rate_pool[pk] = rt
        except:
            print(f'fails:{future.exception()}')
    
    def test(self):
        self.testing = True
        
    def train(self):
        self.testing = False
    
    def forward_data(self):
        self.rate_pool = {}
        self.data_pool = {}
        pool_idx = -1
        try:
            self.executor.clear_futures()
            for pk in self.key_list:
                pool_idx = (pool_idx + 1)%self.nthread
                self.executor.submit_task(pk,run_model,self.worker_pool[pool_idx],self.slice_pool[pk]['param'],self.slice_pool[pk]['grids'],self.qp,self.device_pool[pool_idx],self.target_device)
            self.executor.all_tasks_done()
            for future in self.executor.futures.keys():
                pk = self.executor.futures[future]
                y,noise = future.result()
                self.data_pool[pk] = nn.Parameter(y,requires_grad=True)
                self.data_pool[pk].grad = torch.zeros_like(self.data_pool[pk].data)
                self.slice_pool[pk]['noise'] = noise
                self.rate_pool[pk] = self.slice_pool[pk]['grids'].bpp
        except:
            print(f'fails:{future.exception()}')

    
    def backward(self):
        pool_idx = -1
        try:
            self.executor.clear_futures()
            for pk in self.key_list:
                #print(self.qp,self.data_pool[pk].grad.device)
                pool_idx = (pool_idx + 1)%self.nthread
                self.executor.submit_task(pk,run_model_backward,self.worker_pool[pool_idx],self.slice_pool[pk]['param'],self.slice_pool[pk]['grids'],
                                          self.qp,self.data_pool[pk].grad,self.slice_pool[pk]['noise'],self.device_pool[pool_idx],self.device_pool[pool_idx])
            self.executor.all_tasks_done()
            for future in self.executor.futures.keys():
                pass
        except:
            print(f'fails:{future.exception()}')
            pass
        
    
