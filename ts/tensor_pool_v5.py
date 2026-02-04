import torch
import random
from ts.tensor_data_func_v5 import TensorData, DPParams,TrainingHelper,_linear_schedule, TrainParam
import concurrent.futures
import time
from torch import nn


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
    model.to_run()
    y,noise = model.mimic_forward(qp)
    y = y.to(target_device)
    noise = noise.to(target_device)
    model.to(param_device)
    return y,noise

def run_model_test(model:TensorData, param_device, target_device):
    model.to_run()
    y,rt = model.forward_for_test()
    npiexls = y.shape[0] * y.shape[-1]*y.shape[-2]
    rt = torch.sum(rt).item() / npiexls
    y = y.to(target_device)
    model.to(param_device)
    return y,rt

def run_model_backward(model:TensorData, qp:TrainParam, data_grad, noise, target_device):
    running_device = model.device
    noise = noise.to(running_device)
    data_grad = data_grad.to(running_device)
    model.empty_grad()
    y,rt = model.forward_per_sample(qp,noise)
    drt = rt.clone().detach().requires_grad_()
    drt.retain_grad()
    #print('backward',data_grad.device,running_device)
    npiexls = data_grad.shape[0] * data_grad.shape[-1]*data_grad.shape[-2]
    bpp = drt.sum() / npiexls
    rt_loss = bpp * qp.ldb
    rt_loss.backward()
    torch.autograd.backward([y,rt],[data_grad,drt.grad])
    model.dp.bpp = bpp.item()
    
    model.step()
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
    
    def __init__(self,nclass,slice_in_class,sample_per_slice, cuda_devices_idxs:list[int], nthread = 4,
                    batch_size = 16, ldb=0.001,img_size=(512,512),max_iter=50000)->None:
        self.nclass = nclass
        self.max_iter = max_iter
        self.slice_in_class = slice_in_class
        self.sample_per_slice = sample_per_slice
        self.nslice = slice_in_class * sample_per_slice
        self.cuda_devices_idx = cuda_devices_idxs
        self.nsample = nclass*slice_in_class*sample_per_slice
        self.batch_size = batch_size
        self.nbatch = (self.nsample + batch_size - 1) // batch_size
        self.slice_pool = {f'{cidx}_{sidx}':{'param':DPParams()} for sidx in range(slice_in_class) for cidx in range(nclass)}
        self.key_list = self.slice_pool.keys()
        self.target_device= f'cuda:{(cuda_devices_idxs[-1])}'
        self.nthread = nthread
        self.worker_pool = {tidx: TensorData(image_size=img_size,batch_size=self.sample_per_slice,  
                                             device='cuda:{}'.format(cuda_devices_idxs[tidx%len(cuda_devices_idxs)])) for tidx in range(nthread)}
        self.helper = TrainingHelper(ldb,max_iter)
        self.executor = ThreadPoolManager(nthread)
        self.schedule = [] 
        self.ldb = ldb
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
     
    def record(self):
        for pk in self.slice_pool.keys(): all_record(self.slice_pool[pk]['param'])
    
    def load_best(self):
        if not self.helper.dop.phase.schedule_lr: return
        for pk in self.slice_pool.keys(): all_load_best(self.slice_pool[pk]['param'],'cpu')
    
    def load_best_param(self):
        for pk in self.slice_pool.keys(): all_load_best_param(self.slice_pool[pk]['param'],'cpu')
        
    def schedule_lr(self):
        for pk in self.slice_pool.keys(): all_schedul_lr(self.slice_pool[pk]['param'])
                
    def set_training_phase(self, phase):
        self.helper.set_training_phase(phase)
        self.helper.qp.start()
        
    def init_solvers(self):
        for pk in self.slice_pool.keys(): self.helper.init_solver(self.slice_pool[pk]['param'])
                
    def load_slice_pool(self,pool_path):
        self.slice_pool = torch.load(pool_path,map_location='cpu',weights_only=False)
    
    def load_from_files(self,root):
        for cidx in range(self.nclass):
            for sidx in range(self.slice_in_class):
                mdict = torch.load(f'{root}/{cidx}_{sidx}.pt',map_location='cpu',weights_only=False)
                self.slice_pool[f'{cidx}_{sidx}']['param'].set_params(mdict['param'],'cpu')
                 
    def save_slice_pool(self,pool_path):
        self.slice_pool = torch.save(self.slice_pool,pool_path)
        
    def split_idx(self,idx):
        cidx = idx % self.sample_per_slice
        idx = idx // self.sample_per_slice
        bidx = idx % self.slice_in_class
        aidx = idx // self.slice_in_class
        return aidx,bidx,cidx
        
    def validate(self,cnt):
        if self.helper.validate(cnt):
            print('-'*20+f'validate epoch {cnt}'+'-'*20)
            self.test()
            self.schedule_lr()
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
        #print('forward testing')
        self.rate_pool = {}
        self.data_pool = {}
        pool_idx = -1
        try:
            self.executor.clear_futures()
            for pk in self.key_list:
                pool_idx =  (pool_idx + 1) % self.nthread
                current_model = self.worker_pool[pool_idx]
                current_model.set_param(self.slice_pool[pk]['param'])
                self.executor.submit_task(pk,run_model_test,current_model,current_model.device,self.target_device)
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
                current_model = self.worker_pool[pool_idx]
                current_model.set_param(self.slice_pool[pk]['param'])
                self.executor.submit_task(pk,run_model,current_model,self.helper.qp,current_model.device,self.target_device)
            self.executor.all_tasks_done()
            for future in self.executor.futures.keys():
                pk = self.executor.futures[future]
                y,noise = future.result()
                self.data_pool[pk] = nn.Parameter(y,requires_grad=True)
                self.data_pool[pk].grad = torch.zeros_like(self.data_pool[pk].data)
                self.slice_pool[pk]['noise'] = noise
                self.rate_pool[pk] = self.slice_pool[pk]['param'].bpp
        except:
            print(f'fails:{future.exception()}')

    
    def backward(self):
        #print('backward...')
        pool_idx = -1
        try:
            self.executor.clear_futures()
            for pk in self.key_list:
                pool_idx = (pool_idx + 1)%self.nthread
                current_model = self.worker_pool[pool_idx]
                current_model.set_param(self.slice_pool[pk]['param'])
                self.executor.submit_task(pk,run_model_backward,current_model,self.helper.qp,
                                          self.data_pool[pk].grad,self.slice_pool[pk]['noise'],current_model.device)
            self.executor.all_tasks_done()
            for future in self.executor.futures.keys():
                pass
        except:
            print(f'fails:{future.exception()}')
            pass
        
    
