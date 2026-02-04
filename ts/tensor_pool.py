import torch
import random
from ts.tensor_data_func_v2 import TensorData,DGrids,DParams,DataOptim,_linear_schedule
import concurrent.futures
import time

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

def all_to(param:DParams, grids:list[DGrids],device):
    param = param.to(device)
    for grid in grids: grid = grid.to(device)

def all_empty_grad(param:DParams, grids:list[DGrids]):
    param.empty_grad()
    for grid in grids: grid.empty_grad()
    
def all_step(param:DParams, grids:list[DGrids],device):
    param.step()
    for grid in grids: grid.step()
    
def all_beat(param:DParams, grids:list[DGrids], loss):
    for dg in grids: dg.beat(loss)
    return param.beat(loss)
    
def all_load_best(param:DParams, grids:list[DGrids],device):
    param.load_best(device)
    for dg in grids: dg.load_best(device)
    
def all_load_best_param(param:DParams, grids:list[DGrids], device):
    param.load_best_param(device)
    for dg in grids: dg.load_best_param(device)
    
def all_schedul_lr(param:DParams, grids:list[DGrids]):
    param.lr_schedule.step()
    for dg in grids: dg.lr_schedule.step()

def run_model(model:TensorData, param:DParams, grids:list[DGrids], qp:TrainParam, running_device, target_device):
    all_to(param,grids,running_device)
    y,noise = model.mimic_forward(grids,param,qp.noise_type,qp.quant_type,qp.qa,qp.qb)
    y = y.to(target_device)
    noise = noise.to(target_device)
    return y,noise

def run_model_test(model:TensorData, param:DParams, grids:list[DGrids], running_device, target_device):
    all_to(param,grids,running_device)
    y,rt = model.forward_for_test(grids,param)
    rt = torch.mean(rt).item()
    y = y.to(target_device)
    return y,rt

def run_model_backward(model:TensorData, param:DParams, grids:list[DGrids], qp:TrainParam, data_grad, noise, running_device, target_device):

    all_to(param,grids,running_device)
    noise = noise.to(running_device)
    data_grad = data_grad.to(running_device)
    all_empty_grad(param,grids)
    y,rt,rtp = model.forward_per_sample(grids,param,noise,qp.noise_type,qp.quant_type,qp.qa,qp.qb)
    drt = rt.clone().detach().requires_grad_()
    drt.retain_grad()
    piexls_p = data_grad.shape[-1]*data_grad.shape[-2]
    rt_loss = drt.sum() / qp.npixels * qp.ldb
    rt_loss.backward()
    torch.autograd.backward([y,rt],[data_grad,drt.grad])
    for idx,pt in enumerate(grids):  pt.bpp = rtp[idx].item() / piexls_p
    all_step(param,grids,running_device)
    all_to(param,grids,target_device)
    return True

class ThreadPoolManager:
    def __init__(self, max_workers=3):
        # 创建一个线程池
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        self.futures = {}
        print(f"Thread pool created with {max_workers} workers.")

    def clear_futures(self):
        self.futures = {}

    def submit_task(self, thread_key, func, *args, **kwargs):
        # 提交任务到线程池
        future = self.executor.submit(func, *args, **kwargs)
        self.futures[future] = thread_key 
        #print(f"Task submitted: {func.__name__}")
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
        self.npixels_per_slice = img_size[0]*img_size[1]*sample_per_slice
        self.slice_in_class = slice_in_class
        self.sample_per_slice = sample_per_slice
        self.nslice = slice_in_class * sample_per_slice
        self.cuda_devices_idx = cuda_devices_idxs
        self.nsample = nclass*slice_in_class*sample_per_slice
        self.batch_size = batch_size
        self.nbatch = (self.nsample + batch_size - 1) // batch_size
        self.slice_pool = {f'{cidx}_{sidx}':{'param':DParams(),'grids':[DGrids() for _ in range(sample_per_slice)]} for sidx in range(slice_in_class) for cidx in range(nclass)}
        self.device_pool = {f'{cidx}_{sidx}': self.get_device_idx(cidx,sidx) for sidx in range(slice_in_class) for cidx in range(nclass)}
        self.target_device= f'cuda:{(cuda_devices_idxs[-1])}'
        self.nthread = nthread
        self.worker_pool = {tidx: TensorData(ldb,n_itr=max_iter,image_size=img_size) for tidx in range(nthread)}
        self.executor = ThreadPoolManager(nthread)
        self.schedule = [] 
        self.ldb = ldb
        self.manager = None
        self.qp = None
        self.bidx = 0
        self.epoch = 0
        self.testing = False
        
    def __iter__(self):
        #print('iter')
        self.bidx = 0
        self.shuffle_idxs()
        return self
    
    def __next__(self):
        #print(f'next {self.bidx}, {self.nbatch}')
        if self.bidx < self.nbatch:
            self.current_batch = self.schedule[self.bidx]
            self.bidx += 1
            return self.bidx - 1
        else:
            raise StopIteration
        
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
        self.qp.set_npixels(self.npixels_per_slice)
        self.qp.start()
        
    def init_solvers(self):
        for pk in self.slice_pool.keys():
            self.manager.init_solver_dp(self.slice_pool[pk]['param'])
            for grid in self.slice_pool[pk]['grids']:
                self.manager.init_solver_dg(grid)
        
    def get_device_idx(self,cidx,sidx):
        return 'cuda:{}'.format(self.cuda_devices_idx[(cidx*self.slice_in_class+sidx)%len(self.cuda_devices_idx)])
                
    def load_slice_pool(self,pool_path):
        self.slice_pool = torch.load(pool_path,map_location='cpu',weights_only=False)
    
    def load_from_files(self,root):
        for cidx in range(self.nclass):
            for sidx in range(self.slice_in_class):
                 mdict = torch.load(f'{root}/{cidx}_{sidx}.pt',map_location='cpu',weights_only=False)
                 self.slice_pool[f'{cidx}_{sidx}']['param'].set_params(mdict['param'],'cpu')
                 for idx in range(self.sample_per_slice): self.slice_pool[f'{cidx}_{sidx}']['grids'][idx].set_params(mdict['grids'][idx],'cpu')
                 
    def save_slice_pool(self,pool_path):
        self.slice_pool = torch.save(self.slice_pool,pool_path)
        
    def split_idx(self,idx):
        cidx = idx % self.sample_per_slice
        idx = idx // self.sample_per_slice
        bidx = idx % self.slice_in_class
        aidx = idx // self.slice_in_class
        return aidx,bidx,cidx
    
    def update_quant_param(self):
        self.qp.step(self.epoch)
        
    def validate(self,cnt):
        if ((cnt + 1) % self.qp.phase.freq_valid == 0) or (cnt + 1 == self.qp.phase.max_itr):
            self.test()
            self.schedule_lr()
            self.qp.step(cnt)
            return True
        return False
            
    def shuffle_idxs(self):
        from functools import reduce
        self.bidx = 0
        self.epoch += 1
        #myshuffle = lambda x: x if random.shuffle(x) is None else x
        #idxs_pool = [myshuffle([i+j*self.sample_per_slice for i in range(self.sample_per_slice)]) for j in range(self.nclass*self.slice_in_class)]
        #idxs_pool = reduce(lambda x,y:x+y,idxs_pool)
        #print(idxs_pool)
        idxs_pool = [i for i in range(self.nsample)]
        if not self.testing: random.shuffle(idxs_pool)
        self.idxs_pool = idxs_pool
        self.schedule = []
        for idx in range(self.nbatch):
            batch = {}
            for idy in range(self.batch_size):
                pid = idx * self.batch_size + idy 
                if pid >= self.nsample: break
                aidx,bidx,cidx = self.split_idx(idxs_pool[pid])
                pk = f'{aidx}_{bidx}'
                if not pk in batch: batch[pk] = {'param':self.slice_pool[pk]['param'],'grids':[],'idxs':[]}
                batch[pk]['grids'].append(self.slice_pool[pk]['grids'][cidx])
                batch[pk]['idxs'].append(idxs_pool[pid])
            self.schedule.append(batch)
    
    def forward_test(self):
        batch = self.current_batch
        key_list = list(batch.keys())
        total = len(key_list)
        nblk = (total + self.nthread - 1) // self.nthread
        ds_dict,rt_dict,idx_list = {},{},[]
        try:
            for blk in range(nblk):
                self.executor.clear_futures()
                for iidx in range(self.nthread):
                    cur = blk*self.nthread + iidx
                    if cur>= total: break
                    pk = key_list[cur]
                    self.executor.submit_task(pk,run_model_test,self.worker_pool[iidx],batch[pk]['param'],batch[pk]['grids'],self.device_pool[pk],self.target_device)
                    idx_list +=  batch[pk]['idxs']
                self.executor.all_tasks_done()
                for future in self.executor.futures.keys():
                    y,rt = future.result()
                    pk = self.executor.futures[future]
                    ds_dict[pk] = y
                    rt_dict[pk] = rt
                    #print(f'finish {pk}')
        except:
            print(f'fails:{future.exception()}')
        
        ds_list,rt_list = [ds_dict[pk] for pk in key_list],[rt_dict[pk] for pk in key_list]
        return torch.cat(ds_list,dim=0).detach(), sum(rt_list) / len(rt_list), idx_list
    
    def test(self):
        self.testing = True
        
    def train(self):
        self.testing = False
    
    def forward_data(self):
        batch = self.current_batch
        rt_list = []
        key_list = list(batch.keys())
        total = len(key_list)
        nblk = (total + self.nthread - 1) // self.nthread
        self.num_list = []
        ds_dict,num_dict,idx_list = {},{},[]
        try:
            for blk in range(nblk):
                self.executor.clear_futures()
                for iidx in range(self.nthread):
                    cur = blk*self.nthread + iidx
                    if cur>= total: break
                    pk = key_list[cur]
                    self.executor.submit_task(pk,run_model,self.worker_pool[iidx],batch[pk]['param'],batch[pk]['grids'],self.qp,self.device_pool[pk],self.target_device)
                    rt_list += [grid.bpp for grid in batch[pk]['grids']]
                    idx_list +=  batch[pk]['idxs']
                self.executor.all_tasks_done()
                for future in self.executor.futures.keys():
                    pk = self.executor.futures[future]
                    y,noise = future.result()
                    num_dict[pk] = y.shape[0]
                    ds_dict[pk] = y
                    batch[pk]['noise'] = noise
        except:
            print(f'fails:{future.exception()}')
        ds_list = [ds_dict[pk] for pk in key_list]
        self.num_list = [num_dict[pk] for pk in key_list]
        return torch.cat(ds_list,dim=0).detach(), sum(rt_list) / len(rt_list), idx_list
    
    def backward(self,data_grad):
        batch = self.current_batch
        key_list = list(batch.keys())
        total = len(key_list)
        nblk = (total + self.nthread - 1) // self.nthread
        data_grad_list = torch.split(data_grad,self.num_list,dim=0)
        try:
            for blk in range(nblk):
                self.executor.clear_futures()
                for iidx in range(self.nthread):
                    cur = blk*self.nthread + iidx
                    if cur>= total: break
                    pk = key_list[cur]
                    self.executor.submit_task(pk,run_model_backward,self.worker_pool[iidx],batch[pk]['param'],batch[pk]['grids'],self.qp,data_grad_list[cur],batch[pk]['noise'],self.device_pool[pk],self.device_pool[pk])
                self.executor.all_tasks_done()
        except:
            #print(f'fails:{future.exception()}')
            pass
        
    
