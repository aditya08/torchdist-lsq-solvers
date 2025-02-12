import os
import torch
import torch.distributed as torchdist
import mpi4py.MPI as MPI
import torch.utils.benchmark
import time

class DistributedCGLS:
    def __init__(self, nrows, ncols, niters, tol=1e-4, device='cpu', backend='gloo', precision='fp32'):
        if device == 'cuda' and not torch.cuda.is_available():
            raise ValueError('No GPU found')
        if device == 'cuda' and backend != 'nccl':
            raise ValueError('NCCL is the only backend supported for CUDA')
        if device == 'cpu' and backend == 'nccl':
            raise ValueError('NCCL is not supported for CPU')
        self.device = device
        # self.nprocs = nprocs
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.world_size = self.comm.Get_size()
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12345'
        torchdist.init_process_group(backend=backend, rank=self.rank, world_size=self.world_size)
        self.global_nrows = nrows
        # Generates a 1D-row partitioned random dense matrix
        self.nrows = nrows // self.world_size # rows per rank
        if self.rank < nrows % self.world_size:
            self.nrows += 1

        self.ncols = ncols

        if precision == 'fp64':
            self.dtype = torch.float64
        elif precision == 'fp32':
            self.dtype = torch.float32
        elif precision == 'fp16':
            self.dtype = torch.float16
            raise ValueError('fp16 does not use shifts currently.')
        else:
            raise ValueError('Precision must be one of: fp64, fp32, fp16')
        # print('nrows: ', self.nrows)
        self.weights = torch.rand(self.nrows, 1, dtype=self.dtype, device=self.device)
        self.data = torch.sqrt(self.weights) * torch.randn(self.nrows, self.ncols, dtype=self.dtype, device=self.device)
        self.y = torch.sqrt(self.weights) * torch.randn(self.nrows, 1, dtype=self.dtype, device=self.device)
        self.solution = torch.zeros(self.ncols, 1, dtype=self.dtype, device=self.device)
        self.resnrm = float('inf')
        self.gradnrm = float('inf')
        self.niters = niters
        self.tol = tol
        self.comm_time = 0
        self.comp_time = 0

    def train(self, print_freq=1):
        start = time.time()
        iter = 0
        r = (self.data @ self.solution) - self.y
        s = self.data.T @ r
        self.comp_time += time.time() - start
        start = time.time()
        torchdist.all_reduce(s)
        self.comm_time += time.time() - start
        time.time()
        p = s
        norms0 = torch.linalg.norm(s)
        gamma = norms0 ** 2
        normx = torch.linalg.norm(self.solution)
        xmax = normx
        flag = 0
        self.comp_time += time.time() - start
        while(iter < self.niters and flag == 0):
            start = time.time()
            iter = iter + 1
            q = self.data @ p
            delta = q.T @ q
            torchdist.all_reduce(delta)
            alpha = gamma / delta
            self.solution = self.solution + alpha * p
            r = r - alpha * q
            s = self.data.T @ r
            self.comp_time += time.time() - start
            start = time.time()
            torchdist.all_reduce(s)
            self.comm_time += time.time() - start
            start = time.time()
            norms = torch.linalg.norm(s)
            gamma1 = gamma
            gamma = norms ** 2
            beta = gamma / gamma1
            p = s + beta * p

            normx = torch.linalg.norm(self.solution)
            xmax = max(normx, xmax)
            flag = (norms <= norms0 * self.tol) or (normx * self.tol >= 1)
            self.comp_time += time.time() - start
            if iter % print_freq == 0 and self.rank == 0:
                print('iter: ', iter, ' norm_s: ', norms.item(), 'comp_time: ', self.comp_time, 'comm_time: ', self.comm_time)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    # required args
    parser.add_argument('--nrows', type=int, required=True, help='global nrows of the data matrix to generate')
    parser.add_argument('--ncols', type=int, required=True, help='global cols of the data matrix to generate')
    parser.add_argument('--niters', type=int, required=True, help='number of gradient descent iterations.')
    # optional args
    parser.add_argument('--tol', type=float, default=1e-4, help='resnorm tolerance.')
    parser.add_argument('--device', type=str, default='cpu', help='device to use.')
    parser.add_argument('--backend', type=str, default='gloo', help='backend to use for torch.distributed')
    parser.add_argument('--precision', type=str, default='fp32', help='floating-point precision of computations.')
    args = parser.parse_args()
    cgls = DistributedCGLS(args.nrows, args.ncols, args.niters, args.tol, args.device, args.backend, args.precision)
    start = time.time()
    cgls.train()
    if cgls.rank == 0:
        print('Total Runtime: ', time.time() - start)