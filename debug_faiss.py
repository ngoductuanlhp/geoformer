import faiss                     # make faiss available
import faiss.contrib.torch_utils

import torch

def main():
    cfg = faiss.GpuIndexFlatConfig()
    cfg.useFloat16 = False
    cfg.device = 0

    # self.knn_res = faiss.StandardGpuResources()
    # self.geo_knn = faiss.index_cpu_to_gpu(self.knn_res, 0, faiss.IndexFlatL2(3))
    geo_knn = faiss.GpuIndexFlatL2(faiss.StandardGpuResources(), 3, cfg)

    locs_float_b = torch.ones((10000,3), dtype=torch.float, device='cuda')
    geo_knn.add(locs_float_b)

main()