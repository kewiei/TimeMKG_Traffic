from collections import defaultdict
import torch

cp_path = '/home/nanodt/gitroot/TimeMKG_Traffic/Code/checkpoints/long_term_forecast_Traffic_TimeMKG_Traffic_Multivariate_id_10-8.csv_ftM_sl10080_ll0_pl1440_dm512_nh8_el2_dl2_df512_expand2_dc4_fc1_ebtimeF_dtTrue_test_0/checkpoint.pth'
checkpoint = torch.load(cp_path)
sd = checkpoint

groups = defaultdict(list)
for k, tensor in sd.items():
    module = k.split('.')[0]  # 一级模块名
    groups[module].append((k, tensor))

for module, items in groups.items():
    print(f"=== {module} ===")
    for k, tensor in items:
        numel = tensor.numel()
        size_bytes = numel * tensor.element_size()
        print("  ", k, tensor.shape, f"{size_bytes / 1024:.2f} KB")
