import os
import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
import argparse
from tqdm import tqdm

def create_test_args():
    """创建测试所需的args"""
    parser = argparse.ArgumentParser(description='Test multiple models')
    
    # --- 基础配置 ---
    parser.add_argument('--task_name', type=str, default='long_term_forecast', help='task name')
    parser.add_argument('--is_training', type=int, default=0, help='status, 0 for testing')
    parser.add_argument('--model_id', type=str, default='Traffic', help='model id (will be overridden)')
    parser.add_argument('--model', type=str, default='iTransformer', help='model name')

    # --- 数据加载配置 ---
    parser.add_argument('--data', type=str, default='Traffic_Singlevariate', help='dataset type')
    parser.add_argument('--root_path', type=str, default='traffic/edge_test/', help='root path of test data')
    parser.add_argument('--data_path', type=str, default='id_1-2.csv', help='test data file (will be overwritten)')
    parser.add_argument('--features', type=str, default='M', help='forecasting task')
    parser.add_argument('--freq', type=str, default='h', help='freq for time features encoding')
    parser.add_argument('--target', type=str, default='speed', help='target feature')
    parser.add_argument('--checkpoints', type=str, default='checkpoints/', help='location of model checkpoints')

    # --- 模型结构配置 (必须与训练时一致) ---
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=0, help='start token length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
    parser.add_argument('--enc_in', type=int, default=17, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=17, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=1, help='output size (1 for Singlevariate)')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=2, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=512, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--distil', action='store_true', default=True, help='whether to use distilling')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF', help='time features encoding')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--p_hidden_dims', type=int, nargs='+', default=[128, 128], help='projector hidden dims')
    parser.add_argument('--p_hidden_layers', type=int, default=2, help='projector hidden layers')

    # --- 设备配置 ---
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--gpu_type', type=str, default='cuda', help='gpu type')

    args = parser.parse_args([])

    if torch.cuda.is_available() and args.use_gpu:
        args.device = torch.device('cuda:{}'.format(args.gpu))
    else:
        args.device = torch.device("cpu")
    
    return args

def find_model_path(checkpoint_root, link_id, original_args):
    """
    根据原始 args 和 link_id，参照 run.py 的格式构建并查找模型文件路径。
    """
    # 创建一个 args 的副本，用于构建 setting
    args = argparse.Namespace(**original_args.__dict__.copy())

    base_model_id = "Traffic_Singlevariate" 
    args.model_id = f"{base_model_id}_{link_id}"
    args.expand = 2
    args.d_conv = 4
    args.des = 'test'
    ii = 0 # 实验次数 itr

    # 3. 参照原始 run.py 的格式构建 setting 字符串
    setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_expand{}_dc{}_fc{}_eb{}_dt{}_{}_{}'.format(
        args.task_name,
        args.model_id,
        args.model,
        args.data,
        args.features,
        args.seq_len,
        args.label_len,
        args.pred_len,
        args.d_model,
        args.n_heads,
        args.e_layers,
        args.d_layers,
        args.d_ff,
        args.expand,
        args.d_conv,
        args.factor,
        args.embed,
        args.distil,
        args.des, ii)
    
    # 处理布尔值，将 True/False 转为小写字符串，以匹配文件系统路径
    setting = setting.replace("dtTrue", "dttrue").replace("dtFalse", "dtfalse")

    model_dir = os.path.join(checkpoint_root, setting)
    model_path = os.path.join(model_dir, 'checkpoint.pth')
    
    if os.path.exists(model_path):
        return model_path
    else:
        print(f"警告: 模型文件未找到: {model_path}")
        return None

def load_and_preprocess_test_data(file_path, input_start, input_end, target_start, target_end, scaler=None, fit_scaler=True):
    """加载测试数据，截取指定时间范围，并进行预处理"""
    df_raw = pd.read_csv(file_path)
    
    input_mask = (df_raw['interval_begin'] >= input_start) & (df_raw['interval_begin'] <= input_end)
    df_input = df_raw[input_mask].copy()
    
    target_mask = (df_raw['interval_begin'] >= target_start) & (df_raw['interval_begin'] <= target_end)
    df_target = df_raw[target_mask].copy()

    if df_input.empty or df_target.empty:
        print(f"警告: 在文件 {os.path.basename(file_path)} 中未找到指定时间范围的数据。")
        return None, None, None, None

    drop_cols = ['file_id', 'file_name', 'interval_begin', 'interval_end', 'interval_id', 'id']
    drop_cols = [col for col in drop_cols if col in df_input.columns]
    
    df_input_features = df_input.drop(columns=drop_cols, errors='ignore')
    df_target_feature = df_target[[args.target]]

    if fit_scaler:
        scaler = StandardScaler()
        scaler.fit(df_input_features.values)
    
    input_data = scaler.transform(df_input_features.values)
    target_data = scaler.transform(df_target_feature.values)

    return input_data, target_data, scaler, df_target

def main():
    global args
    args = create_test_args()

    test_data_root = args.root_path
    checkpoint_root = args.checkpoints
    
    INPUT_START, INPUT_END = 0, 5700
    TARGET_START, TARGET_END = 5760, 11400

    all_metrics = {}

    print(f"开始测试，数据根目录: {test_data_root}")
    print(f"输入时间范围: {INPUT_START} - {INPUT_END}")
    print(f"目标时间范围: {TARGET_START} - {TARGET_END}")
    print("-" * 80)

    for filename in tqdm(os.listdir(test_data_root), desc="Processing test files"):
        if filename.endswith('.csv'):
            link_id = os.path.splitext(filename)[0]
            test_file_path = os.path.join(test_data_root, filename)
            
            print(f"\n正在处理: {filename} (Link ID: {link_id})")

            # 调用新的 find_model_path 函数
            model_path = find_model_path(checkpoint_root, link_id, args)
            if not model_path:
                continue

            input_data, target_data, scaler, df_target = load_and_preprocess_test_data(
                test_file_path, INPUT_START, INPUT_END, TARGET_START, TARGET_END
            )
            if input_data is None:
                continue

            if len(input_data) != args.seq_len or len(target_data) != args.pred_len:
                print(f"警告: {filename} 的数据长度不符合要求。输入: {len(input_data)}, 目标: {len(target_data)}。跳过此文件。")
                continue

            exp = Exp_Long_Term_Forecast(args)
            model = exp.model
            model.load_state_dict(torch.load(model_path, map_location=args.device), strict=False)
            model.to(args.device)
            model.eval()
            
            input_tensor = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0).to(args.device)
            dec_inp = torch.zeros((1, args.pred_len, args.dec_in)).float().to(args.device)
            seq_x_mark = torch.zeros((1, args.seq_len, 0)).to(args.device)
            seq_y_mark = torch.zeros((1, args.pred_len, 0)).to(args.device)

            with torch.no_grad():
                outputs = model(input_tensor, seq_x_mark, dec_inp, seq_y_mark)
            
            outputs = outputs[:, -args.pred_len:, :]
            
            pred = outputs.detach().cpu().numpy().reshape(-1, 1)
            true = target_data.reshape(-1, 1)

            pred_inv = scaler.inverse_transform(pred)
            true_inv = scaler.inverse_transform(true)

            mae = np.mean(np.abs(pred_inv - true_inv))
            rmse = np.sqrt(np.mean((pred_inv - true_inv)**2))
            
            all_metrics[link_id] = {'MAE': mae, 'RMSE': rmse}
            
            print(f"  - 模型路径: {model_path}")
            print(f"  - 测试结果: MAE = {mae:.4f}, RMSE = {rmse:.4f}")

    print("\n" + "="*80)
    print("所有链路测试完成，汇总结果如下:")
    print("="*80)
    avg_mae = 0
    avg_rmse = 0
    for link_id, metrics in all_metrics.items():
        print(f"Link {link_id}: MAE = {metrics['MAE']:.4f}, RMSE = {metrics['RMSE']:.4f}")
        avg_mae += metrics['MAE']
        avg_rmse += metrics['RMSE']
    
    if all_metrics:
        avg_mae /= len(all_metrics)
        avg_rmse /= len(all_metrics)
        print("-" * 80)
        print(f"平均 MAE: {avg_mae:.4f}")
        print(f"平均 RMSE: {avg_rmse:.4f}")
        
        results_df = pd.DataFrame.from_dict(all_metrics, orient='index')
        results_df.loc['Average'] = [avg_mae, avg_rmse]
        results_df.to_csv('test_results_summary.csv')
        print(f"\n结果已保存至 test_results_summary.csv")
    else:
        print("没有成功测试任何模型。")

if __name__ == '__main__':
    main()