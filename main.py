# main.py
import argparse
import os
import tasks

# ================= 設定區 =================
# 請在這裡修改您的檔案路徑
DEFAULT_FILE_PATH = r"D:\ML_Project\dataset\data.xlsx"
SHEET_DATA2 = "Data2"  # 訓練資料
SHEET_DATA1 = "Data1"  # 外部驗證資料
# ==========================================

def main():
    parser = argparse.ArgumentParser(description="Multimorbidity Prediction Tasks")
    
    # 定義指令參數
    parser.add_argument('--task', type=str, required=True, choices=['train', 'validate'],
                        help="選擇任務: 'train' (訓練模型) 或 'validate' (外部驗證)")
    
    parser.add_argument('--mode', type=str, default='all', 
                        choices=['basic_hrv', 'extended_hrv', 'psych', 'all'],
                        help="訓練模式 (僅在 --task train 時有效)")
    
    parser.add_argument('--model_dir', type=str, default=None,
                        help="模型資料夾路徑 (僅在 --task validate 時必填，例如 'Run_all_20251120_...')")

    args = parser.parse_args()

    # 執行邏輯
    if args.task == 'train':
        print(f"🚀 開始訓練任務: Mode = {args.mode}")
        tasks.run_kfold_training(DEFAULT_FILE_PATH, SHEET_DATA2, mode=args.mode)
        
    elif args.task == 'validate':
        if not args.model_dir:
            print("❌ 錯誤: 執行外部驗證時，必須提供 --model_dir 參數 (指向訓練好的資料夾)")
            return
        
        if not os.path.exists(args.model_dir):
            print(f"❌ 錯誤: 找不到資料夾: {args.model_dir}")
            return
            
        print(f"🚀 開始外部驗證任務: Model Dir = {args.model_dir}")
        tasks.run_external_validation(args.model_dir, DEFAULT_FILE_PATH, SHEET_DATA1)

if __name__ == "__main__":
    main()