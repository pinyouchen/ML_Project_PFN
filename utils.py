import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from joblib import dump, load  # 確保這裡有 load

# --- 強健的套件檢查機制 ---
_HAS_RICH = False
_HAS_TABULATE = False

try:
    from rich.console import Console
    from rich.table import Table
    _HAS_RICH = True
except ImportError:
    pass

try:
    from tabulate import tabulate
    _HAS_TABULATE = True
except ImportError:
    pass
# -------------------------

def specificity_npv(y_true, y_pred):
    """計算 Specificity 和 NPV"""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    if cm.shape != (2, 2):
        return np.nan, np.nan
    tn, fp, fn, tp = cm.ravel()
    spec = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    npv  = tn / (tn + fn) if (tn + fn) > 0 else np.nan
    return spec, npv

def pretty_print_table(df, title=None, float_cols=None, float_digits=4):
    """美化輸出表格 (自動降級：Rich -> Tabulate -> Print)"""
    if float_cols is None:
        float_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    df_show = df.copy()
    for c in float_cols:
        df_show[c] = df_show[c].astype(float).round(float_digits)

    # 優先嘗試 Rich
    if _HAS_RICH:
        try:
            console = Console()
            if title:
                console.rule(f"[bold]{title}")
            table = Table(show_header=True, header_style="bold", box=None, pad_edge=False)
            for c in df_show.columns:
                align = "right" if c in float_cols else "left"
                table.add_column(str(c), justify=align, no_wrap=True)
            for _, row in df_show.iterrows():
                row_vals = []
                for c in df_show.columns:
                    v = row[c]
                    if pd.isna(v):
                        row_vals.append("-")
                    elif c in float_cols:
                        row_vals.append(f"{float(v):.{float_digits}f}")
                    else:
                        row_vals.append(str(v))
                table.add_row(*row_vals)
            console.print(table)
            return
        except Exception:
            pass 

    # 其次嘗試 Tabulate
    if _HAS_TABULATE:
        try:
            print(f"\n--- {title} ---" if title else "")
            print(tabulate(df_show, headers="keys", tablefmt="github", showindex=False, floatfmt=f".{float_digits}f"))
            return
        except Exception:
            pass

    # 最後使用 Pandas 預設輸出
    print(f"\n--- {title} ---" if title else "")
    print(df_show)

def save_best_model(models_dir, label, model_obj, scaler, imputer,
                    feature_columns, outlier_bounds, threshold, fold_id=None):
    """儲存最佳模型與相關預處理物件"""
    os.makedirs(models_dir, exist_ok=True)
    base = f"{label}_best"
    model_path   = os.path.join(models_dir, base + ".joblib")
    scaler_path  = os.path.join(models_dir, base + "_scaler.joblib")
    imputer_path = os.path.join(models_dir, base + "_imputer.joblib")
    meta_path    = os.path.join(models_dir, base + ".json")

    dump(model_obj, model_path)
    if scaler is not None:
        dump(scaler, scaler_path)
    if imputer is not None:
        dump(imputer, imputer_path)

    meta = {
        "label": label,
        "threshold": float(threshold),
        "feature_columns": list(feature_columns),
        "outlier_bounds": outlier_bounds,
        "best_fold": fold_id,
        "files": {
            "model": os.path.basename(model_path),
            "scaler": os.path.basename(scaler_path) if scaler is not None else None,
            "imputer": os.path.basename(imputer_path) if imputer is not None else None,
        }
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"💾 已保存整體最佳模型：{model_path}")

def load_best_model_and_meta(models_dir, label):
    """
    載入指定 label 的最佳模型與其 Metadata
    """
    meta_path = os.path.join(models_dir, f"{label}_best.json")
    if not os.path.exists(meta_path):
        return None
    
    with open(meta_path, 'r', encoding='utf-8') as f:
        meta = json.load(f)
        
    files = meta.get('files', {})
    
    # 載入主要物件
    model_path = os.path.join(models_dir, files['model'])
    model = load(model_path)
    
    scaler = None
    if files.get('scaler'):
        scaler = load(os.path.join(models_dir, files['scaler']))
        
    imputer = None
    if files.get('imputer'):
        imputer = load(os.path.join(models_dir, files['imputer']))
        
    return {
        'model': model,
        'scaler': scaler,
        'imputer': imputer,
        'threshold': float(meta.get('threshold', 0.5)),
        'feature_columns': meta.get('feature_columns', []),
        'outlier_bounds': meta.get('outlier_bounds', {}),
        'meta_raw': meta
    }