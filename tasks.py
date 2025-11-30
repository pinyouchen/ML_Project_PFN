# tasks.py
import os
import copy
import json
import numpy as np
import pandas as pd
from datetime import datetime
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
# 修正：補齊所有需要的指標函式
from sklearn.metrics import (
    roc_curve, roc_auc_score, precision_recall_curve, auc,
    f1_score, accuracy_score, precision_score, recall_score
)
from joblib import load
import shap  # 確保安裝了 shap

# 引用我們自己的模組
from processors import DataProcessor
from model_trainer import ModelTrainer 
# 修正：補上 specificity_npv
from utils import pretty_print_table, save_best_model, load_best_model_and_meta, specificity_npv
from visualization import Visualizer 

def run_kfold_training(file_path, sheet_name, mode='basic_hrv'):
    """執行 5-Fold 訓練任務"""
    print("\n" + "="*70)
    print(f"🏁 啟動任務: Mode={mode}")
    print("="*70)

    # 建立 runs 資料夾
    timestamp = datetime.now().strftime(f"Run_{mode}_%Y%m%d_%H%M%S")
    runs_root = os.path.join(os.getcwd(), "runs") 
    os.makedirs(runs_root, exist_ok=True)         
    
    run_dir = os.path.join(runs_root, timestamp)
    models_dir = os.path.join(run_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    
    print(f"📂 輸出目錄: {run_dir}")

    # 1. 載入資料
    processor = DataProcessor(file_path, sheet_name, mode=mode)
    if not processor.load_data(): return
    if not processor.prepare_features_and_labels(): return

    X, y_dict = processor.X, processor.y_dict
    label_names = processor.label_names
    Y_multi = pd.concat([y_dict[lb] for lb in label_names], axis=1)

    # 2. 5-Fold Split
    mskf = MultilabelStratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # --- 統計與繪圖資料容器 ---
    summary_data = [] 
    
    # Global OOF SHAP 容器
    global_shap = {lb: {'vals': [], 'data': []} for lb in label_names}
    
    multi_label_metrics_list = []
    global_roc_data = {} 
    global_pr_data = {}

    label_fold_data = {lb: {
        'tprs': [], 'aucs': [], 'mean_fpr': np.linspace(0, 1, 100),
        'precisions': [], 'pr_aucs': [], 'mean_recall': np.linspace(0, 1, 100),
        'y_true_all': [], 'y_pred_all': [], 
        'metrics_list': [], 
        'feature_imp_list': [], 
        'X_all': [] 
    } for lb in label_names}

    overall_best = {lb: {"f1": -1.0, "model_obj": None} for lb in label_names}

    fold_id = 1
    for train_idx, test_idx in mskf.split(X, Y_multi):
        print(f"\n📂 Fold {fold_id}/5")
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        
        # Preprocessing
        X_train_p, X_test_p = processor.impute_and_scale(X_train, X_test, fit=True)
        
        for label in label_names:
            print(f"\n🔹 Label: {label}")
            y_train = y_dict[label].iloc[train_idx]
            y_test  = y_dict[label].iloc[test_idx]

            # 目標設定
            base_f1 = {'Health':0.79, 'SSD':0.66, 'MDD':0.47, 'Panic':0.50, 'GAD':0.58}.get(label, 0.5)
            target  = {'Health':0.80, 'SSD':0.65, 'MDD':0.50, 'Panic':0.55, 'GAD':0.65}.get(label, 0.7)
            
            # 訓練
            trainer = ModelTrainer(label, y_train.sum(), len(y_train)-y_train.sum(), base_f1, target)
            trainer.build_models()
            results = trainer.train_and_evaluate(X_train_p, X_test_p, y_train, y_test)

            # 選擇最佳結果
            chosen_name = 'Ensemble' if 'Ensemble' in results else max(results, key=lambda k: results[k]['f1_score'])
            res = results[chosen_name]
            
            # --- 收集數據 (Visualization) ---
            container = label_fold_data[label]
            
            # 1. Metrics
            metrics_dict = {
                'F1': res['f1_score'], 'Acc': res['accuracy'], 'AUC': res['auc'], 
                'Prec': res['precision'], 'Recall': res['recall'], 
                'Spec': res['specificity'], 'NPV': res['npv']
            }
            container['metrics_list'].append(metrics_dict)
            
            # 2. ROC Data (Interpolation)
            try:
                fpr, tpr, _ = roc_curve(y_test, res['y_pred_proba'])
                # 使用 np.interp
                interp_tpr = np.interp(container['mean_fpr'], fpr, tpr)
                interp_tpr[0] = 0.0
                container['tprs'].append(interp_tpr)
                container['aucs'].append(res['auc'])
            except: pass
            
            # 3. PR Data (Interpolation)
            try:
                precision, recall, _ = precision_recall_curve(y_test, res['y_pred_proba'])
                # 使用 np.interp
                interp_prec = np.interp(container['mean_recall'], recall[::-1], precision[::-1])
                container['precisions'].append(interp_prec)
                container['pr_aucs'].append(auc(recall, precision))
            except: pass

            # 4. CM Data
            container['y_true_all'].extend(y_test)
            container['y_pred_all'].extend(res['y_pred'])
            container['X_all'].append(X_test_p) 
            
            # 5. Feature Importance & SHAP (Global OOF)
            # 優先使用 RF 模型計算 SHAP，因為 TreeExplainer 支援最好
            # 其次使用當前最佳單模 (如果是 Tree-based)
            model_for_shap = None
            
            # 從 trainer.fitted_models 取得訓練好的模型實例
            if 'RF' in trainer.fitted_models:
                model_for_shap = trainer.fitted_models['RF']
                # 同時收集 Feature Importance
                if hasattr(model_for_shap, 'feature_importances_'):
                    imp_df = pd.DataFrame({
                        'Feature': X_test_p.columns,
                        'Importance': model_for_shap.feature_importances_
                    })
                    container['feature_imp_list'].append(imp_df)
            elif chosen_name != 'Ensemble' and chosen_name in trainer.fitted_models:
                 model = trainer.fitted_models[chosen_name]
                 if hasattr(model, 'feature_importances_'):
                     model_for_shap = model

            # 計算並收集當折的 SHAP 值
            if model_for_shap is not None:
                try:
                    explainer = shap.TreeExplainer(model_for_shap)
                    shap_vals = explainer.shap_values(X_test_p)
                    
                    # === 修正部分：嚴格確保 2D 格式 (Samples, Features) ===
                    # 情況 1: 回傳 List (Scikit-Learn RF) -> [Neg, Pos] -> 取 index 1
                    if isinstance(shap_vals, list):
                        shap_vals = shap_vals[1]
                    
                    # 情況 2: 回傳 3D Array (XGBoost/LGBM 某些版本) -> (Samples, Features, Class) -> 取 index 1
                    elif isinstance(shap_vals, np.ndarray) and shap_vals.ndim == 3:
                        # 檢查最後一維是否為類別數 (通常是 2)
                        if shap_vals.shape[2] == 2:
                            shap_vals = shap_vals[:, :, 1]
                        else:
                            # 異常情況，保留原樣或報錯
                            pass
                        
                    # 存入 global 容器
                    global_shap[label]['vals'].append(shap_vals)
                    global_shap[label]['data'].append(X_test_p)
                except Exception:
                    pass # 若模型不支援或報錯則略過

            # --- 更新整體最佳單一模型 ---
            for mname in [m for m in results if m != 'Ensemble']:
                r = results[mname]
                if r['f1_score'] > overall_best[label]['f1']:
                    overall_best[label] = {
                        "f1": r['f1_score'],
                        "model_name": mname,
                        "model_obj": r['model'],
                        "threshold": r['threshold'],
                        "scaler": copy.deepcopy(processor.scaler),
                        "imputer": copy.deepcopy(processor.knn_imputer),
                        "outlier_bounds": copy.deepcopy(processor.outlier_bounds_),
                        "feature_columns": list(processor.X.columns),
                        "fold": fold_id,
                        # Store metrics for table
                        "prec": r['precision'], "recall": r['recall'],
                        "spec": r['specificity'], "npv": r['npv'],
                        "auc": r['auc'], "acc": r['accuracy']
                    }
        
        fold_id += 1

    # --- 3. 繪圖與匯總 ---
    print("\n" + "="*70 + "\n🏁 最終彙整與繪圖\n" + "="*70)
    
    viz_summary = Visualizer("Summary", run_dir)
    
    for label in label_names:
        print(f"📊 Processing {label}...")
        container = label_fold_data[label]
        best = overall_best[label]
        
        viz = Visualizer(label, run_dir)
        
        if best['model_obj']:
            save_best_model(models_dir, label, best['model_obj'], best['scaler'], best['imputer'],
                           best['feature_columns'], best['outlier_bounds'], best['threshold'], best['fold'])

        df_metrics = pd.DataFrame(container['metrics_list'])
        
        # 只對數值欄位做平均
        metric_cols = ['F1', 'Acc', 'AUC', 'Prec', 'Recall', 'Spec', 'NPV']
        valid_metric_cols = [c for c in metric_cols if c in df_metrics.columns]
        avg_metrics = df_metrics[valid_metric_cols].mean()
        std_metrics = df_metrics[valid_metric_cols].std()
        
        row = {
            "Label": label,
            "BestModel": best.get('model_name', 'N/A'),
            "F1(Best)": best['f1'],
            "P(Best)": best.get('prec', 0),
            "R(Best)": best.get('recall', 0),
            "Spec(Best)": best.get('spec', 0),
            "NPV(Best)": best.get('npv', 0),
            "AUC(Best)": best.get('auc', 0),
            "ACC(Best)": best.get('acc', 0),
            "F1(avg)": avg_metrics.get('F1', 0),
            "P(avg)": avg_metrics.get('Prec', 0),
            "R(avg)": avg_metrics.get('Recall', 0),
            "Spec(avg)": avg_metrics.get('Spec', 0),
            "NPV(avg)": avg_metrics.get('NPV', 0),
            "AUC(avg)": avg_metrics.get('AUC', 0),
            "ACC(avg)": avg_metrics.get('Acc', 0)
        }
        summary_data.append(row)
        
        # Plots
        metrics_summary = pd.DataFrame({
            'Metric': valid_metric_cols,
            'Mean': avg_metrics.values,
            'Std': std_metrics.values
        })
        viz.plot_performance_metrics(metrics_summary)
        
        for _, m_row in metrics_summary.iterrows():
            multi_label_metrics_list.append({
                'Label': label, 'Metric': m_row['Metric'], 
                'Mean': m_row['Mean'], 'Std': m_row['Std']
            })

        if container['tprs']:
            viz.plot_roc_curve_with_ci(container['tprs'], container['mean_fpr'], container['aucs'])
            mean_tpr = np.mean(container['tprs'], axis=0); mean_tpr[-1] = 1.0
            mean_auc = auc(container['mean_fpr'], mean_tpr)
            global_roc_data[label] = (container['mean_fpr'], mean_tpr, mean_auc)

        if container['precisions']:
            no_skill = np.sum(container['y_true_all']) / len(container['y_true_all'])
            viz.plot_pr_curve_with_ci(container['precisions'], container['mean_recall'], container['pr_aucs'], no_skill)
            mean_prec = np.mean(container['precisions'], axis=0)
            mean_pr_auc = np.mean(container['pr_aucs'])
            global_pr_data[label] = (container['mean_recall'], mean_prec, mean_pr_auc)

        viz.plot_confusion_matrix_aggregated(container['y_true_all'], container['y_pred_all'])

        if container['X_all']:
            X_concat = pd.concat(container['X_all'], axis=0)
            y_concat = np.array(container['y_true_all'])
            if len(X_concat) > 1000:
                idx = np.random.choice(len(X_concat), 1000, replace=False)
                X_plot, y_plot = X_concat.iloc[idx], y_concat[idx]
            else:
                X_plot, y_plot = X_concat, y_concat
            viz.plot_pca_scatter(X_plot, y_plot)
            viz.plot_correlation_matrix(X_concat)

        if container['feature_imp_list']:
            all_imp = pd.concat(container['feature_imp_list'], axis=0)
            viz.plot_feature_importance_boxplot(all_imp, top_n=20)

        viz.plot_radar_chart(avg_metrics.to_dict())

        # [NEW] EBM Shape Plot (如果最佳模型是 EBM)
        if best.get('model_name') == 'EBM' and best['model_obj']:
            try:
                viz.plot_ebm_detail(best['model_obj'])
            except Exception as e:
                print(f"   ⚠️ EBM Plot Error: {e}")

        # [關鍵修改] 繪製 Global OOF SHAP
        if global_shap[label]['vals']:
            print(f"   ℹ️ Drawing SHAP Global OOF for {label}...")
            try:
                # 串接所有 Fold 的 SHAP 值與數據
                shap_vals_all = np.concatenate(global_shap[label]['vals'], axis=0)
                shap_data_all = pd.concat(global_shap[label]['data'], axis=0)
                
                # 二次安全檢查：如果串接後變成了 3D，再次嘗試降維 (Double Safety)
                if shap_vals_all.ndim == 3 and shap_vals_all.shape[2] == 2:
                    shap_vals_all = shap_vals_all[:, :, 1]
                
                # 呼叫視覺化
                viz.plot_shap_summary_oof(shap_vals_all, shap_data_all)
            except Exception as e:
                print(f"   ⚠️ SHAP Concatenate/Plot Error: {e}")

    # --- 4. 繪製 Global Comparison Charts ---
    print("📊 Generating Summary Comparison Plots...")
    
    if multi_label_metrics_list:
        df_all_metrics = pd.DataFrame(multi_label_metrics_list)
        viz_summary.plot_multilabel_metrics(df_all_metrics)
    
    if global_roc_data:
        viz_summary.plot_multilabel_roc(global_roc_data)
        
    if global_pr_data:
        viz_summary.plot_multilabel_pr(global_pr_data)

    # --- 5. 輸出表格 ---
    df_summary = pd.DataFrame(summary_data)
    cols_order = [
        "Label", "BestModel", 
        "F1(Best)", "P(Best)", "R(Best)", "Spec(Best)", "NPV(Best)", "AUC(Best)", "ACC(Best)",
        "F1(avg)", "P(avg)", "R(avg)", "Spec(avg)", "NPV(avg)", "AUC(avg)", "ACC(avg)"
    ]
    # 防呆：只選存在的欄位
    cols_order = [c for c in cols_order if c in df_summary.columns]
    df_summary = df_summary[cols_order]
    
    df_summary.to_excel(os.path.join(run_dir, "Final_Summary_Table.xlsx"), index=False)
    pretty_print_table(df_summary, title="Final Performance Summary")
    
    print(f"\n✅ 任務完成，結果已存至: {run_dir}")


def run_external_validation(train_run_dir, data_file, sheet_name):
    """執行外部驗證任務 (Ref File 5)"""
    print("\n" + "="*70)
    print("🏁 外部驗證任務")
    print(f"   Model Dir: {train_run_dir}")
    print(f"   Target Data: {sheet_name}")
    print("="*70)
    
    models_dir = os.path.join(train_run_dir, "models")
    if not os.path.exists(models_dir):
        print("❌ 找不到 models 資料夾"); return

    timestamp = datetime.now().strftime("External_Val_%Y%m%d_%H%M%S")
    out_dir = os.path.join(train_run_dir, timestamp)
    os.makedirs(out_dir, exist_ok=True)

    df_test = pd.read_excel(data_file, sheet_name=sheet_name)
    label_names = ['Health', 'SSD', 'MDD', 'Panic', 'GAD']
    
    # 這裡 mode 設為 all 以便能產生所有可能的 engineering features
    processor = DataProcessor(data_file, sheet_name, mode='all') 
    
    # 手動給 dataframe 以便 prepare_features 能運作
    processor.df = df_test
    # 產生衍生特徵 (HRV_Mean, Ratio 等)
    processor.prepare_features_and_labels()

    metrics_rows = []
    
    for label in label_names:
        meta_path = os.path.join(models_dir, f"{label}_best.json")
        if not os.path.exists(meta_path): 
            print(f"⚠️ Skip {label} (No model found)")
            continue
        
        with open(meta_path, 'r', encoding='utf-8') as f: meta = json.load(f)
        
        model = load(os.path.join(models_dir, meta['files']['model']))
        scaler = load(os.path.join(models_dir, meta['files']['scaler'])) if meta['files']['scaler'] else None
        imputer = load(os.path.join(models_dir, meta['files']['imputer'])) if meta['files']['imputer'] else None
        
        # 套用訓練時的轉換 (Impute, Scale, Outlier)
        X_test = processor.apply_external_transform(
            processor.X, meta['feature_columns'], meta['outlier_bounds'], imputer, scaler
        )
        
        try: proba = model.predict_proba(X_test)[:, 1]
        except: proba = model.decision_function(X_test)
        
        y_pred = (proba >= meta['threshold']).astype(int)
        
        res = {"Label": label}
        if label in df_test.columns:
            y_true = df_test[label].values.astype(int)
            
            # --- 完整計算 7 個指標 ---
            f1 = f1_score(y_true, y_pred)
            acc = accuracy_score(y_true, y_pred)
            prec = precision_score(y_true, y_pred, zero_division=0)
            rec = recall_score(y_true, y_pred, zero_division=0)
            spec, npv = specificity_npv(y_true, y_pred)
            try:
                auc_val = roc_auc_score(y_true, proba)
            except:
                auc_val = np.nan

            res.update({
                "F1": f1,
                "Precision": prec,
                "Recall": rec,
                "Spec": spec,
                "NPV": npv,
                "AUC": auc_val,
                "ACC": acc
            })
        
        metrics_rows.append(res)
        print(f"   -> {label}: Done.")
        
    res_df = pd.DataFrame(metrics_rows)
    
    # 指定輸出欄位順序
    cols_order = ["Label", "F1", "Precision", "Recall", "Spec", "NPV", "AUC", "ACC"]
    cols_order = [c for c in cols_order if c in res_df.columns]
    res_df = res_df[cols_order]

    res_df.to_excel(os.path.join(out_dir, "External_Metrics.xlsx"), index=False)
    pretty_print_table(res_df, title="External Validation Result")
    print(f"\n✅ 外部驗證完成，結果已存至: {out_dir}")