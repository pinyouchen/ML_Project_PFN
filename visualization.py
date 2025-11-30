# visualization.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import auc, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from math import pi

# 設定學術風格
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
plt.rcParams['font.family'] = 'sans-serif'
# 嘗試設定常用字體，避免某些系統報錯
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans', 'Bitstream Vera Sans', 'sans-serif']

class Visualizer:
    def __init__(self, label, out_dir, sub_folder=None):
        """
        label: 標籤名稱 (如 SSD, MDD) 或 'Summary'
        out_dir: 根輸出目錄
        sub_folder: 強制指定子資料夾名稱 (若 None 則預設為 label)
        """
        self.label = label
        # 如果是 Summary，就放在 plots/Summary_Comparison
        # 如果是單一 label，就放在 plots/SSD
        folder_name = sub_folder if sub_folder else label
        self.out_dir = os.path.join(out_dir, "plots", folder_name)
        os.makedirs(self.out_dir, exist_ok=True)
        self.colors = sns.color_palette("deep")
        # 為 4 個 label 準備固定顏色，方便比較
        self.label_colors = {
            'SSD': 'C0', 'MDD': 'C1', 'Panic': 'C2', 'GAD': 'C3',
            'Health': 'C4', 'Comparison': 'black'
        }

    def save_fig(self, fig, name):
        # 檔名包含 label 以免搞混
        filename = f"{name}_{self.label}.png"
        path_png = os.path.join(self.out_dir, filename)
        fig.savefig(path_png, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"   📊 圖表已儲存: {path_png}")

    # ===========================
    # 單一模型/疾病的圖表 (放在各 label 資料夾)
    # ===========================

    def plot_pca_scatter(self, X_data, y_data):
        """
        圖0: PCA 散佈圖 (2 Components)
        """
        try:
            # 標準化 (PCA 前必要步驟)
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_data)
            
            # PCA
            pca = PCA(n_components=2, random_state=42)
            X_pca = pca.fit_transform(X_scaled)
            
            fig, ax = plt.subplots(figsize=(8, 6))
            
            # 繪製 Health (0)
            mask_health = (y_data == 0)
            ax.scatter(X_pca[mask_health, 0], X_pca[mask_health, 1], 
                       color=self.colors[0], label='Health', alpha=0.6, s=30, edgecolor='w', linewidth=0.5)
            
            # 繪製 Disease (1)
            mask_disease = (y_data == 1)
            col_disease = self.label_colors.get(self.label, self.colors[3])
            ax.scatter(X_pca[mask_disease, 0], X_pca[mask_disease, 1], 
                       color=col_disease, label=f'{self.label}', alpha=0.7, s=30, edgecolor='w', linewidth=0.5)
            
            ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} var)')
            ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} var)')
            ax.set_title(f'PCA Visualization ({self.label})')
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.5)
            
            self.save_fig(fig, "PCA_Scatter")
            
        except Exception as e:
            print(f"   ⚠️ PCA 繪圖失敗: {e}")

    def plot_shap_summary_oof(self, shap_values, X_test):
        """
        Global OOF SHAP Summary Plot
        修正：確保正確抓取 SHAP 繪製的 Figure，解決空白圖問題
        """
        try:
            import shap
            
            # 1. 先清空當前的繪圖環境，避免殘留
            plt.close('all')
            
            # 2. 讓 SHAP 繪圖，並設定 show=False 防止直接彈出視窗
            # 注意：不使用 plt.figure() 預先建立，因為 shap 會自己建立 figure
            shap.summary_plot(shap_values, X_test, show=False, max_display=30)
            
            # 3. 關鍵步驟：抓取當前活動中的 Figure (這就是 shap 剛剛畫的那張)
            fig = plt.gcf()
            fig.set_size_inches(10, 12) # 在這裡調整大小
            
            # 4. 添加標題並調整版面
            plt.title(f"SHAP Summary (Global OOF) - {self.label}", fontsize=16, y=1.02)
            plt.tight_layout()
            
            # 5. 存檔
            self.save_fig(fig, "SHAP_Summary_Global_OOF")
            
        except Exception as e:
            print(f"   ⚠️ SHAP 繪圖失敗: {e}")

    # ==========================================
    # [NEW] EBM Shape Function Plotter
    # ==========================================
    def plot_ebm_detail(self, ebm_model):
        """
        繪製 EBM 模型的 Shape Functions (特徵貢獻圖)
        支援連續型 (Step Plot) 與 類別型 (Bar Chart)
        """
        try:
            # 取得解釋物件
            ebm_global = ebm_model.explain_global()
            
            print(f"   ℹ️ Drawing EBM Shape Plots for {self.label}...")
            
            # 遍歷前 15 個最重要的特徵 (避免圖太多)
            # 先根據重要性排序
            importances = ebm_model.term_importances()
            sorted_indices = np.argsort(importances)[::-1][:15] # Top 15
            
            for idx in sorted_indices:
                feature_name = ebm_global.feature_names[idx]
                feature_type = ebm_global.feature_types[idx]
                data = ebm_global.data(idx)
                
                fig, ax = plt.subplots(figsize=(8, 5))
                
                # X軸數值 與 Y軸分數
                x_vals = data['names']
                y_vals = data['scores']
                
                # 處理信賴區間 (如果有)
                upper = data.get('upper_bounds', np.zeros_like(y_vals))
                lower = data.get('lower_bounds', np.zeros_like(y_vals))
                
                if feature_type == 'continuous':
                    # 連續特徵：畫階梯圖
                    # EBM 的 x_vals 是 bin 的邊界，y_vals 是 bin 的值
                    # 為了畫出正確的階梯，我們使用 step(where='post')
                    # 有時候 x_vals 長度會比 y_vals 多 1 (bin edges)，需調整
                    if len(x_vals) == len(y_vals) + 1:
                        plot_x = x_vals[:-1]
                    else:
                        plot_x = x_vals
                        
                    ax.step(plot_x, y_vals, where='post', color=self.colors[3], linewidth=2, label='Score')
                    ax.fill_between(plot_x, lower, upper, step='post', alpha=0.2, color=self.colors[3], label='Confidence')
                    ax.set_xlabel(f"Feature Value: {feature_name}")
                    
                    # 加入密度分佈 (Density) 在底部
                    if 'density' in data:
                        # 創建雙軸
                        ax2 = ax.twinx()
                        ax2.bar(plot_x, data['density']['scores'], width=np.diff(x_vals)[0] if len(x_vals)>1 else 1, 
                                align='edge', alpha=0.1, color='gray')
                        ax2.set_yticks([]) # 隱藏密度軸刻度
                        
                elif feature_type == 'categorical':
                    # 類別特徵：畫長條圖
                    x_pos = np.arange(len(x_vals))
                    ax.bar(x_pos, y_vals, yerr=[y_vals-lower, upper-y_vals], 
                           align='center', alpha=0.6, color=self.colors[3], capsize=5, edgecolor='black')
                    ax.set_xticks(x_pos)
                    ax.set_xticklabels(x_vals, rotation=45, ha='right')
                    ax.set_xlabel(feature_name)
                    
                else:
                    # 交互作用項 (Interaction) 或其他，暫時跳過
                    plt.close(fig)
                    continue

                ax.set_ylabel('Contribution to Score (Log Odds)')
                ax.set_title(f'EBM Feature Contribution: {feature_name}', fontsize=14)
                ax.axhline(0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
                ax.grid(True, linestyle='--', alpha=0.5)
                
                # 存檔 (處理檔名中的特殊字元)
                safe_name = feature_name.replace(" x ", "_X_").replace(" ", "_").replace("/", "_")
                self.save_fig(fig, f"EBM_Shape_{safe_name}")
                
        except Exception as e:
            print(f"   ⚠️ EBM Plotting Failed: {e}")

    def plot_performance_metrics(self, df_metrics):
        if df_metrics.empty: return
        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(df_metrics))
        # 依照 label 選擇顏色，若無則預設 C0
        col = self.label_colors.get(self.label, self.colors[0])
        
        ax.bar(x, df_metrics['Mean'], yerr=df_metrics['Std'], 
               align='center', alpha=0.8, ecolor='black', capsize=10, 
               color=col, width=0.6)
        
        ax.set_ylabel('Score')
        ax.set_xticks(x)
        ax.set_xticklabels(df_metrics['Metric'])
        ax.set_title(f'Performance Metrics ({self.label})')
        ax.set_ylim(0, 1.05)
        for i, v in enumerate(df_metrics['Mean']):
            ax.text(i, v + 0.05, f"{v:.3f}", ha='center', fontweight='bold')
        self.save_fig(fig, "Metrics_Bar")

    def plot_roc_curve_with_ci(self, tprs, mean_fpr, aucs):
        fig, ax = plt.subplots(figsize=(8, 6))
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)
        
        col = self.label_colors.get(self.label, self.colors[0])
        ax.plot(mean_fpr, mean_tpr, color=col,
                label=f'Mean ROC (AUC={mean_auc:.3f} $\pm${std_auc:.3f})',
                lw=2, alpha=.8)
        
        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color=col, alpha=.2,
                        label=r'$\pm$ 1 std. dev.')
        
        ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='grey', alpha=.8)
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(f'ROC Curve ({self.label})')
        ax.legend(loc="lower right")
        self.save_fig(fig, "ROC_Curve")

    def plot_pr_curve_with_ci(self, precisions, mean_recall, pr_aucs, no_skill):
        fig, ax = plt.subplots(figsize=(8, 6))
        
        mean_precision = np.mean(precisions, axis=0)
        mean_auc = np.mean(pr_aucs)
        std_auc = np.std(pr_aucs)
        
        col = self.label_colors.get(self.label, self.colors[1])
        ax.plot(mean_recall, mean_precision, color=col,
                label=f'Mean PR (AUC={mean_auc:.3f} $\pm${std_auc:.3f})',
                lw=2, alpha=.8)
        
        std_precision = np.std(precisions, axis=0)
        upper = np.minimum(mean_precision + std_precision, 1)
        lower = np.maximum(mean_precision - std_precision, 0)
        ax.fill_between(mean_recall, lower, upper, color=col, alpha=.2)
        
        ax.plot([0, 1], [no_skill, no_skill], linestyle='--', lw=2, color='grey', label='No Skill')
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title(f'PR Curve ({self.label})')
        ax.legend(loc="lower left")
        self.save_fig(fig, "PR_Curve")

    def plot_confusion_matrix_aggregated(self, y_true_all, y_pred_all):
        cm = confusion_matrix(y_true_all, y_pred_all)
        # 避免除以 0
        with np.errstate(divide='ignore', invalid='ignore'):
            cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            cm_norm = np.nan_to_num(cm_norm) # 將 NaN 轉為 0
        
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm_norm, annot=False, cmap='Blues', cbar=True, ax=ax)
        
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                count = cm[i, j]
                pct = cm_norm[i, j] * 100
                col = "white" if cm_norm[i, j] > 0.5 else "black"
                ax.text(j + 0.5, i + 0.5, f"{count}\n({pct:.1f}%)",
                        ha="center", va="center", color=col, fontweight='bold')
        
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_xticklabels(['Health', 'Disease'])
        ax.set_yticklabels(['Health', 'Disease'])
        ax.set_title(f'Confusion Matrix ({self.label})')
        self.save_fig(fig, "CM_Aggregated")

    def plot_radar_chart(self, metrics_dict):
        """
        圖5: 雷達圖 (Mean Metrics)
        metrics_dict: {'F1': 0.8, 'Acc': 0.9, ...}
        """
        categories = list(metrics_dict.keys())
        values = list(metrics_dict.values())
        N = len(categories)
        
        # 封閉多邊形
        values += values[:1]
        angles = [n / float(N) * 2 * pi for n in range(N)]
        angles += angles[:1]
        
        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
        
        ax.set_theta_offset(pi / 2)
        ax.set_theta_direction(-1)
        
        plt.xticks(angles[:-1], categories)
        ax.set_rlabel_position(0)
        plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], ["0.2", "0.4", "0.6", "0.8", "1.0"], color="grey", size=8)
        plt.ylim(0, 1)
        
        col = self.label_colors.get(self.label, self.colors[2])
        ax.plot(angles, values, linewidth=2, linestyle='solid', color=col)
        ax.fill(angles, values, color=col, alpha=0.25)
        
        plt.title(f"Performance Radar ({self.label})", y=1.1)
        self.save_fig(fig, "Radar_Chart")

    def plot_feature_importance_boxplot(self, importance_df, top_n=20):
        if importance_df.empty: return
        order = importance_df.groupby('Feature')['Importance'].mean().sort_values(ascending=False).index[:top_n]
        subset = importance_df[importance_df['Feature'].isin(order)]
        
        fig, ax = plt.subplots(figsize=(10, 10))
        sns.boxplot(data=subset, x='Importance', y='Feature', order=order, palette="viridis", ax=ax)
        ax.set_title(f'Top {top_n} Feature Importance ({self.label})')
        self.save_fig(fig, "Feature_Importance")

    def plot_multilabel_metrics(self, all_metrics_df):
        fig, ax = plt.subplots(figsize=(12, 7))
        labels = all_metrics_df['Label'].unique()
        metrics = all_metrics_df['Metric'].unique()
        
        x = np.arange(len(metrics))
        width = 0.2  # bar 寬度
        
        for i, lbl in enumerate(labels):
            subset = all_metrics_df[all_metrics_df['Label'] == lbl]
            subset = subset.set_index('Metric').reindex(metrics).reset_index()
            
            offset = (i - len(labels)/2) * width + width/2
            ax.bar(x + offset, subset['Mean'], width, yerr=subset['Std'], 
                   label=lbl, color=self.label_colors.get(lbl, 'grey'), capsize=5, edgecolor='black')
            
        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
        ax.set_ylabel('Score')
        ax.set_title('Comparison of Metrics Across Diseases')
        ax.set_ylim(0, 1.05)
        ax.legend(loc='lower right')
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        
        self.save_fig(fig, "MultiLabel_Metrics_Comparison")

    def plot_multilabel_roc(self, roc_data_dict):
        fig, ax = plt.subplots(figsize=(9, 7))
        for label, (fpr, tpr, auc_val) in roc_data_dict.items():
            col = self.label_colors.get(label, 'black')
            ax.plot(fpr, tpr, label=f'{label} (AUC = {auc_val:.3f})', 
                    color=col, lw=2.5)
            
        ax.plot([0, 1], [0, 1], 'k--', lw=2, alpha=0.7)
        ax.set_xlim([-0.05, 1.05])
        ax.set_ylim([-0.05, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curve Comparison')
        ax.legend(loc="lower right")
        self.save_fig(fig, "MultiLabel_ROC_Comparison")

    def plot_multilabel_pr(self, pr_data_dict):
        fig, ax = plt.subplots(figsize=(9, 7))
        for label, (rec, prec, auc_val) in pr_data_dict.items():
            col = self.label_colors.get(label, 'black')
            ax.plot(rec, prec, label=f'{label} (AUC = {auc_val:.3f})', 
                    color=col, lw=2.5)
            
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('PR Curve Comparison')
        ax.legend(loc="lower left")
        self.save_fig(fig, "MultiLabel_PR_Comparison")

    def plot_correlation_matrix(self, df, method='pearson'):
        """
        繪製特徵相關係數矩陣熱力圖
        """
        if df.empty: return
        
        # 計算相關係數
        corr = df.corr(method=method)
        
        # 設定圖表大小 (根據特徵數量自動調整)
        n_features = len(df.columns)
        figsize = (min(20, max(10, n_features * 0.8)), min(18, max(8, n_features * 0.8)))
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # 產生遮罩 (只顯示下半三角形，讓圖更乾淨)
        mask = np.triu(np.ones_like(corr, dtype=bool))
        
        # 繪製熱力圖
        sns.heatmap(
            corr, mask=mask, cmap='coolwarm', vmin=-1, vmax=1, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5},
            annot=False, # 特徵多時不顯示數字，以免太亂
            fmt=".2f", ax=ax
        )
        
        ax.set_title(f'Feature Correlation Matrix ({self.label})', fontsize=16)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        self.save_fig(fig, "Correlation_Matrix")