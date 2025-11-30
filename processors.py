# processors.py
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer

class DataProcessor:
    def __init__(self, file_path, sheet_name='Data2', 
                 mode='all', # mode: 'basic_hrv', 'extended_hrv', 'psych', 'all'
                 iqr_multiplier=3.0, 
                 treat_zero_as_missing_in_hrv=True):
        
        self.file_path = file_path
        self.sheet_name = sheet_name
        self.mode = mode
        self.iqr_multiplier = iqr_multiplier
        self.treat_zero_as_missing_in_hrv = treat_zero_as_missing_in_hrv

        # 定義特徵群組
        self.basic_features = ['Age', 'Sex', 'BMI']
        self.label_names = ['Health', 'SSD', 'MDD', 'Panic', 'GAD']
        
        # 根據 mode 決定要載入哪些特徵
        self.hrv_features = []
        self.psych_features = []
        self.log_hrv_cols = []
        self.log_engineered_cols = []

        # 1. Basic HRV (File: test2_data2_binary_Multimorbidity.py)
        if mode == 'basic_hrv':
            self.hrv_features = ['SDNN', 'LF', 'HF', 'LFHF']
            self.log_hrv_cols = ['LF', 'HF', 'LFHF']
            self.log_engineered_cols = ['HRV_Mean', 'LF_HF_Ratio']

        # 2. Extended HRV (File: test2_data2_binary_hrv_Multimorbidity.py)
        elif mode == 'extended_hrv':
            self.hrv_features = ['SDNN', 'LF', 'HF', 'LFHF', 'MEANH', 'TP', 'VLF', 'NLF']
            self.log_hrv_cols = ['LF', 'HF', 'LFHF', 'TP', 'VLF', 'NLF']
            self.log_engineered_cols = [
                'HRV_Mean', 'LF_HF_Ratio', 'Sympathetic_Index', 'Parasympathetic_Index',
                'HF_TP_Ratio', 'SDNN_MEANH_Ratio', 'GAD_Risk'
            ]

        # 3. Psych Only (File: test2_data2_binary_psych_Multimorbidity.py)
        elif mode == 'psych':
            self.psych_features = ['phq15', 'haq21', 'cabah', 'bdi', 'bai']
            # Psych 不做 log1p, 也不做 HRV 工程特徵
        
        # 4. All Combined (File: test2_data2_binary_all_Multimorbidity.py)
        elif mode == 'all':
            self.hrv_features = ['SDNN', 'LF', 'HF', 'LFHF', 'MEANH', 'TP', 'VLF', 'NLF']
            self.psych_features = ['phq15', 'haq21', 'cabah', 'bdi', 'bai']
            self.log_hrv_cols = ['LF', 'HF', 'LFHF', 'TP', 'VLF', 'NLF']
            self.log_engineered_cols = ['HRV_Mean', 'LF_HF_Ratio'] 
            # 注意：all 版本原始碼中沒有 GAD_Risk 等進階工程特徵，保持與原始碼一致

        self.df = None
        self.X = None
        self.y_dict = {}
        
        # 狀態儲存
        self.knn_imputer = None
        self.scaler = None
        self.outlier_bounds_ = None

    def load_data(self):
        try:
            self.df = pd.read_excel(self.file_path, sheet_name=self.sheet_name)
            print(f"✓ 載入: {self.df.shape[0]} 筆（工作表：{self.sheet_name}，模式：{self.mode}）")
            return True
        except Exception as e:
            print(f"❌ {e}")
            return False

    def prepare_features_and_labels(self):
        all_cols_needed = self.basic_features + self.hrv_features + self.psych_features
        available = [f for f in all_cols_needed if f in self.df.columns]
        self.X = self.df[available].copy()

        print(f"\n🔨 特徵工程 ({self.mode})...")

        # 通用工程特徵
        hrv_cols_present = [c for c in self.hrv_features if c in self.X.columns]
        if len(hrv_cols_present) >= 3:
             # 注意：Basic HRV 與 Extended HRV 的 mean 計算欄位不同，這裡自動根據存在的欄位算
            self.X['HRV_Mean'] = self.X[hrv_cols_present].mean(axis=1)
            
        if 'LF' in self.X.columns and 'HF' in self.X.columns:
            self.X['LF_HF_Ratio'] = self.X['LF'] / (self.X['HF'] + 1e-6)
            self.X['LF_HF_Ratio'].replace([np.inf, -np.inf], np.nan, inplace=True)

        # 僅在 Extended HRV 模式下特有的特徵 (參照 hrv_Multimorbidity.py)
        if self.mode == 'extended_hrv':
            if 'LF' in self.X.columns and 'HF' in self.X.columns:
                denom = (self.X['LF'] + self.X['HF'] + 1e-6)
                self.X['Sympathetic_Index'] = self.X['LF'] / denom
                self.X['Parasympathetic_Index'] = self.X['HF'] / denom
            
            if 'HF' in self.X.columns and 'TP' in self.X.columns:
                self.X['HF_TP_Ratio'] = self.X['HF'] / (self.X['TP'] + 1e-6)
                self.X['HF_TP_Ratio'].replace([np.inf, -np.inf], np.nan, inplace=True)

            if 'SDNN' in self.X.columns and 'MEANH' in self.X.columns:
                self.X['SDNN_MEANH_Ratio'] = self.X['SDNN'] / (self.X['MEANH'] + 1e-6)
                self.X['SDNN_MEANH_Ratio'].replace([np.inf, -np.inf], np.nan, inplace=True)

            if 'Sympathetic_Index' in self.X.columns and 'LFHF' in self.X.columns:
                lf_hf_clip = self.X['LFHF'].copy()
                lf_hf_clip[lf_hf_clip < 0] = 0
                self.X['GAD_Risk'] = self.X['Sympathetic_Index'] * np.log1p(lf_hf_clip)

        for label in self.label_names:
            if label in self.df.columns:
                self.y_dict[label] = self.df[label].copy()

        print(f"✓ 特徵數量: {self.X.shape[1]}")
        return len(self.y_dict) > 0

    def _numeric_feature_list_for_outlier(self, X_frame: pd.DataFrame):
        candidates = []
        # 包含 HRV, Psych, Age, BMI
        check_list = self.hrv_features + self.psych_features + ['Age', 'BMI']
        for col in check_list:
            if col in X_frame.columns:
                candidates.append(col)
        # 包含衍生特徵
        for col in self.log_engineered_cols:
             if col in X_frame.columns:
                candidates.append(col)
        
        out = []
        for c in candidates:
            s = pd.to_numeric(X_frame[c], errors='coerce')
            if s.notnull().any():
                out.append(c)
        return out

    def _compute_iqr_bounds(self, s: pd.Series, k: float):
        q1 = s.quantile(0.25); q3 = s.quantile(0.75)
        iqr = q3 - q1
        if pd.isna(iqr) or iqr == 0:
            lower = s.quantile(0.001); upper = s.quantile(0.999)
        else:
            lower = q1 - k * iqr; upper = q3 + k * iqr
        return float(lower), float(upper)

    def _fit_outlier_bounds(self, X_train: pd.DataFrame):
        num_cols = self._numeric_feature_list_for_outlier(X_train)
        bounds = {}
        for col in num_cols:
            s = pd.to_numeric(X_train[col], errors='coerce')
            lower, upper = self._compute_iqr_bounds(s.dropna(), self.iqr_multiplier)
            bounds[col] = (lower, upper)
        self.outlier_bounds_ = bounds

    def _apply_outlier_to_nan(self, X_frame: pd.DataFrame, outlier_bounds=None, stage_note=""):
        bounds = outlier_bounds if outlier_bounds else self.outlier_bounds_
        if not bounds:
            return X_frame

        Xp = X_frame.copy()
        total_flagged = 0

        # HRV 0 -> NaN
        if self.treat_zero_as_missing_in_hrv:
            for col in [c for c in self.hrv_features if c in Xp.columns]:
                s = pd.to_numeric(Xp[col], errors='coerce')
                zero_mask = (s == 0)
                total_flagged += int(zero_mask.sum())
                Xp.loc[zero_mask, col] = np.nan

        for col, (lb, ub) in bounds.items():
            if col not in Xp.columns:
                continue
            s = pd.to_numeric(Xp[col], errors='coerce')
            mask = (s < lb) | (s > ub)
            total_flagged += int(mask.sum())
            Xp.loc[mask, col] = np.nan
        
        if stage_note:
            print(f"   • [{stage_note}] 離群值→NaN：{total_flagged} 個")
        return Xp

    def _apply_log1p(self, X_frame: pd.DataFrame):
        Xp = X_frame.copy()
        # 只有定義在 log_hrv_cols 和 log_engineered_cols 的才做轉換 (Psych 不做)
        target_cols = self.log_hrv_cols + self.log_engineered_cols
        for col in target_cols:
            if col not in Xp.columns:
                continue
            s = pd.to_numeric(Xp[col], errors='coerce')
            neg_mask = s < 0
            if neg_mask.any():
                Xp.loc[neg_mask, col] = np.nan
            Xp[col] = np.log1p(Xp[col])
        return Xp

    def impute_and_scale(self, X_train, X_test=None, fit=True):
        X_train_p = X_train.copy()
        X_test_p = X_test.copy() if X_test is not None else None

        if fit:
            self._fit_outlier_bounds(X_train_p)
        
        X_train_p = self._apply_outlier_to_nan(X_train_p, stage_note="Train")
        if X_test_p is not None:
            X_test_p = self._apply_outlier_to_nan(X_test_p, stage_note="Test")

        X_train_p = self._apply_log1p(X_train_p)
        if X_test_p is not None:
            X_test_p = self._apply_log1p(X_test_p)

        knn_f = self._numeric_feature_list_for_outlier(X_train_p)
        if len(knn_f) > 0 and X_train_p[knn_f].isnull().any().any():
            if fit or (self.knn_imputer is None):
                self.knn_imputer = KNNImputer(n_neighbors=5, weights='distance')
                X_train_p[knn_f] = self.knn_imputer.fit_transform(X_train_p[knn_f])
            else:
                X_train_p[knn_f] = self.knn_imputer.transform(X_train_p[knn_f])
            if X_test_p is not None:
                X_test_p[knn_f] = self.knn_imputer.transform(X_test_p[knn_f])

        # 剩餘 NaN 用中位數
        if X_train_p.isnull().any().any():
            X_train_p.fillna(X_train_p.median(numeric_only=True), inplace=True)
            if X_test_p is not None:
                X_test_p.fillna(X_train_p.median(numeric_only=True), inplace=True)

        cols = X_train_p.columns.tolist()
        num_cols = [c for c in cols if c != 'Sex' and pd.api.types.is_numeric_dtype(X_train_p[c])]
        other_cols = [c for c in cols if c not in num_cols]

        if fit or (self.scaler is None):
            self.scaler = StandardScaler()
            X_train_num = pd.DataFrame(
                self.scaler.fit_transform(X_train_p[num_cols]),
                columns=num_cols, index=X_train_p.index
            )
        else:
            X_train_num = pd.DataFrame(
                self.scaler.transform(X_train_p[num_cols]),
                columns=num_cols, index=X_train_p.index
            )

        X_train_s = pd.concat([X_train_num, X_train_p[other_cols]], axis=1)[cols]

        if X_test_p is not None:
            X_test_num = pd.DataFrame(
                self.scaler.transform(X_test_p[num_cols]),
                columns=num_cols, index=X_test_p.index
            )
            X_test_s = pd.concat([X_test_num, X_test_p[other_cols]], axis=1)[cols]
            return X_train_s, X_test_s

        return X_train_s

    def apply_external_transform(self, X_raw, feature_columns, outlier_bounds, imputer, scaler):
        """外部驗證專用：使用訓練好的物件進行轉換"""
        # 1. 建立特徵 (基於目前的 mode)
        # 注意：外部驗證時，要先確保 X_raw 是原始資料，然後補上工程特徵
        Xp = X_raw.copy()
        
        # 重建工程特徵 (必須與 prepare_features_and_labels 邏輯一致)
        # 這裡為了簡化，假設外部呼叫前已經做過 basic build_raw_features，
        # 或者我們可以在這裡重做一次邏輯，但為了安全起見，建議外部驗證流程先呼叫 prepare_features_and_labels 
        # 但因為 external script 的邏輯是 load raw -> process，所以這裡我們只做 transform
        
        # 對齊欄位
        for col in feature_columns:
            if col not in Xp.columns:
                Xp[col] = np.nan
        Xp = Xp[feature_columns].copy()

        # 2. 離群值處理
        Xp = self._apply_outlier_to_nan(Xp, outlier_bounds=outlier_bounds)

        # 3. Log1p
        Xp = self._apply_log1p(Xp)

        # 4. Impute
        knn_f = self._numeric_feature_list_for_outlier(Xp)
        if imputer is not None and len(knn_f) > 0:
            try:
                Xp[knn_f] = imputer.transform(Xp[knn_f])
            except:
                pass # Fallback to median
        
        if Xp.isnull().any().any():
            Xp.fillna(Xp.median(numeric_only=True), inplace=True)

        # 5. Scale
        cols = Xp.columns.tolist()
        num_cols = [c for c in cols if c != 'Sex' and pd.api.types.is_numeric_dtype(Xp[c])]
        other_cols = [c for c in cols if c not in num_cols]

        if scaler is not None and len(num_cols) > 0:
            X_num = pd.DataFrame(scaler.transform(Xp[num_cols]), columns=num_cols, index=Xp.index)
        else:
            X_num = Xp[num_cols].copy()
            
        X_scaled = pd.concat([X_num, Xp[other_cols]], axis=1)[cols]
        return X_scaled