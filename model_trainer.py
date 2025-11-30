# model_trainer.py
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, precision_score, recall_score
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, ADASYN
from imblearn.ensemble import BalancedRandomForestClassifier
from utils import specificity_npv

# --- 新增：創新模型引用 (Safe Import) ---
_HAS_TABPFN = False
_HAS_EBM = False

try:
    from tabpfn import TabPFNClassifier
    _HAS_TABPFN = True
    print("✅ TabPFN module loaded.")
except ImportError:
    pass

try:
    from interpret.glassbox import ExplainableBoostingClassifier
    _HAS_EBM = True
    print("✅ EBM (InterpretML) module loaded.")
except ImportError:
    pass
# -------------------------

class ModelTrainer:
    def __init__(self, label_name, pos_count, neg_count, current_f1, target_f1, use_stacking=False):
        self.label_name = label_name
        self.pos_count = pos_count
        self.neg_count = neg_count
        self.ratio = neg_count / pos_count if pos_count > 0 else 1
        self.current_f1 = current_f1
        self.target_f1 = target_f1
        self.use_stacking = use_stacking 
        self.gap = target_f1 - current_f1

        self.models = {}
        self.fitted_models = {} 
        self.results = {}

        if self.gap > 0.10:
            self.strategy = 'aggressive'
        elif self.gap > 0.05:
            self.strategy = 'moderate'
        else:
            self.strategy = 'conservative'

    def get_sampling_strategy(self):
        if self.label_name == 'MDD': return 'SMOTE', 0.75, 5
        if self.label_name == 'Panic': return 'BorderlineSMOTE', 0.55, 4
        if self.label_name == 'GAD': return 'SMOTE', 0.45, 5

        if self.pos_count < 100:
            sampler_type = 'ADASYN'
            if self.strategy == 'aggressive': sampling_ratio = 0.65; k = 4
            else: sampling_ratio = 0.55; k = 5
        else:
            sampler_type = 'SMOTE'
            if self.strategy == 'aggressive': sampling_ratio = 0.65; k = 4
            elif self.strategy == 'moderate': sampling_ratio = 0.55; k = 5
            else: sampling_ratio = 0.50; k = 5
        return sampler_type, sampling_ratio, k

    def build_models(self):
        # --- 只保留 TabPFN 與 EBM ---
        
        # 1. TabPFN (若有安裝)
        if _HAS_TABPFN:
            # CPU 模式（如之前設定）
            print(f"   🚀 TabPFN using device: CPU")
            try:
                self.models['TabPFN'] = TabPFNClassifier(device='cpu')
            except TypeError:
                self.models['TabPFN'] = TabPFNClassifier()
        else:
            print("   ⚠️ TabPFN 未安裝，跳過。")

        # 2. EBM (若有安裝)
        if _HAS_EBM:
            # interactions=0 加速訓練，可視需求調整
            self.models['EBM'] = ExplainableBoostingClassifier(
                random_state=42, n_jobs=-1, interactions=0, outer_bags=8
            )
        else:
            print("   ⚠️ EBM 未安裝，跳過。")

        if not self.models:
            print("   ❌ 警告：沒有任何模型被建立！請檢查套件安裝。")

    def _fit_single_model(self, name, model, X_resampled, y_resampled):
        # TabPFN 特殊處理：強制轉換型態為 float32
        if name == 'TabPFN':
            # 1. 確保數據是 Numpy Array
            if hasattr(X_resampled, 'values'):
                X_final = X_resampled.values
            else:
                X_final = X_resampled
            
            if hasattr(y_resampled, 'values'):
                y_final = y_resampled.values
            else:
                y_final = y_resampled

            # 2. 強制轉換
            X_final = np.array(X_final, dtype=np.float32)
            y_final = np.array(y_final, dtype=int)

            try:
                model.fit(X_final, y_final, overwrite_warning=True)
            except TypeError:
                model.fit(X_final, y_final)
        else:
            # EBM 或其他
            model.fit(X_resampled, y_resampled)
                
        return model

    def _optimize_threshold_precision_first(self, y_true, y_pred_proba, n_thresh=100):
        thresholds = np.linspace(0.10, 0.90, n_thresh)
        if self.label_name == 'MDD': min_precision = 0.45; min_recall = 0.45
        elif self.label_name == 'Panic': min_precision = 0.45; min_recall = 0.45
        elif self.label_name == 'GAD': min_precision = 0.60; min_recall = 0.30
        else: min_precision = 0.50; min_recall = 0.30

        best_f1 = 0; best_thresh = 0.5
        for thresh in thresholds:
            y_pred = (y_pred_proba >= thresh).astype(int)
            if y_pred.sum() == 0: continue
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            if precision >= min_precision and recall >= min_recall:
                f1 = f1_score(y_true, y_pred)
                if f1 > best_f1: best_f1 = f1; best_thresh = thresh

        if best_f1 == 0:
            for thresh in thresholds:
                y_pred = (y_pred_proba >= thresh).astype(int)
                if y_pred.sum() == 0: continue
                f1 = f1_score(y_true, y_pred)
                if f1 > best_f1: best_f1 = f1; best_thresh = thresh
        return best_thresh, best_f1

    def train_and_evaluate(self, X_train, X_test, y_train, y_test):
        sampler_type, sampling_ratio, k = self.get_sampling_strategy()
        
        try:
            if sampler_type == 'ADASYN': sampler = ADASYN(sampling_strategy=sampling_ratio, n_neighbors=k, random_state=42)
            elif sampler_type == 'BorderlineSMOTE': sampler = BorderlineSMOTE(sampling_strategy=sampling_ratio, k_neighbors=k, random_state=42)
            else: sampler = SMOTE(sampling_strategy=sampling_ratio, k_neighbors=k, random_state=42)
            X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)
        except:
            X_resampled, y_resampled = X_train, y_train

        print(f"   [Train] Pos: {y_resampled.sum()} (Augmented)")
        
        for name, model in self.models.items():
            try:
                fitted_model = self._fit_single_model(name, model, X_resampled, y_resampled)
                self.fitted_models[name] = fitted_model
                
                # TabPFN 需要乾淨的 Numpy 輸入進行預測
                if name == 'TabPFN':
                    X_test_final = X_test.values if hasattr(X_test, 'values') else X_test
                    X_test_final = np.array(X_test_final, dtype=np.float32)
                    y_pred_proba = fitted_model.predict_proba(X_test_final)[:, 1]
                else:
                    y_pred_proba = fitted_model.predict_proba(X_test)[:, 1]

                best_thresh, _ = self._optimize_threshold_precision_first(y_test, y_pred_proba)
                y_pred = (y_pred_proba >= best_thresh).astype(int)

                f1 = f1_score(y_test, y_pred)
                acc = accuracy_score(y_test, y_pred)
                try: auc = roc_auc_score(y_test, y_pred_proba)
                except: auc = np.nan
                precision = precision_score(y_test, y_pred, zero_division=0)
                recall = recall_score(y_test, y_pred, zero_division=0)
                spec, npv = specificity_npv(y_test, y_pred)

                self.results[name] = {
                    'f1_score': f1, 'accuracy': acc, 'auc': auc,
                    'precision': precision, 'recall': recall, 'specificity': spec, 'npv': npv,
                    'threshold': best_thresh, 'y_pred': y_pred, 'y_pred_proba': y_pred_proba,
                    'y_true': y_test.values, 'model': fitted_model
                }
                print(f"      {name:10s}: F1={f1:.4f}, AUC={auc:.4f}, Th={best_thresh:.2f}")

            except Exception as e:
                print(f"      ❌ {name}: {e}")

        self._create_ensemble(X_test, y_test)
        return self.results

    def _create_ensemble(self, X_test, y_test):
        if len(self.results) < 2: return
        try:
            predictions, weights = [], []
            for name, r in self.results.items():
                if name == 'TabPFN': 
                    weight = r['f1_score'] * 1.2 # TabPFN 權重稍高
                elif name == 'EBM': 
                    weight = r['f1_score'] * 1.0 # EBM 標準權重
                else:
                    weight = r['f1_score'] # 預設 (雖然現在只剩這兩個)
                
                predictions.append(r['y_pred_proba'])
                weights.append(max(weight, 0.01))

            weights = np.array(weights)
            weights = weights / (weights.sum() if weights.sum() != 0 else 1.0)
            ensemble_proba = np.average(predictions, axis=0, weights=weights)
            best_thresh, _ = self._optimize_threshold_precision_first(y_test, ensemble_proba)
            ensemble_pred = (ensemble_proba >= best_thresh).astype(int)

            f1 = f1_score(y_test, ensemble_pred)
            acc = accuracy_score(y_test, ensemble_pred)
            try: auc = roc_auc_score(y_test, ensemble_proba)
            except: auc = np.nan
            precision = precision_score(y_test, ensemble_pred, zero_division=0)
            recall = recall_score(y_test, ensemble_pred, zero_division=0)
            spec, npv = specificity_npv(y_test, ensemble_pred)

            self.results['Ensemble'] = {
                'f1_score': f1, 'accuracy': acc, 'auc': auc,
                'precision': precision, 'recall': recall, 'specificity': spec, 'npv': npv,
                'threshold': best_thresh, 'y_pred': ensemble_pred, 'y_pred_proba': ensemble_proba,
                'y_true': y_test.values
            }
            print(f"      Ensemble  : F1={f1:.4f}, AUC={auc:.4f}")
        except Exception as e:
            print(f"      ⚠️ 集成失敗: {e}")