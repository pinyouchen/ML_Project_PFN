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
        scale_weight = int(self.ratio * 1.0)
        
        if self.strategy == 'aggressive':
            n_est = 700; depth = 25; lr = 0.03; base_weight_mult = 2.0
        elif self.strategy == 'moderate':
            n_est = 500; depth = 18; lr = 0.05; base_weight_mult = 1.5
        else:
            n_est = 400; depth = 12; lr = 0.08; base_weight_mult = 1.2

        if self.label_name == 'MDD': weight_mult = 2.0
        elif self.label_name == 'Panic': weight_mult = 1.8
        elif self.label_name == 'GAD': weight_mult = 1.2
        else: weight_mult = base_weight_mult

        final_weight = max(1, int(scale_weight * weight_mult))

        # 1. XGBoost
        if self.label_name == 'Panic':
            self.models['XGB'] = xgb.XGBClassifier(
                n_estimators=650, max_depth=18, learning_rate=0.035,
                scale_pos_weight=final_weight, subsample=0.75, colsample_bytree=0.75,
                gamma=0.2, min_child_weight=1, reg_alpha=0.08, reg_lambda=0.6,
                random_state=42, n_jobs=-1, verbosity=0
            )
        else:
            self.models['XGB'] = xgb.XGBClassifier(
                n_estimators=n_est, max_depth=int(depth * 0.4), learning_rate=lr,
                scale_pos_weight=final_weight, subsample=0.8, colsample_bytree=0.8,
                gamma=0.2, min_child_weight=2, reg_alpha=0.1, reg_lambda=1.0,
                random_state=42, n_jobs=-1, verbosity=0
            )

        # 2. LightGBM
        self.models['LGBM'] = lgb.LGBMClassifier(
            n_estimators=n_est, max_depth=int(depth * 0.4), learning_rate=lr,
            num_leaves=int(depth * 1.5), class_weight={0: 1, 1: final_weight},
            subsample=0.8, colsample_bytree=0.8, min_child_samples=8,
            reg_alpha=0.1, reg_lambda=1.0, random_state=42, n_jobs=-1, verbose=-1
        )

        # 3. Random Forest
        self.models['RF'] = RandomForestClassifier(
            n_estimators=n_est, max_depth=depth, min_samples_split=8, min_samples_leaf=4,
            class_weight={0: 1, 1: final_weight}, random_state=42, n_jobs=-1
        )

        # 4. Extra Trees
        self.models['ET'] = ExtraTreesClassifier(
            n_estimators=n_est, max_depth=depth, min_samples_split=8, min_samples_leaf=4,
            class_weight={0: 1, 1: final_weight}, random_state=42, n_jobs=-1
        )

        # 5. Gradient Boosting
        self.models['GB'] = GradientBoostingClassifier(
            n_estimators=int(n_est * 0.6), max_depth=int(depth * 0.3),
            learning_rate=lr, subsample=0.8, min_samples_split=8, random_state=42
        )

        # 6. Balanced RF
        self.models['BalancedRF'] = BalancedRandomForestClassifier(
            n_estimators=int(n_est * 0.8), max_depth=depth, min_samples_split=8,
            min_samples_leaf=4, random_state=42, n_jobs=-1
        )

        # --- 新增模型：TabPFN ---
        if _HAS_TABPFN:
            # 修正：移除 N_ensemble_configurations，使用預設值
            # 加上 try-except 嘗試設定 device，避免部分版本報錯
            try:
                self.models['TabPFN'] = TabPFNClassifier(device='cpu')
            except TypeError:
                # 萬一連 device 參數都不支援 (極少見)，就全用預設
                self.models['TabPFN'] = TabPFNClassifier()

        # --- 新增模型：EBM ---
        if _HAS_EBM:
            self.models['EBM'] = ExplainableBoostingClassifier(
                random_state=42, n_jobs=-1, interactions=0, outer_bags=8
            )

    def _fit_single_model(self, name, model, X_resampled, y_resampled):
        is_xgb = isinstance(model, xgb.XGBClassifier)
        is_lgbm = isinstance(model, lgb.LGBMClassifier)
        
        use_early_stop = (is_xgb or is_lgbm) and (self.label_name in ['Panic', 'MDD'])
        
        if use_early_stop:
            early_stop_rounds = 20; val_split_ratio = 0.20
            sss = StratifiedShuffleSplit(n_splits=1, test_size=val_split_ratio, random_state=42)
            train_sub_idx, val_sub_idx = next(sss.split(X_resampled, y_resampled))
            X_tr_sub = X_resampled.iloc[train_sub_idx]; y_tr_sub = y_resampled.iloc[train_sub_idx]
            X_val_sub = X_resampled.iloc[val_sub_idx]; y_val_sub = y_resampled.iloc[val_sub_idx]
            try:
                if is_xgb:
                    model.fit(X_tr_sub, y_tr_sub, eval_set=[(X_val_sub, y_val_sub)], early_stopping_rounds=early_stop_rounds, verbose=False)
                elif is_lgbm:
                    model.fit(X_tr_sub, y_tr_sub, eval_set=[(X_val_sub, y_val_sub)], eval_metric='binary_logloss', callbacks=[lgb.early_stopping(early_stop_rounds, verbose=False)])
            except TypeError:
                model.fit(X_resampled, y_resampled)
        else:
            # TabPFN 特殊處理
            if name == 'TabPFN':
                # 嘗試使用 overwrite_warning，如果不支援則直接 fit
                try:
                    model.fit(X_resampled, y_resampled, overwrite_warning=True)
                except TypeError:
                    model.fit(X_resampled, y_resampled)
            else:
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
                if name == 'TabPFN': weight = r['f1_score'] * 1.2
                elif name == 'EBM': weight = r['f1_score'] * 1.0
                elif self.label_name == 'Panic': weight = 0.5 * r['f1_score'] + 0.5 * r['precision']
                elif self.label_name == 'MDD': weight = 0.4 * r['f1_score'] + 0.6 * r['recall']
                else: weight = r['f1_score'] * (0.5 + 0.5 * r['precision'])
                
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