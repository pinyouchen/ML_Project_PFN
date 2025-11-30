Task 1: 訓練 Extended HRV (進階生理訊號)
對應原本的 test2_data2_binary_hrv_Multimorbidity.py

Bash

python main.py --task train --mode extended_hrv
Task 2: 訓練 Basic HRV (基礎生理訊號 + SHAP)
對應原本的 test2_data2_binary_Multimorbidity.py

Bash

python main.py --task train --mode basic_hrv
Task 3: 訓練 Psych (僅心理量表)
對應原本的 test2_data2_binary_psych_Multimorbidity.py

Bash

python main.py --task train --mode psych
Task 4: 訓練 All Features (全特徵融合)
對應原本的 test2_data2_binary_all_Multimorbidity.py

Bash

python main.py --task train --mode all
Task 5: 執行外部驗證
對應原本的 external_validate_A_Data1_Multimorbidity.py 注意：這裡的 Run_xxx 資料夾名稱必須替換成您實際訓練跑出來的資料夾名稱。

Bash

python main.py --task validate --model_dir "D:\ML_Project\runs\Run_basic_hrv_20251129_210509"

TabPFN 登入步驟

步驟 1：進入 Python 互動模式
在您的 Anaconda Prompt (終端機) 中，輸入：
python
(會看到 >>> 出現，代表進入了 Python 環境)

步驟 2：輸入登入程式碼
請複製以下兩行程式碼，在 >>> 後面貼上並按下 Enter:
from huggingface_hub import login
login()

步驟 3：貼上 Token
這時候程式會顯示： Enter your token (input will not be visible):

請去 Hugging Face 網頁複製您的 Write 或 Read Token (以 hf_ 開頭)。

回到黑色視窗，按一下滑鼠右鍵 (這就是貼上，雖然畫面完全不會有變化，游標也不會動)。
按下 Enter。

如果成功，它會顯示 Login successful 或 Token is valid。

步驟 4：離開 Python
輸入以下指令離開：
exit()"# ML_Project_PFN" 
