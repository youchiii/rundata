ローカルで実行する時に打つコマンドpython -m streamlit run rundata_app.py

WSLで仮想環境を有効にする方法
source venv/bin/activate
仕様ライブラリやpythonのバージョンは
requirements.txt
runtime.txtを参照してください。
uvicorn backend.app.main:app --reload
