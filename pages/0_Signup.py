import streamlit as st
from auth.auth_utils import create_user

st.set_page_config(page_title="Signup", page_icon="📝")

st.header("ユーザー登録")

st.write("新しいアカウントを作成します。ユーザー名とパスワードを入力してください。")

with st.form("signup_form"):
    username = st.text_input("ユーザー名", key="username")
    password = st.text_input("パスワード", type="password", key="password")
    confirm_password = st.text_input("パスワード（確認）", type="password", key="confirm_password")
    
    submitted = st.form_submit_button("登録する")

if submitted:
    if not username or not password or not confirm_password:
        st.error("すべてのフィールドを入力してください。")
    elif password != confirm_password:
        st.error("パスワードが一致しません。")
    else:
        success = create_user(username, password)
        if success:
            st.success(f"ユーザー「{username}」の登録を申請しました。管理者の承認をお待ちください。")
            st.info("承認後、ログインページからログインできます。")
        else:
            st.error(f"ユーザー名「{username}」は既に使用されています。別のユーザー名を選択してください。")
