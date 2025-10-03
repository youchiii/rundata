import streamlit as st
from auth.auth_utils import get_user, verify_password

st.set_page_config(page_title="Login", page_icon="🔑")

st.header("ログイン")

if 'user' in st.session_state:
    st.success(f"`{st.session_state['user']['username']}`としてログイン済みです。")
    st.info("メインページに移動して、利用可能な機能を確認してください。")
else:
    with st.form("login_form"):
        username = st.text_input("ユーザー名", key="username")
        password = st.text_input("パスワード", type="password", key="password")
        
        submitted = st.form_submit_button("ログイン")

    if submitted:
        if not username or not password:
            st.error("ユーザー名とパスワードを入力してください。")
        else:
            user = get_user(username)
            
            if user and verify_password(password, user['password_hash']):
                if user['status'] == 'active':
                    # Store user info in session state and rerun to trigger router
                    st.session_state['user'] = {
                        'id': user['id'],
                        'username': user['username'],
                        'role': user['role']
                    }
                    st.rerun()
                elif user['status'] == 'pending':
                    st.warning("このアカウントは現在、管理者の承認待ちです。")
                elif user['status'] == 'rejected':
                    st.error("このアカウントは管理者に拒否されました。")
                else:
                    st.error("不明なアカウントステータスです。")
            else:
                st.error("ユーザー名またはパスワードが正しくありません。")
