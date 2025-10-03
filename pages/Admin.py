import streamlit as st
from auth.auth_utils import get_pending_users, update_user_status

st.set_page_config(page_title="Admin Console", page_icon="👑")

st.header("管理者コンソール")

# --- Authentication Check ---
if 'user' not in st.session_state or st.session_state['user']['role'] != 'teacher':
    st.error("このページにアクセスする権限がありません。")
    st.info("先生アカウントでログインしてください。")
    st.stop()

# --- Page Content ---
st.write(f"ようこそ、`{st.session_state['user']['username']}`さん。")
st.subheader("保留中のユーザーアカウント")

pending_users = get_pending_users()

if not pending_users:
    st.info("現在、承認待ちのユーザーはいません。")
else:
    for user in pending_users:
        col1, col2, col3, col4 = st.columns([2, 2, 1, 1])
        with col1:
            st.write(f"**ユーザー名:** {user['username']}")
        with col2:
            st.write(f"**登録日:** {user['created_at']}")
        with col3:
            # Use a unique key for each button
            if st.button("承認", key=f"approve_{user['id']}"):
                update_user_status(user['id'], 'active')
                st.success(f"`{user['username']}`を承認しました。")
                st.rerun()
        with col4:
            if st.button("却下", key=f"reject_{user['id']}", type="secondary"):
                update_user_status(user['id'], 'rejected')
                st.warning(f"`{user['username']}`を却下しました。")
                st.rerun()
        st.divider()
