import streamlit as st
from auth.auth_utils import get_all_users, reset_password

st.set_page_config(page_title="User List", page_icon="👥")

st.header("登録ユーザー一覧")

# --- Authentication Check ---
if 'user' not in st.session_state or st.session_state['user']['role'] != 'teacher':
    st.error("このページにアクセスする権限がありません。")
    st.info("先生アカウントでログインしてください。")
    st.stop()

# --- Page Content ---
all_users = get_all_users()

if not all_users:
    st.info("登録されているユーザーはいません。")
else:
    st.write(f"合計 {len(all_users)} 人のユーザーが登録されています。")
    st.divider()

    for user in all_users:
        st.subheader(f"ユーザー名: `{user['username']}`")
        col1, col2, col3 = st.columns(3)
        col1.metric("ID", user['id'])
        col2.metric("役割", user['role'])
        
        # Color code the status
        if user['status'] == 'active':
            col3.metric("ステータス", "✅ Active")
        elif user['status'] == 'pending':
            col3.metric("ステータス", "⏳ Pending")
        else:
            col3.metric("ステータス", "❌ Rejected")

        with st.expander("パスワードをリセットする"):
            with st.form(key=f"reset_form_{user['id']}"):
                new_password = st.text_input("新しいパスワード", type="password", key=f"new_pass_{user['id']}")
                confirm_new_password = st.text_input("新しいパスワード（確認）", type="password", key=f"confirm_pass_{user['id']}")
                
                submitted = st.form_submit_button("リセットを実行")

                if submitted:
                    if not new_password or not confirm_new_password:
                        st.error("新しいパスワードを両方のフィールドに入力してください。")
                    elif new_password != confirm_new_password:
                        st.error("パスワードが一致しません。")
                    else:
                        try:
                            reset_password(user['id'], new_password)
                            st.success(f"`{user['username']}` のパスワードが正常にリセットされました。")
                        except Exception as e:
                            st.error(f"パスワードのリセット中にエラーが発生しました: {e}")
        st.divider()
