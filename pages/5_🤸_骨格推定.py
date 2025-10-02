
import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import os
import tempfile

st.title("🤸 骨格推定")

st.info("2つの動画をアップロードして、骨格を比較します。")

use_sample_videos = st.checkbox("サンプル動画を使用する", value=True)

video1_path = None
video2_path = None

if use_sample_videos:
    st.write("サンプル動画を使って骨格推定を実行します。")
    video1_path = "motion_analysis/videos/2025_9_25_sample_pro.mp4"
    video2_path = "motion_analysis/videos/2025_9_25_sample_user.mp4"
    st.video(video1_path)
    st.video(video2_path)

else:
    uploaded_file1 = st.file_uploader("動画1をアップロード", type=["mp4", "mov"])
    uploaded_file2 = st.file_uploader("動画2をアップロード", type=["mp4", "mov"])

    if uploaded_file1 and uploaded_file2:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tfile1:
            tfile1.write(uploaded_file1.read())
            video1_path = tfile1.name
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tfile2:
            tfile2.write(uploaded_file2.read())
            video2_path = tfile2.name

def process_video(input_path, pose_model, draw_spec_mark, draw_spec_mesh, align_point=None):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        st.error(f"エラー: {input_path} を開けませんでした。")
        return None, None

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    processed_frames = []
    initial_point = None

    frame_count = 0
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break

        # The image is used as the background to draw on.
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose_model.process(image_rgb)

        if frame_count == 0 and results.pose_landmarks:
            landmark = results.pose_landmarks.landmark[28] # Right ankle
            initial_point = (landmark.x * width, landmark.y * height)

        if results.pose_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                image, # Draw on the original image
                results.pose_landmarks,
                mp.solutions.pose.POSE_CONNECTIONS,
                landmark_drawing_spec=draw_spec_mark,
                connection_drawing_spec=draw_spec_mesh
            )

        if align_point and initial_point:
            move_x = align_point[0] - initial_point[0]
            move_y = align_point[1] - initial_point[1]
            trans_mat = np.float32([[1, 0, move_x], [0, 1, move_y]])
            processed_image = cv2.warpAffine(image, trans_mat, (width, height))
        else:
            processed_image = image
            
        processed_frames.append(processed_image.copy())
        frame_count += 1

    cap.release()
    return processed_frames, initial_point

if st.button("骨格推定を実行"):
    if video1_path and video2_path:
        mp_pose = mp.solutions.pose
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles
        
        custom_mesh_spec = mp_drawing.DrawingSpec(thickness=2, color=(0, 255, 0))
        custom_mark_spec = mp_drawing.DrawingSpec(thickness=3, circle_radius=3, color=(0, 0, 255))
        custom_mark_spec2 = mp_drawing.DrawingSpec(thickness=3, circle_radius=3, color=(255, 0, 0)) # Blue
        default_connection_style = mp_drawing.DrawingSpec(thickness=2, color=(255, 255, 255))

        with st.spinner("骨格推定を実行中..."):
            with mp_pose.Pose(static_image_mode=False, model_complexity=2, min_detection_confidence=0.1) as pose:
                frames1, point1 = process_video(video1_path, pose, custom_mark_spec, custom_mesh_spec)
                frames2, _ = process_video(video2_path, pose, custom_mark_spec2, default_connection_style, align_point=point1)

            if frames1 and frames2:
                st.success("骨格推定が完了しました。")
                
                blended_frames = []
                num_frames = min(len(frames1), len(frames2))
                for i in range(num_frames):
                    alpha = 0.5
                    beta = 1.0 - alpha
                    blended_image = cv2.addWeighted(frames1[i], alpha, frames2[i], beta, 0.0)
                    blended_frames.append(blended_image)

                def create_video_bytes(frames, base_video_path):
                    output_video_path = None
                    try:
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tfile_out:
                            output_video_path = tfile_out.name
                            cap_info = cv2.VideoCapture(base_video_path)
                            fps = 30
                            width = int(cap_info.get(cv2.CAP_PROP_FRAME_WIDTH))
                            height = int(cap_info.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            cap_info.release()
                            
                            fourcc = cv2.VideoWriter_fourcc(*'avc1')
                            out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
                            for frame in frames:
                                out.write(frame)
                            out.release()

                            with open(output_video_path, "rb") as f:
                                return f.read()
                    finally:
                        if output_video_path and os.path.exists(output_video_path):
                            os.remove(output_video_path)
                
                video1_bytes = create_video_bytes(frames1, video1_path)
                video2_bytes = create_video_bytes(frames2, video2_path)
                blended_bytes = create_video_bytes(blended_frames, video1_path)

                st.session_state['video1_bytes'] = video1_bytes
                st.session_state['video2_bytes'] = video2_bytes
                st.session_state['blended_bytes'] = blended_bytes

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.header("動画1（骨格）")
                    st.video(video1_bytes)
                with col2:
                    st.header("動画2（骨格）")
                    st.video(video2_bytes)
                with col3:
                    st.header("重ね合わせ")
                    st.video(blended_bytes)

            else:
                st.error("骨格推定中にエラーが発生しました。")

        if not use_sample_videos:
            os.remove(video1_path)
            os.remove(video2_path)
    else:
        st.warning("動画が選択されていません。")

if 'blended_bytes' in st.session_state:
    st.header("動画の保存")
    output_dir = "motion_analysis/output"
    os.makedirs(output_dir, exist_ok=True)

    def sanitize_filename(filename):
        # Remove directory traversal characters and other potentially unsafe characters
        return os.path.basename(filename.replace("..", "").replace("/", "").replace("\\", ""))

    col1, col2, col3 = st.columns(3)
    with col1:
        video1_filename = st.text_input("動画1のファイル名", value="output_video_pro.mp4", key="fn1")
        if st.button("動画1の骨格を保存"):
            clean_filename = sanitize_filename(video1_filename)
            if not clean_filename.endswith(".mp4"):
                clean_filename += ".mp4"
            save_path = os.path.join(output_dir, clean_filename)
            with open(save_path, "wb") as f:
                f.write(st.session_state['video1_bytes'])
            st.success(f"動画を {save_path} に保存しました。")
    with col2:
        video2_filename = st.text_input("動画2のファイル名", value="output_video_user.mp4", key="fn2")
        if st.button("動画2の骨格を保存"):
            clean_filename = sanitize_filename(video2_filename)
            if not clean_filename.endswith(".mp4"):
                clean_filename += ".mp4"
            save_path = os.path.join(output_dir, clean_filename)
            with open(save_path, "wb") as f:
                f.write(st.session_state['video2_bytes'])
            st.success(f"動画を {save_path} に保存しました。")
    with col3:
        blended_filename = st.text_input("重ね合わせ動画のファイル名", value="overlap_video.mp4", key="fn3")
        if st.button("重ね合わせ動画を保存"):
            clean_filename = sanitize_filename(blended_filename)
            if not clean_filename.endswith(".mp4"):
                clean_filename += ".mp4"
            save_path = os.path.join(output_dir, clean_filename)
            with open(save_path, "wb") as f:
                f.write(st.session_state['blended_bytes'])
            st.success(f"動画を {save_path} に保存しました。")
