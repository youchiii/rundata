import cv2
import numpy as np
import mediapipe as mp
import os

def process_video(input_path, pose_model, draw_spec_mark, draw_spec_mesh, align_point=None):
    """
    単一の動画を読み込み、骨格推定を行い、結果フレームのリストを返す関数。
    """
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"エラー: {input_path} を開けませんでした。")
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

        # 背景用の黒い画像を作成
        background = np.zeros((height, width, 3), np.uint8)

        # MediaPipeで処理するために画像をRGBに変換
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose_model.process(image_rgb)

        # 最初のフレームで基準となる右足首の座標を取得
        if frame_count == 0 and results.pose_landmarks:
            landmark = results.pose_landmarks.landmark[28] # 28は右足首
            initial_point = (landmark.x * width, landmark.y * height)

        # 骨格を描画
        if results.pose_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                background,
                results.pose_landmarks,
                mp.solutions.pose.POSE_CONNECTIONS,
                landmark_drawing_spec=draw_spec_mark,
                connection_drawing_spec=draw_spec_mesh
            )

        # 位置合わせのためのアフィン変換
        if align_point and initial_point:
            move_x = align_point[0] - initial_point[0]
            move_y = align_point[1] - initial_point[1]
            trans_mat = np.float32([[1, 0, move_x], [0, 1, move_y]])
            background = cv2.warpAffine(background, trans_mat, (width, height))
            
        processed_frames.append(background)
        frame_count += 1

    cap.release()
    return processed_frames, initial_point

def main():
    """
    メインの処理を実行する関数
    """
    # --- ファイルパスの設定 ---
    video1_path = 'videos/2025_9_25_sample_pro.mp4'
    video2_path = 'videos/2025_9_25_sample_user.mp4'
    output_dir = 'output'
    overlap_dir ='overlap'
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(overlap_dir, exist_ok=True)
    
    output1_path = os.path.join(output_dir, 'output_video_pro.mp4')
    output2_path = os.path.join(output_dir, 'output_video_user.mp4')
    overlap_path = os.path.join(overlap_dir, 'overlap_video.mp4')

    # --- MediaPipeの初期化 ---
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    
    # 描画スタイルの定義
    custom_mesh_spec = mp_drawing.DrawingSpec(thickness=2, color=(0, 255, 0)) # 緑の線
    custom_mark_spec = mp_drawing.DrawingSpec(thickness=3, circle_radius=3, color=(0, 0, 255)) # 赤い点
    default_landmark_style = mp_drawing_styles.get_default_pose_landmarks_style()
    default_connection_style = mp_drawing.DrawingSpec(thickness=2, color=(255, 255, 255))  # 白線で代用

    # --- 動画処理の開始 ---
    print("動画処理を開始します...")

    with mp_pose.Pose(static_image_mode=False, model_complexity=2, min_detection_confidence=0.1) as pose:
        
        # 動画1の処理
        print(f"1本目の動画を処理中: {video1_path}")
        frames1, point1 = process_video(video1_path, pose, custom_mark_spec, custom_mesh_spec)
        
        # 動画2の処理 (動画1の基準点に位置を合わせる)
        print(f"2本目の動画を処理中: {video2_path}")
        frames2, _ = process_video(video2_path, pose, default_landmark_style, default_connection_style, align_point=point1)
        
        if not frames1 or not frames2:
            print("動画処理中にエラーが発生したため、終了します。")
            return
            
    # --- 動画の重ね合わせ ---
    print("骨格動画を重ね合わせています...")
    blended_frames = []
    num_frames = min(len(frames1), len(frames2))
    
    for i in range(num_frames):
        alpha = 0.5
        beta = 1.0 - alpha
        blended_image = cv2.addWeighted(frames1[i], alpha, frames2[i], beta, 0.0)
        blended_frames.append(blended_image)

    # --- 動画ファイルの保存 ---
    cap_info = cv2.VideoCapture(video1_path)
    fps = 30
    width = int(cap_info.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap_info.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap_info.release()
    
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')

    print(f"処理済み動画1を保存中: {output1_path}")
    out1 = cv2.VideoWriter(output1_path, fourcc, fps, (width, height))
    for frame in frames1:
        out1.write(frame)
    out1.release()

    print(f"処理済み動画2を保存中: {output2_path}")
    out2 = cv2.VideoWriter(output2_path, fourcc, fps, (width, height))
    for frame in frames2:
        out2.write(frame)
    out2.release()

    print(f"重ね合わせ動画を保存中: {overlap_path}")
    out_blend = cv2.VideoWriter(overlap_path, fourcc, fps, (width, height))
    for frame in blended_frames:
        out_blend.write(frame)
    out_blend.release()
    
    print("すべての処理が完了しました。")

if __name__ == '__main__':
    main()
