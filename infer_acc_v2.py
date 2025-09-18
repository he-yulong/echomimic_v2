import argparse
import os
from datetime import datetime
from pathlib import Path
from PIL import Image
import time
from moviepy.editor import AudioFileClip
from infer_v2 import load_config, get_weight_dtype, select_device, init_models, build_scheduler, \
    init_seed, run_inference, save_silent_video, mux_audio_to_video, build_pipeline, build_pose_tensor

ffmpeg_path = os.getenv('FFMPEG_PATH')
if ffmpeg_path is None:
    print(
        "please download ffmpeg-static and export to FFMPEG_PATH. \nFor example: export FFMPEG_PATH=./ffmpeg-4.4-amd64-static")
elif ffmpeg_path not in os.getenv('PATH'):
    print("add ffmpeg to path")
    os.environ["PATH"] = f"{ffmpeg_path}:{os.environ['PATH']}"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/prompts/infer_acc_v2.yaml")
    parser.add_argument("-W", type=int, default=768)
    parser.add_argument("-H", type=int, default=768)
    parser.add_argument("-L", type=int, default=240)
    parser.add_argument("--seed", type=int, default=420)

    parser.add_argument("--context_frames", type=int, default=12)
    parser.add_argument("--context_overlap", type=int, default=3)

    parser.add_argument("--motion_sync", type=int, default=1)

    parser.add_argument("--cfg", type=float, default=1.0)
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument("--ref_images_dir", type=str, default=f'./assets/halfbody_demo/refimag')
    parser.add_argument("--audio_dir", type=str, default='./assets/halfbody_demo/audio')
    parser.add_argument("--pose_dir", type=str, default="./assets/halfbody_demo/pose")
    parser.add_argument("--refimg_name", type=str, default='natural_bk_openhand/0035.png')
    parser.add_argument("--audio_name", type=str, default='chinese/echomimicv2_woman.wav')
    parser.add_argument("--pose_name", type=str, default="01")

    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    config = load_config(args.config)
    infer_config = load_config(config.inference_config)
    weight_dtype = get_weight_dtype(config)
    device = select_device(args.device)

    vae, reference_unet, denoising_unet, pose_net, audio_processor, a2p_model = init_models(
        config, infer_config, weight_dtype, device
    )

    scheduler = build_scheduler(infer_config)

    pipe = build_pipeline(
        vae, reference_unet, denoising_unet, audio_processor, pose_net,
        scheduler, device, weight_dtype, variant="acc", a2p_model=a2p_model
    )

    date_str = datetime.now().strftime("%Y%m%d")
    time_str = datetime.now().strftime("%H%M")
    save_dir_name = f"{time_str}--step_{args.steps}-{args.W}x{args.H}--cfg_{args.cfg}"
    save_dir = Path(f"output/{date_str}/{save_dir_name}")
    save_dir.mkdir(exist_ok=True, parents=True)

    for ref_image_path in config["test_cases"].keys():
        for file_path in config["test_cases"][ref_image_path]:
            if ".wav" in file_path:
                audio_path = file_path
            else:
                pose_dir = file_path

        generator = init_seed(args.seed)

        ref_name = Path(ref_image_path).stem
        audio_name = Path(audio_path).stem
        final_fps = args.fps

        inputs_dict = {
            "refimg": f'{ref_image_path}',
            "audio": f'{audio_path}',
            "pose": f'{pose_dir}',
        }

        start_idx = 0

        print('Pose:', inputs_dict['pose'])
        print('Reference:', inputs_dict['refimg'])
        print('Audio:', inputs_dict['audio'])

        save_path = Path(f"{save_dir}/{ref_name}")
        save_path.mkdir(exist_ok=True, parents=True)
        save_name = f"{save_path}/{ref_name}-a-{audio_name}-i{start_idx}"

        ref_img_pil = Image.open(ref_image_path).convert("RGB")
        audio_clip = AudioFileClip(inputs_dict['audio'])

        args.L = min(args.L, int(audio_clip.duration * final_fps), len(os.listdir(inputs_dict['pose'])))
        # ==================== face_locator =====================
        # poses_tensor = build_pose_tensor(
        #     pose_dir, start_idx=0, length=args.L,
        #     width=args.W, height=args.H, device=device, dtype=weight_dtype
        # )
        poses_tensor = None
        # Run pipeline
        t0 = time.time()
        video = run_inference(pipe, ref_img_pil, inputs_dict["audio"], poses_tensor, args, generator, start_idx)
        final_length = min(video.shape[2], poses_tensor.shape[2], args.L)

        video_sig = video[:, :, :final_length, :, :]
        tmp_path = save_silent_video(video_sig, save_name, args.fps)
        final_path = mux_audio_to_video(tmp_path, inputs_dict["audio"], save_name, args.L, args.fps)
        t1 = time.time()
        print(f"[INFO] Inference time: {t1 - t0:.2f} seconds")
        print(f"[INFO] Saved: {final_path}")


if __name__ == "__main__":
    main()
