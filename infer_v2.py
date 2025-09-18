# echomimic_v2/infer_v2.py
import argparse
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from diffusers import AutoencoderKL, DDIMScheduler
from omegaconf import OmegaConf
from PIL import Image

from src.models.unet_2d_condition import UNet2DConditionModel
from src.models.unet_3d_emo import EMOUNet3DConditionModel
from src.models.whisper.audio2feature import load_audio_model
from src.pipelines.pipeline_echomimicv2 import EchoMimicV2Pipeline as BasePipe
from src.pipelines.pipeline_echomimicv2_acc_v2 import EchoMimicV2Pipeline as AccPipe
from src.utils.util import save_videos_grid
from src.models.pose_encoder import PoseEncoder
from src.utils.dwpose_util import draw_pose_select_v2
from experiments.a2p.model_v2 import Audio2Pose
from src.models.dwpose.dwpose_detector import dwpose_detector

from moviepy.editor import VideoFileClip, AudioFileClip

ffmpeg_path = os.getenv('FFMPEG_PATH')
if ffmpeg_path is None:
    print(
        "please download ffmpeg-static and export to FFMPEG_PATH. \nFor example: export FFMPEG_PATH=./ffmpeg-4.4-amd64-static")
elif ffmpeg_path not in os.getenv('PATH'):
    print("add ffmpeg to path")
    os.environ["PATH"] = f"{ffmpeg_path}:{os.environ['PATH']}"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/prompts/infer.yaml")
    parser.add_argument("-W", type=int, default=768)
    parser.add_argument("-H", type=int, default=768)
    parser.add_argument("-L", type=int, default=240)
    parser.add_argument("--seed", type=int, default=3407)

    parser.add_argument("--context_frames", type=int, default=12)
    parser.add_argument("--context_overlap", type=int, default=3)

    parser.add_argument("--cfg", type=float, default=2.5)
    parser.add_argument("--steps", type=int, default=30)
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


def load_config(config_path: str):
    """Load YAML config into OmegaConf object."""
    return OmegaConf.load(config_path)


def get_weight_dtype(config):
    """
    Decide weight precision from config.
    Supports fp16, bf16, or defaults to fp32.
    """
    # TODO: try bf16
    if config.weight_dtype == "fp16":
        return torch.float16
    elif config.weight_dtype == "bf16":
        return torch.bfloat16
    else:
        return torch.float32


def select_device(device_str: str) -> str:
    """Return 'cuda' or 'cpu' depending on availability."""
    if "cuda" in device_str and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def build_save_dir(config, seed: int) -> Path:
    """Build save directory path based on motion module and seed."""
    module = config.motion_module_path.split("/")[-2]
    iteration = config.motion_module_path.split("/")[-1].split("-")[-1][:-4]
    save_dir = Path(f"outputs/{module}-iter{iteration}-seed{seed}/")
    save_dir.mkdir(exist_ok=True, parents=True)
    print(f"[INFO] Save dir: {save_dir}")
    return save_dir


def setup_environment(args):
    """Full setup: configs, dtype, device, save dir."""
    config = load_config(args.config)
    infer_config = load_config(config.inference_config)
    weight_dtype = get_weight_dtype(config)
    device = select_device(args.device)
    save_dir = build_save_dir(config, args.seed)
    return config, infer_config, weight_dtype, device, save_dir


############# model_init #############
def init_vae(config, device, dtype):
    """Load pretrained VAE (latent image encoder/decoder)."""
    return AutoencoderKL.from_pretrained(config.pretrained_vae_path).to(device, dtype=dtype)


def init_reference_unet(config, device, dtype):
    """Load reference UNet for appearance consistency."""
    model = UNet2DConditionModel.from_pretrained(
        config.pretrained_base_model_path,
        subfolder="unet",
    ).to(device, dtype=dtype)
    model.load_state_dict(torch.load(config.reference_unet_path, map_location="cpu"))
    return model


def init_denoising_unet(config, infer_config, device, dtype):
    """Load 3D denoising UNet with motion module."""
    if os.path.exists(config.motion_module_path):
        # stage1 + stage2
        model = EMOUNet3DConditionModel.from_pretrained_2d(
            config.pretrained_base_model_path,
            config.motion_module_path,
            subfolder="unet",
            unet_additional_kwargs=infer_config.unet_additional_kwargs,
        ).to(device, dtype=dtype)
    else:
        ### only stage1
        model = EMOUNet3DConditionModel.from_pretrained_2d(
            config.pretrained_base_model_path,
            "",
            subfolder="unet",
            unet_additional_kwargs={
                "use_motion_module": False,
                "unet_use_temporal_attention": False,
                "cross_attention_dim": infer_config.unet_additional_kwargs.cross_attention_dim
            }
        ).to(device, dtype=dtype)

    model.load_state_dict(
        torch.load(config.denoising_unet_path, map_location="cpu"),
        strict=False
    )
    return model


def init_pose_encoder(config, device, dtype):
    """Load pose encoder network."""
    model = PoseEncoder(
        320, conditioning_channels=3, block_out_channels=(16, 32, 96, 256)
    ).to(device, dtype=dtype)
    model.load_state_dict(torch.load(config.pose_encoder_path))
    return model


def init_audio_processor(config, device):
    """Load pretrained audio feature extractor."""
    return load_audio_model(model_path=config.audio_model_path, device=device)


def init_a2p_model(ckpt_path, device):
    model = Audio2Pose.load_from_checkpoint(ckpt_path, map_location=device).to(device).eval()
    return model


def init_models(config, infer_config, dtype, device):
    a2p_model = init_a2p_model(config.a2p_ckpt_path, device)
    vae = init_vae(config, device, dtype)
    reference_unet = init_reference_unet(config, device, dtype)
    denoising_unet = init_denoising_unet(config, infer_config, device, dtype)
    pose_net = init_pose_encoder(config, device, dtype)
    audio_processor = init_audio_processor(config, device)
    return vae, reference_unet, denoising_unet, pose_net, audio_processor, a2p_model


######################################
def build_scheduler(infer_config):
    """Create the diffusion noise scheduler from config."""
    sched_kwargs = OmegaConf.to_container(infer_config.noise_scheduler_kwargs)
    return DDIMScheduler(**sched_kwargs)


def build_pipeline(vae, reference_unet, denoising_unet, audio_processor, pose_net,
                   scheduler, device, dtype, a2p_model, variant: str = "base"):
    """Assemble EchoMimicV2 pipeline and move it to the right device/dtype."""
    Pipeline = AccPipe if variant == "acc" else BasePipe
    pipe = Pipeline(
        vae=vae,
        reference_unet=reference_unet,
        denoising_unet=denoising_unet,
        audio_guider=audio_processor,
        pose_encoder=pose_net,
        scheduler=scheduler,
        a2p_model=a2p_model
    )
    return pipe.to(device, dtype=dtype)


######################################
def init_seed(seed: int):
    """Return a torch.Generator seeded deterministically or randomly."""
    if seed is not None and seed > -1:
        return torch.manual_seed(seed)
    return torch.manual_seed(random.randint(100, 1000000))


def build_inputs(args):
    """Collect input file paths for reference image, audio, and pose."""
    inputs = {
        "refimg": f"{args.ref_images_dir}/{args.refimg_name}",
        "audio": f"{args.audio_dir}/{args.audio_name}",
        "pose": f"{args.pose_dir}/{args.pose_name}",
    }
    print("Pose:", inputs["pose"])
    print("Reference:", inputs["refimg"])
    print("Audio:", inputs["audio"])
    return inputs


def build_save_paths(save_dir: Path, refimg_name: str, pose_name: str, audio_name: str, start_idx: int = 0):
    """Create output directory and return save path prefix."""
    ref_flag = ".".join([refimg_name.split("/")[-2], refimg_name.split("/")[-1]])
    save_path = Path(f"{save_dir}/{ref_flag}/{pose_name}")
    save_path.mkdir(exist_ok=True, parents=True)

    ref_s = refimg_name.split("/")[-1].split(".")[0]
    save_name = f"{save_path}/{ref_s}-a-{audio_name}-i{start_idx}"
    return save_path, save_name


def load_inputs(inputs_dict, width, height):
    """Load reference image and audio file."""
    ref_image_pil = Image.open(inputs_dict["refimg"]).resize((width, height))
    audio_clip = AudioFileClip(inputs_dict["audio"])
    return ref_image_pil, audio_clip


def adjust_length(args, audio_clip, fps: int, pose_dir: str):
    """Ensure sequence length is valid given audio duration and pose files."""
    max_frames = int(audio_clip.duration * fps)
    num_pose_files = len(os.listdir(pose_dir))
    args.L = min(args.L, max_frames, num_pose_files)
    return args


######################################
def load_pose_frame(pose_dir: str, index: int, width: int, height: int):
    """Load and draw a single pose frame from npy file."""
    tgt_mask = np.zeros((width, height, 3), dtype="uint8")
    pose_path = os.path.join(pose_dir, f"{index}.npy")
    detected_pose = np.load(pose_path, allow_pickle=True).tolist()

    imh_new, imw_new, rb, re, cb, ce = detected_pose["draw_pose_params"]
    im = draw_pose_select_v2(detected_pose, imh_new, imw_new, ref_w=800)
    im = np.transpose(np.array(im), (1, 2, 0))
    tgt_mask[rb:re, cb:ce, :] = im
    return Image.fromarray(tgt_mask).convert("RGB")


def pose_to_tensor(pose_img: Image.Image, device, dtype):
    """Convert pose PIL image to normalized torch tensor [C,H,W]."""
    arr = np.array(pose_img)
    tensor = torch.Tensor(arr).to(device=device, dtype=dtype).permute(2, 0, 1) / 255.0
    return tensor


def build_pose_tensor(pose_dir: str, start_idx: int, length: int, width: int, height: int, device, dtype):
    """Build full pose tensor [1, C, T, H, W] for a sequence."""
    pose_list = []
    for idx in range(start_idx, start_idx + length):
        pose_img = load_pose_frame(pose_dir, idx, width, height)
        pose_tensor = pose_to_tensor(pose_img, device, dtype)
        pose_list.append(pose_tensor)
    return torch.stack(pose_list, dim=1).unsqueeze(0)


######################################
def run_inference(pipe, ref_image, audio_path, poses_tensor, args, generator, start_idx):
    """Run the EchoMimicV2 pipeline to generate a video tensor."""
    return pipe(
        ref_image,
        audio_path,
        poses_tensor[:, :, :args.L, ...],
        args.W,
        args.H,
        args.L,
        args.steps,
        args.cfg,
        generator=generator,
        audio_sample_rate=args.sample_rate,
        context_frames=args.context_frames,
        fps=args.fps,
        context_overlap=args.context_overlap,
        start_idx=start_idx,
        detector=dwpose_detector
    ).videos


def trim_video(video, poses_tensor, length: int):
    """Clip generated video to min length across video, poses, and target length."""
    final_length = min(video.shape[2], poses_tensor.shape[2], length)
    return video[:, :, :final_length, :, :]


def save_silent_video(video_tensor, save_name: str, fps: int):
    """Save silent video grid (no audio) to temporary mp4."""
    tmp_path = save_name + "_woa_sig.mp4"
    save_videos_grid(video_tensor, tmp_path, n_rows=1, fps=fps)
    return tmp_path


def mux_audio_to_video(video_path: str, audio_path: str, save_name: str, length: int, fps: int):
    """Mux audio into video, save final mp4, and delete temp file."""
    video_clip = VideoFileClip(video_path)
    audio_clip = AudioFileClip(audio_path).set_duration(length / fps)
    video_clip = video_clip.set_audio(audio_clip)
    final_path = save_name + "_sig.mp4"
    video_clip.write_videofile(final_path, codec="libx264", audio_codec="aac", threads=2)
    os.remove(video_path)  # remove temporary silent video
    return final_path


######################################
def setup_all(args):
    """Setup configs, models, scheduler, pipeline, and generator."""
    config, infer_config, weight_dtype, device, save_dir = setup_environment(args)
    vae, reference_unet, denoising_unet, pose_net, audio_processor, a2p_model = init_models(
        config, infer_config, weight_dtype, device
    )
    scheduler = build_scheduler(infer_config)
    pipe = build_pipeline(
        vae, reference_unet, denoising_unet, audio_processor, pose_net,
        scheduler, device, weight_dtype, a2p_model
    )
    generator = init_seed(args.seed)
    return pipe, config, infer_config, weight_dtype, device, save_dir, generator


def prepare_inputs_and_paths(args, save_dir, start_idx=0):
    """Prepare inputs, save paths, reference image, and audio clip."""
    inputs_dict = build_inputs(args)
    save_path, save_name = build_save_paths(save_dir, args.refimg_name, args.pose_name, args.audio_name, start_idx)
    ref_image_pil, audio_clip = load_inputs(inputs_dict, args.W, args.H)
    args = adjust_length(args, audio_clip, args.fps, inputs_dict["pose"])
    return inputs_dict, save_path, save_name, ref_image_pil, audio_clip, args


def generate_video(pipe, ref_image_pil, inputs_dict, poses_tensor, args, generator, start_idx, save_name):
    """Run pipeline, trim video, save silent version, mux audio, return final path."""
    video = run_inference(pipe, ref_image_pil, inputs_dict["audio"], poses_tensor, args, generator, start_idx)
    video_sig = trim_video(video, poses_tensor, args.L)
    tmp_path = save_silent_video(video_sig, save_name, args.fps)
    final_path = mux_audio_to_video(tmp_path, inputs_dict["audio"], save_name, args.L, args.fps)
    return final_path


######################################

def main():
    args = parse_args()
    pipe, config, infer_config, weight_dtype, device, save_dir, generator = setup_all(args)
    inputs_dict, save_path, save_name, ref_image_pil, audio_clip, args = prepare_inputs_and_paths(args, save_dir)

    poses_tensor = build_pose_tensor(
        inputs_dict["pose"], start_idx=0, length=args.L,
        width=args.W, height=args.H, device=device, dtype=weight_dtype
    )
    t0 = time.time()
    final_path = generate_video(pipe, ref_image_pil, inputs_dict, poses_tensor, args, generator, start_idx=0,
                                save_name=save_name)
    t1 = time.time()
    print(f"[INFO] Inference time: {t1 - t0:.2f} seconds")
    print(final_path)


if __name__ == "__main__":
    main()
