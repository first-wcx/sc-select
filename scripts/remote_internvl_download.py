from huggingface_hub import snapshot_download

snapshot_download(
    "OpenGVLab/InternVL2_5-4B",
    resume_download=True,
    local_dir_use_symlinks=False,
)
