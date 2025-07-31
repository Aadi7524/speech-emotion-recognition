from huggingface_hub import HfApi

api = HfApi()
api.upload_file(
    path_or_fileobj="C:/Users/usaad/Desktop/human_model.h5",
    path_in_repo="human_model.h5",
    repo_id="Aadi75240/Speech_Emotion_Recognition",
    repo_type="space"
)
