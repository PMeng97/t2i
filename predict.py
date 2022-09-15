import uuid
from diffusers import DiffusionPipeline
import torch


def txt2img(prompt):
    # RUN THE TWO COMMANDS BELOW FIRST TO CACHE
    # git lfs install
    # git clone https://huggingface.co/CompVis/stable-diffusion-v1-4
    prompt = prompt.replace('+', ' ')
    print(prompt)

    model_id = "CompVis/ldm-text2im-large-256"

    print("MODEL LOADING")
    # load model and scheduler
    ldm = DiffusionPipeline.from_pretrained(model_id)
    print("START PREDICTING")
    # run pipeline in inference (sample random noise and denoise)
    prompt = "A painting of a squirrel eating a burger"
    images = ldm([prompt], num_inference_steps=50, eta=0.3, guidance_scale=6)["sample"]
    image = images[0]


    # # save images
    # for idx, image in enumerate(images):
    #     image.save(f"squirrel-{idx}.png")

    # if torch.cuda.is_available():
    #     print("@@Predict: Starting generation with gpu")
    #     pipe = pipe.to("cuda")
    #     pipe.enable_attention_slicing()
    #     with torch.autocast("cuda"):
    #         image = pipe(prompt).images[0]
    # else:
    #     print("@@Predict: Starting generation with cpu")
    #     pipe = pipe.to("cpu")
    #     pipe.enable_attention_slicing()
    #     image = pipe(prompt).images[0]
    # print('@@Predict: End generation')

    data_id = str(uuid.uuid4())
    img_name = data_id+'_'+('_').join(prompt.split())
    # Modification needed for MongoDB
    image.save(img_name+".png")
    return image

