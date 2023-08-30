# %%
import gradio as gr
from fastai.vision.all import*
import dill
from PIL import Image

# %%
with open('models/pet_breed_50.pkl','rb') as f:
    learn = dill.load(f)

# %%
def classify(image):
    a,_,_=learn.predict(image)
    return a
    # print(b)
    # print(c)

# %%
# pil_image = Image.open('test_images/download.jpg')

# %%
# pil_image.show()
# type(pil_image)

# %%
classify(pil_image)

# %%
gr.Interface(fn=classify,inputs=gr.inputs.Image(),outputs='text',title='Pet Breeds Classifier', description='Classifies 36 dog and cat breeds').launch(share=True)

# %%



