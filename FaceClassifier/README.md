## Facial Expressions Classifier 

To use a trained model to predict the class probabilities for an image use this command:

`python3 predict.py --model_path weights/some_model_name.h5 --image_path path_to_image.png --resize_dims 224 224`

- The `predict.py` does all the pre-processing for you and will print and return a dictionary mapping the class names to their probabilities from the sigmoid layer.

- Images are available at [this link](https://drive.google.com/open?id=1-3wFHSnP0VtUAVvUaOxV0nafMlp4AeH1)
- Use `python3 predict.py --help` to view the command line arguments in detail.
