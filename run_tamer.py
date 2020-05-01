import argparse
import asyncio
import os
import dlib
from keras.models import load_model
import gym
from TAMER.linearTAMER import TAMERAgent, LOGS_DIR


async def main(args):
    env = gym.make("MountainCar-v0")

    # TODO: move to args?
    # hyperparameters
    discount_factor = 1
    epsilon = 0  # vanilla Q learning actually works well with no random exploration
    min_eps = 0
    num_episodes = 5
    tame = True  # set to false for vanilla Q learning
    loaded_face_classifier_model = load_model(args.face_classifier_model_path)  # load pre-trained face classifier model
    loaded_dlib_detector = dlib.get_frontal_face_detector() # load dlib detector
    loaded_dlib_predictor = dlib.shape_predictor(args.dlib_model_path) # load dlib predictor
    # df = pd.read_csv("FaceClassifier/data/master.csv")
    # classes = df.columns[1:].to_list()

    # labels.npy is a dictionary, keys are ["AU01","AU02"...], values are ["Upper Lip","Raised Eyebrow"...]
    classes = list(np.load('FaceClassifier/data/labels.npy',allow_pickle=True).item().values())
    agent = TAMERAgent(
        env,
        discount_factor,
        epsilon,
        min_eps,
        num_episodes,
        tame,
        args.tamer_training_timestep,
        AU_classes = classes,
        output_dir = args.output,
        face_classifier_model = loaded_face_classifier_model,
        dlib_detector = loaded_dlib_detector,
        dlib_predictor = loaded_dlib_predictor,
        use_cnn = args.use_cnn,
        model_file_to_load = None,  # pretrained model name here
    )

    # TODO: move capture_video to args

    await agent.train(model_file_to_save="autosave", capture_video=True)
    # agent.load_model(filename='2_episodes_0.2s')
    # agent.play(n_episodes=1, render=True)
    # agent.evaluate()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output", type=str, default=LOGS_DIR)
    # set a timestep for training TAMER
    # the more time per step, the easier for the human
    # but the longer it takes to train (in real time)
    # 0.2 seconds is fast but doable
    parser.add_argument("-t", "--tamer_training_timestep", default=0.2)
    parser.add_argument(
        "--face_classifier_model_path",
        default=None,
        type=str,
        help="Path to the trained model, choose either CNN or landmarks model"
    )
    parser.add_argument(
        "--dlib_model_path",
        default='FaceClassifier/weights/shape_predictor_68_face_landmarks.dat',
        type=str,
        help="Path to the dlib model that detects and predicts facial landmarks on an image/frame"
    )
    parser.add_argument(
        "--use_cnn",
        default=False,
        type=bool,
        help="Flag, set to True if using CNN model for facial expression classifier"
    )
    args = parser.parse_args()
    asyncio.run(main(args))
