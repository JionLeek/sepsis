
import argparse


def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file_path", default="./datas/train_sample_log.csv", type=str)
    parser.add_argument("--eval_file_path", default="./datas/eval_sample_log.csv", type=str)
    parser.add_argument("--predict_file_path", default="./release_model/deep_sepsis_v1_predict.csv", type=str)
    parser.add_argument("--model_path", default="./release_model/deep_sepsis_v1.pickle", type=str)
    parser.add_argument('--Epochs', type=int, default=1)
    parser.add_argument('--train_batch_size', type=int, default=512, help="train batch size")
    parser.add_argument('--eval_batch_size', type=int, default=512, help="eval batch size")

    parser.add_argument('--learning_rate', type=float, default=0.001, help="learning rate")
    parser.add_argument('--weight_decay', type=float, default=0.0001, help="weight_decay")

    args = parser.parse_args()
    return args
