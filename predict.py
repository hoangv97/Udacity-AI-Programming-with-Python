import argparse
import json
from model_processing import load_model, predict


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path", help="Image path")
    parser.add_argument("checkpoint", help="Network checkpoint")
    parser.add_argument(
        "--topk", default=5, type=int, help="Return top K most likely classes"
    )
    parser.add_argument(
        "--category_names",
        default="cat_to_name.json",
        help="Use a mapping of categories to real names",
    )
    parser.add_argument("--gpu", action="store_true", help="Using GPU")

    return parser.parse_args()


def main():
    args = parse_args()

    model = load_model(args.checkpoint)
    probs, classes = predict(model, args.image_path, args.gpu, args.topk)

    with open(args.category_names, "r") as f:
        cat_to_name = json.load(f)

    result_classes = []

    for predict_class in classes:
        result_classes.append(cat_to_name[predict_class])

    print(f"Predict image {args.image_path}")
    print(probs)
    print(result_classes)


if __name__ == "__main__":
    main()
