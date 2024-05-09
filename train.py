import argparse
from data_processing import load_data
from model_processing import build_model, train_model, test_model, save_model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", help="Data directory")
    parser.add_argument("--save_dir", default="", help="Save directory")
    parser.add_argument(
        "--arch", default="densenet121", help="Model arch, default: densenet121"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.001,
        help="Learning rate, default: 0.001",
    )
    parser.add_argument(
        "--hidden_units",
        type=int,
        default=512,
        help="Number of hidden units, default: 512",
    )
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--gpu", action="store_true", help="Using GPU")

    return parser.parse_args()


def main():
    args = parse_args()

    train_datasets, train_loader, valid_loader, test_loader = load_data(args.data_dir)

    model = build_model(args.arch, args.hidden_units)

    model, criterion = train_model(
        model, args.gpu, args.epochs, args.learning_rate, train_loader, valid_loader
    )
    test_model(model, test_loader, args.gpu, criterion)
    save_model(
        model,
        train_datasets,
        args.arch,
        args.epochs,
        args.hidden_units,
        args.learning_rate,
        args.save_dir,
    )


if __name__ == "__main__":
    main()
