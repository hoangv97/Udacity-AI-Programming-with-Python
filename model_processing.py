import torch
from torch import nn
from torch import optim
from torchvision import models
from collections import OrderedDict
from data_processing import process_image


def get_device(gpu):
    if gpu:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")

    return device


def build_model(arch, hidden_units):
    print(f"Build model arch {arch}, hidden units: {hidden_units}")

    if arch == "vgg16":
        model = models.vgg16(pretrained=True)
        input_units = 25088
    elif arch == "vgg13":
        model = models.vgg13(pretrained=True)
        input_units = 25088
    elif arch == "densenet121":
        model = models.densenet121(pretrained=True)
        input_units = 1024

    # Turn off gradients for model
    for param in model.parameters():
        param.requires_grad = False

    # define the new classifier
    model.classifier = nn.Sequential(
        OrderedDict(
            [
                ("fc1", nn.Linear(input_units, hidden_units)),
                ("relu", nn.ReLU()),
                ("dropout", nn.Dropout(p=0.2)),
                ("fc2", nn.Linear(hidden_units, 102)),
                ("output", nn.LogSoftmax(dim=1)),
            ]
        )
    )

    return model


def train_model(model, gpu, epochs, learning_rate, train_loader, valid_loader):
    print(f"Start training: epochs {epochs}, learning rate: {learning_rate}")

    device = get_device(gpu)

    criterion = nn.NLLLoss()

    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

    model.to(device)

    steps = 0
    running_loss = 0
    print_every = 5

    for epoch in range(epochs):
        for images, labels in train_loader:
            steps += 1

            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            logps = model.forward(images)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                model.eval()
                test_loss = 0
                accuracy = 0

                with torch.no_grad():
                    for inputs, labels in valid_loader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)

                        test_loss += batch_loss.item()

                        # Calculate accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                print(
                    f"Epoch {epoch + 1}/{epochs}..."
                    f"Train loss: {running_loss/print_every:.3f}..."
                    f"Test loss: {test_loss/len(valid_loader):.3f}..."
                    f"Test accuracy: {accuracy/len(valid_loader):.3f}..."
                )

                running_loss = 0
                model.train()

    print("Finished training.")

    return model, criterion


def test_model(model, test_loader, gpu, criterion):
    print(f"Test model")

    device = get_device(gpu)

    model.to(device)

    accuracy = 0
    model.eval()

    with torch.no_grad():
        for inputs, labels in test_loader:

            inputs, labels = inputs.to(device), labels.to(device)
            logps = model.forward(inputs)

            # Calculate accuracy
            ps = torch.exp(logps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

    print(f"Test accuracy: {accuracy/len(test_loader):.3f}")

    model.train()


def save_model(
    model, train_datasets, arch, epochs, hidden_units, learning_rate, save_dir
):
    model.class_to_idx = train_datasets.class_to_idx

    checkpoint = {
        "arch": arch,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "hidden_units": hidden_units,
        "class_to_idx": model.class_to_idx,
        "model_state_dict": model.state_dict(),
    }

    checkpoint_path = save_dir + "checkpoint.pth"

    torch.save(checkpoint, checkpoint_path)

    print(f"Save model to {checkpoint_path}")


def load_model(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model = build_model(checkpoint["arch"], checkpoint["hidden_units"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.class_to_idx = checkpoint["class_to_idx"]

    print(f"Load model from {checkpoint_path}")

    return model


def predict(model, image_path, gpu, topk):
    device = get_device(gpu)

    model.to(device)
    model.eval()

    image = process_image(image_path)
    image = torch.from_numpy(image).type(torch.FloatTensor)
    image = image.unsqueeze(0)
    image = image.to(device)

    with torch.no_grad():
        output = model.forward(image)

    output_prob = torch.exp(output)

    probs, indeces = output_prob.topk(topk)
    probs = probs.to("cpu").numpy().tolist()[0]
    indeces = indeces.to("cpu").numpy().tolist()[0]

    mapping = {val: key for key, val in model.class_to_idx.items()}
    classes = [mapping[item] for item in indeces]

    return probs, classes
