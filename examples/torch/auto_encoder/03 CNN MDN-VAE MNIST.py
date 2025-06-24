if __name__ == '__main__':
    import os
    import numpy as np
    from scipy import stats

    import seaborn as sns
    import matplotlib.pyplot as plt
    from ipywidgets import interact, fixed

    import torch
    import torchvision
    import torchvision.transforms as transforms

    from mindcraft.torch.module import AutoEncoder
    from mindcraft.torch.module import Conv
    from mindcraft.torch.module import FeedForward
    from mindcraft.torch.module import ConvT


    def get_MNIST(batch_size, dataset_directory, dataloader_workers, source='MNIST'):
        # Prepare dataset for training
        train_transformation = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), ])

        data_source = getattr(torchvision.datasets, source)

        train_dataset = data_source(root=dataset_directory, train=True, download=True, transform=train_transformation)

        test_dataset = data_source(root=dataset_directory, train=False, download=True, transform=train_transformation)

        # Prepare Data Loaders for training and validation
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                                   pin_memory=True, num_workers=dataloader_workers)

        # Prepare Data Loaders for training and validation
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                                  pin_memory=True, num_workers=dataloader_workers)

        return train_dataset, test_dataset, train_loader, test_loader


    ENCODER_DROPOUT = 0.3
    DECODER_DROPOUT = 0.2
    USE_BATCHNORM = True

    BATCHSIZE = 32
    LEARNINGRATE = 0.005

    NUM_COMPONENTS = 3

    train_dataset, test_dataset, train_loader, test_loader = get_MNIST(BATCHSIZE, 'examples/dataset', 12, 'MNIST')

    device = 'cpu'
    if torch.cuda.is_available():
        print(f'GPU available')
        device = 'cuda'

    encoder = Conv(
        input_size=1,
        input_dim=2,
        kernel_size=(3, 3, 3, 3,),
        filters=(32, 64, 64, 64,),
        strides=(1, 2, 2, 2,),
        activation=('ReLU', 'ReLU', 'ReLU', 'ReLU',),
        flatten=True,
        dropout=ENCODER_DROPOUT,
        batch_norm=USE_BATCHNORM
    )

    encoder = encoder.to(device)

    mu = FeedForward(input_size=1024, output_size=2 * NUM_COMPONENTS).to(device)
    weights = FeedForward(input_size=1024, output_size=2 * NUM_COMPONENTS).to(device)
    log_var = FeedForward(input_size=1024, output_size=2 * NUM_COMPONENTS).to(device)

    decoder = ConvT(
        input_size=2,
        input_dim=2,
        filters=[64, 64, 32, 1],
        kernel_size=[4, 4, 4, 4],
        strides=[1, 3, 3, 2],
        activation=['ReLU', 'ReLU', 'ReLU', 'Sigmoid'],
        dropout=DECODER_DROPOUT,
        batch_norm=USE_BATCHNORM
    )

    decoder = decoder.to('cuda')

    model = AutoEncoder(
        encoder=encoder,
        mu=mu,
        weight=weights,
        log_var=log_var,
        num_components=NUM_COMPONENTS,
        decoder=decoder,
        beta=1e-3,
    )

    model = model.to(device)
    print(model.parameters_str)


    def show_latent(data_loader=test_loader):
        y_latent = []
        y_labels = []
        for batch_features, batch_labels in data_loader:
            batch_features = batch_features.to(device)
            y = model.encode(batch_features).detach().cpu().numpy()
            y_latent.extend(y)
            y_labels.extend(batch_labels.numpy())

        y_latent = np.asarray(y_latent)
        y_labels = np.asarray(y_labels)

        digit_map = [y_labels == i for i in range(10)]

        plt.figure(figsize=(10, 10))
        for i, where_i in enumerate(digit_map):
            selection = y_latent[where_i]
            plt.scatter(*selection[..., :2].T, label=f'#{i}', alpha=0.5)
        plt.legend()
        plt.show()

        return y_latent

    latent_space = show_latent()
    print(f'{len(latent_space)} datapoints')

    BATCHSIZE = 32
    LEARNINGRATE = 0.0005

    EPOCHS = 200
    WINDOW = 5
    STOPPING = 1e-3

    N_BATCHSIZE_UPDATES = 5

    # create an optimizer object
    optimizer = torch.optim.Adam(model.parameters(),  # link to VAE parameters
                                 lr=LEARNINGRATE,  # set learning rate
                                 weight_decay=1e-3)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, EPOCHS)

    loss_history = []
    kl_loss_history = []
    r_loss_history = []
    valid_history = []
    variations = []


    def early_stopping(loss_history, rate=1e-4, window=5):
        if len(loss_history) < window:
            return False

        if np.absolute(np.std(loss_history[-window:])) < rate:
            print(np.absolute(np.std(loss_history[-window:])))
            return True

        return False


    batchsize_updates = 0
    batchsize_update_rate = 1.125

    for epoch in range(EPOCHS):
        loss, r_loss, kl_loss = 0, 0, 0

        for batch in train_loader:
            # reshape mini-batch data to [N, 784] matrix
            # load it to the active device
            batch = [b.to(device) for b in batch]

            # reset the gradients back to zero
            # PyTorch accumulates gradients on subsequent backward passes
            optimizer.zero_grad()

            # TRAINING STEP: x -> (mu, std) -> (q -> z) -> x_hat
            elbo_loss, kl_loss, r_loss = model.batch_loss(batch, None)

            # compute accumulated gradients
            elbo_loss.backward()

            # perform parameter update based on current gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            optimizer.step()
            scheduler.step()

            # add the mini-batch training loss to epoch loss
            loss += elbo_loss.item()
            r_loss += r_loss.item()
            kl_loss += kl_loss.item()

        valid = 0
        for test_batch in test_loader:
            test_batch = [b.to(device) for b in test_batch]
            test_loss, *_ = model.batch_loss(test_batch, None)
            valid += test_loss.item()

        # compute the epoch training loss
        loss_history.append(loss / len(train_loader))
        r_loss_history.append(r_loss / len(train_loader))
        kl_loss_history.append(kl_loss / len(train_loader))
        valid_history.append(valid / len(test_loader))

        # display the epoch training loss
        print("epoch : {}/{}, loss = {:.6f} (LATENT: {:.6f}, RECON.: {:.6f}), valid = {:.6f}".format(epoch + 1,
                                                                                                     EPOCHS,
                                                                                                     loss_history[-1],
                                                                                                     kl_loss_history[-1],
                                                                                                     r_loss_history[-1],
                                                                                                     valid_history[-1],
                                                                                                     ))

        if early_stopping(loss_history, rate=STOPPING, window=WINDOW):
            break

        if epoch > 0 and epoch % 10 == 0:
            if batchsize_updates < N_BATCHSIZE_UPDATES:
                batchsize_updates += 1
                BATCHSIZE = int(batchsize_update_rate * BATCHSIZE)
                train_dataset, test_dataset, train_loader, test_loader = get_MNIST(
                    BATCHSIZE, 'examples/dataset/', 12, 'MNIST')
                print(f"update batchsize to {BATCHSIZE}")

            else:
                print(
                    f"early stopping: could not improve validation of model in range {STOPPING} in {WINDOW} consecutive epochs")
                break

