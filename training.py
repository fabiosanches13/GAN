import networks
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import time

# --------------------------------------------------------------------------
# Data preparation
# --------------------------------------------------------------------------

def fetch_data(batch_size):

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    train_set = torchvision.datasets.MNIST(root='./data', train=True,
                                           download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_set,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=2)

    return train_loader


# --------------------------------------------------------------------------
# Objective functions
# --------------------------------------------------------------------------

def gen_objective(discriminator, generator, batch_size, verbose=False):

    generated = generator(seed(batch_size, generator.input_dimension))
    gen_belief = discriminator(generated)
    objective = ((1.0 - gen_belief).log()).mean()

    if verbose and torch.rand(1).item() < .1:
        print("Gen belief: ", gen_belief.mean().item())

    over_trained = False
    if gen_belief.mean().item() > .7:  # TODO: hard coded
        over_trained = True

    under_trained = False
    if (gen_belief.mean()).item() < .3:
        under_trained = True

    return objective, (over_trained, under_trained)


def dis_objective(discriminator, generator, data, batch_size, verbose=False):

    data_belief = discriminator(data)

    objective = -(data_belief.log()).mean()
    objective = objective - gen_objective(discriminator, generator, batch_size)[0]

    if verbose and torch.rand(1).item() < 1.1:
        print("Data belief: ", data_belief.mean().item())

    over_trained = False
    if data_belief.mean().item() > .6:  # TODO: hard coded
        over_trained = True

    return objective, over_trained


# --------------------------------------------------------------------------
# Main training function
# --------------------------------------------------------------------------

def trainer(num_epochs,
            gen_input_dimension=20,
            batch_size=5,
            dis_lr=.0001, dis_betas=(.5, .999),
            gen_lr=.0006, gen_betas=(.5, .999),
            generator=None,
            discriminator=None):

    if generator is None:
        generator = networks.DCGen(gen_input_dimension)
    if discriminator is None:
        discriminator = networks.DiscriminatorMBD(30,30)

    dis_optimizer = optim.Adam(discriminator.parameters(),
                               lr=dis_lr, betas=dis_betas)

    gen_optimizer = optim.Adam(generator.parameters(),
                               lr=gen_lr, betas=gen_betas)


    # gen_optimizer = optim.SGD([
    #     {'params': generator.planning_layers.parameters()},
    #     {'params': generator.transpose_conv_1.parameters()},
    #     {'params': generator.transpose_conv_2.parameters()},
    #     {'params': generator.sharpening_layer.parameters(), 'lr': gen_lr/70.}
    # ], lr=gen_lr, momentum=gen_momentum)
    #
    #
    # def constant(x):
    #     return x
    #
    # def delayed(x):
    #     if x < 10:
    #         return x / 10.
    #     else:
    #         return 1
    #
    # delayed = lambda x : min(x, 1)
    # lambda_list = [constant, constant, constant, delayed]
    # scheduler = lr_scheduler.LambdaLR(gen_optimizer,lambda_list)



    # fetch data
    train_loader = fetch_data(batch_size)

    # train
    for epoch in range(num_epochs):
        print('epoch:', epoch)
        #scheduler.step()

        gen_steps = dis_steps = 0

        train_generator = True

        data_iterator = iter(train_loader)
        count = 0
        for data in data_iterator:

            while train_generator:  # TODO: This condition is ad hoc and hard-coded.

                gen_steps += 1
                gen_optimizer.zero_grad()
                dis_optimizer.zero_grad()

                objective, flags = gen_objective(discriminator, generator, batch_size,verbose=True)

                objective.backward()
                gen_optimizer.step()

                over_trained = flags[0]
                under_trained = flags[1]

                stop_condition = gen_steps > 10 and not under_trained
                stop_condition = stop_condition or over_trained

                if stop_condition:  # TODO: hard-coded conditions
                    train_generator = False
                    gen_steps = 0

                    # Show a sample from the generator
                    grid_show(generator(seed(8, gen_input_dimension)))

            data_images, _ = data

            #try:
            #    data_images_2, _ = data_iterator.__next__()
            #except StopIteration:
            #    return discriminator, generator

            gen_optimizer.zero_grad()
            dis_optimizer.zero_grad()

            objective, over_trained = dis_objective(discriminator,
                                                    generator,
                                                    data_images,
                                                    batch_size,verbose=True)

            objective.backward()
            dis_optimizer.step()

            count += 1
            dis_steps += 1

            if dis_steps > 10 or over_trained:
                train_generator = True
                dis_steps = 0

            if count % 100 ==0:
                print(count)

    return discriminator, generator


# --------------------------------------------------------------------------
# Ancillary tools
# --------------------------------------------------------------------------

def grid_show(images):
    img = images.detach()
    img = torchvision.utils.make_grid(img)
    img = img / 2 + 0.5
    img = img.numpy()
    plt.imshow(np.transpose(img, (1, 2, 0)))

    #plt.interactive(True)
    plt.show(block=False)
    # plt.interactive(False)

    plt.pause(.01)
    plt.close()


def seed(*shape):

    return 2.*torch.rand(shape) - 1.

#

#

#

discriminator, generator = trainer(1)