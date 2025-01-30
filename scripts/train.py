import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

def train(save_model=False):
    mean_generator_loss = 0
    mean_discriminator_loss = 0
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    cur_step = 0

    for epoch in range(n_epochs):
        for real_A, real_B in tqdm(dataloader):
            # Resize images to target dimension
            real_A = nn.functional.interpolate(real_A, size=target_shape)
            real_B = nn.functional.interpolate(real_B, size=target_shape)
            cur_batch_size = len(real_A)
            real_A = real_A.to(device)
            real_B = real_B.to(device)

            # *Update Discriminator A*
            disc_A_opt.zero_grad() # Zero the gradients for discriminator A
            with torch.no_grad():
                fake_A = gen_BA(real_B) # Generate fake images
            disc_A_loss = get_disc_loss(real_A, fake_A, disc_A, adv_criterion) # Calculate loss
            disc_A_loss.backward(retain_graph=True) # Backpropogation
            disc_A_opt.step() # Update discriminator weights

            # *Update Discriminator B*
            disc_B_opt.zero_grad() # Zero the gradients for discriminator B
            with torch.no_grad():
                fake_B = gen_AB(real_A) # Generate fake images
            disc_B_loss = get_disc_loss(real_B, fake_B, disc_B, adv_criterion) # Calculate loss
            disc_B_loss.backward(retain_graph=True) # Backpropogation
            disc_B_opt.step() # Update discriminator weights

            # *Update Generator*
            gen_opt.zero_grad() # Zero the gradients for generator
            gen_loss, fake_A, fake_B = get_gen_loss(
                real_A, real_B, gen_AB, gen_BA, disc_A, disc_B, adv_criterion, recon_criterion, recon_criterion
            ) # Calculate loss
            gen_loss.backward() # Backpropogation
            gen_opt.step() # Update generator weights

            mean_discriminator_loss += disc_A_loss.item() / display_step
            mean_generator_loss += gen_loss.item() / display_step

            # *Visualization*
            if cur_step % display_step == 0:
                print(f"Epoch {epoch}: Step {cur_step}: Generator (U-Net) loss: {mean_generator_loss}, Discriminator loss: {mean_discriminator_loss}")
                show_tensor_images(torch.cat([real_A, real_B]), size=(dim_A, target_shape, target_shape))
                show_tensor_images(torch.cat([fake_B, fake_A]), size=(dim_B, target_shape, target_shape))
                mean_generator_loss = 0
                mean_discriminator_loss = 0
                if save_model:
                    torch.save({
                        'gen_AB': gen_AB.state_dict(),
                        'gen_BA': gen_BA.state_dict(),
                        'gen_opt': gen_opt.state_dict(),
                        'disc_A': disc_A.state_dict(),
                        'disc_A_opt': disc_A_opt.state_dict(),
                        'disc_B': disc_B.state_dict(),
                        'disc_B_opt': disc_B_opt.state_dict()
                    }, f"/content/cycleGAN.pth")
            cur_step += 1