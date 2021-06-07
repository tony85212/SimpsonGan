from model import *
from dataset import *
import torch.optim as optim
import time
import pandas as pd
import itertools

class SimpsonGan():
    def __init__(self, params):
        self.params = params
        self.G_A = Generator(3, self.params['ngf'], 3, self.params['num_resnet']).cuda() # input_dim, num_filter, output_dim, num_resnet
        self.G_B = Generator(3, self.params['ngf'], 3, self.params['num_resnet']).cuda()
        self.D_A = Discriminator(3, self.params['ndf'], 1).cuda() # input_dim, num_filter, output_dim
        self.D_B = Discriminator(3, self.params['ndf'], 1).cuda()
        #local discriminator
        self.D_A_eyes = Discriminator(3, self.params['ndf'], 1).cuda() # input_dim, num_filter, output_dim
        self.D_B_eyes = Discriminator(3, self.params['ndf'], 1).cuda()
        self.D_A_mouth = Discriminator(3, self.params['ndf'], 1).cuda() # input_dim, num_filter, output_dim
        self.D_B_mouth = Discriminator(3, self.params['ndf'], 1).cuda()
        #initialize weight
        self.G_A.normal_weight_init(mean=0.0, std=0.02)
        self.G_B.normal_weight_init(mean=0.0, std=0.02)
        self.D_A.normal_weight_init(mean=0.0, std=0.02)
        self.D_B.normal_weight_init(mean=0.0, std=0.02)
        self.D_A_eyes.normal_weight_init(mean=0.0, std=0.02)
        self.D_B_eyes.normal_weight_init(mean=0.0, std=0.02)
        self.D_A_mouth.normal_weight_init(mean=0.0, std=0.02)
        self.D_B_mouth.normal_weight_init(mean=0.0, std=0.02)
        #landmarks
        self.landmarks = []

    def load_model(self):
        self.G_A.load_state_dict(torch.load('model/G_A.pth'))
        self.G_B.load_state_dict(torch.load('model/G_B.pth'))
        self.D_A.load_state_dict(torch.load('model/D_A.pth'))
        self.D_B.load_state_dict(torch.load('model/D_B.pth'))
        #local discriminator
        self.D_A_eyes.load_state_dict(torch.load('model/D_A_eyes.pth'))
        self.D_B_eyes.load_state_dict(torch.load('model/D_B_eyes.pth'))
        self.D_A_mouth.load_state_dict(torch.load('model/D_A_mouth.pth'))
        self.D_B_mouth.load_state_dict(torch.load('model/D_B_mouth.pth'))

    def save_model(self):
        torch.save(self.G_A.state_dict(), 'model/G_A.pth')
        torch.save(self.G_B.state_dict(), 'model/G_B.pth')
        torch.save(self.D_A.state_dict(), 'model/D_A.pth')
        torch.save(self.D_B.state_dict(), 'model/D_B.pth')
        #local discriminator
        torch.save(self.D_A_eyes.state_dict(), 'model/D_A_eyes.pth')
        torch.save(self.D_B_eyes.state_dict(), 'model/D_B_eyes.pth')
        torch.save(self.D_A_mouth.state_dict(), 'model/D_A_mouth.pth')
        torch.save(self.D_B_mouth.state_dict(), 'model/D_B_mouth.pth')

    def save_checkpoints(self, epoch):
        torch.save(self.G_A.state_dict(), 'model/G_A_' + epoch + '.pth')
        torch.save(self.G_B.state_dict(), 'model/G_B_' + epoch + '.pth')
        torch.save(self.D_A.state_dict(), 'model/D_A_' + epoch + '.pth')
        torch.save(self.D_B.state_dict(), 'model/D_B_' + epoch + '.pth')
        #local discriminator
        torch.save(self.D_A_eyes.state_dict(), 'model/D_A_eyes_' + epoch + '.pth')
        torch.save(self.D_B_eyes.state_dict(), 'model/D_B_eyes_' + epoch + '.pth')
        torch.save(self.D_A_mouth.state_dict(), 'model/D_A_mouth_' + epoch + '.pth')
        torch.save(self.D_B_mouth.state_dict(), 'model/D_B_mouth_' + epoch + '.pth')
        print("Save checkpoits!")

    def crop_eyes(self, img, id, AB):

        crop = -torch.ones(1, 3, 256, 256).cuda()
        if (AB == 'A'):
            crop[0, :, :64, :] = img[0, :, 88:152, :]
        else:
            a = self.landmarks[id]
            crop[0, :, :64, :] = img[0, :, a:a+64, :]
        return crop

    def crop_mouth(self, img, id, AB):

        crop = -torch.ones(1, 3, 256, 256).cuda()
        if (AB == 'A'):
            crop[0, :, :64, :] = img[0, :, 152:216, :]
        else:
            a = self.landmarks[id]+64
            k = 64
            if(a + 64 > 256):
                k = 256 - a
            crop[0, :, :k, :] = img[0, :, a:a+64, :]
        return crop

    def train(self, trainA, trainB):
        #initialize optimizer
        G_optimizer = torch.optim.Adam(itertools.chain(self.G_A.parameters(), self.G_B.parameters()), lr=self.params['lrG'], betas=(self.params['beta1'], self.params['beta2']))
        D_A_optimizer = torch.optim.Adam(self.D_A.parameters(), lr=self.params['lrD'], betas=(self.params['beta1'], self.params['beta2']))
        D_B_optimizer = torch.optim.Adam(self.D_B.parameters(), lr=self.params['lrD'], betas=(self.params['beta1'], self.params['beta2']))
        D_A_eyes_optimizer = torch.optim.Adam(self.D_A_eyes.parameters(), lr=self.params['lrD'], betas=(self.params['beta1'], self.params['beta2']))
        D_B_eyes_optimizer = torch.optim.Adam(self.D_B_eyes.parameters(), lr=self.params['lrD'], betas=(self.params['beta1'], self.params['beta2']))
        D_A_mouth_optimizer = torch.optim.Adam(self.D_A_eyes.parameters(), lr=self.params['lrD'], betas=(self.params['beta1'], self.params['beta2']))
        D_B_mouth_optimizer = torch.optim.Adam(self.D_B_eyes.parameters(), lr=self.params['lrD'], betas=(self.params['beta1'], self.params['beta2']))
        #initialize loss
        MSE_Loss = torch.nn.MSELoss().cuda()
        L1_Loss = torch.nn.L1Loss().cuda()
        #record
        D_A_avg_losses = []
        D_B_avg_losses = []
        G_A_avg_losses = []
        G_B_avg_losses = []
        cycle_A_avg_losses = []
        cycle_B_avg_losses = []
        D_A_eyes_avg_losses = []
        D_A_mouth_avg_losses = []
        D_B_eyes_avg_losses = []
        D_B_mouth_avg_losses = []
        #generated image pool
        num_pool = 50
        fake_A_pool = ImagePool(num_pool)
        fake_B_pool = ImagePool(num_pool)
        fake_A_eyes_pool = ImagePool(num_pool)
        fake_B_eyes_pool = ImagePool(num_pool)
        fake_A_mouth_pool = ImagePool(num_pool)
        fake_B_mouth_pool = ImagePool(num_pool)
        #sample
        train_real_A, _ = next(iter(trainA))
        train_real_A = train_real_A.cuda()
        train_real_B, _ = next(iter(trainB))
        train_real_B = train_real_B.cuda()
        self.landmarks = load_landmarks()
        try:
            self.load_model()
            print("Loading previos model")
        except:
            print("Training new model")
        step = 0
        for epoch in range(self.params['num_epochs']):
            D_A_losses = []
            D_B_losses = []
            G_A_losses = []
            G_B_losses = []
            cycle_A_losses = []
            cycle_B_losses = []
            D_A_eyes_losses = []
            D_B_eyes_losses = []
            D_A_mouth_losses = []
            D_B_mouth_losses = []

            # Learing rate decay
            if(epoch + 1) > self.params['decay_epoch']:
                D_A_optimizer.param_groups[0]['lr'] -= self.params['lrD'] / (self.params['num_epochs'] - self.params['decay_epoch'])
                D_B_optimizer.param_groups[0]['lr'] -= self.params['lrD'] / (self.params['num_epochs'] - self.params['decay_epoch'])
                D_A_eyes_optimizer.param_groups[0]['lr'] -= self.params['lrD'] / (self.params['num_epochs'] - self.params['decay_epoch'])
                D_B_eyes_optimizer.param_groups[0]['lr'] -= self.params['lrD'] / (self.params['num_epochs'] - self.params['decay_epoch'])
                D_A_mouth_optimizer.param_groups[0]['lr'] -= self.params['lrD'] / (self.params['num_epochs'] - self.params['decay_epoch'])
                D_B_mouth_optimizer.param_groups[0]['lr'] -= self.params['lrD'] / (self.params['num_epochs'] - self.params['decay_epoch'])
                G_optimizer.param_groups[0]['lr'] -= self.params['lrG'] / (self.params['num_epochs'] - self.params['decay_epoch'])

            start_time = time.time()
            # training
            for i, data in enumerate(zip(trainA, trainB)):

                # input image data
                dataA, _ = data[0]
                dataB, _ = data[1]
                real_A = dataA.cuda()
                real_B = dataB.cuda()

                real_A_eyes = self.crop_eyes(real_A, i, "A")
                real_B_eyes = self.crop_eyes(real_B, i, "B")
                real_A_mouth = self.crop_mouth(real_A, i, "A")
                real_B_mouth = self.crop_mouth(real_B, i, "B")


                # -------------------------- train generator G --------------------------
                # A --> B
                fake_B = self.G_A(real_A)
                D_B_fake_decision = self.D_B(fake_B)
                G_A_loss = MSE_Loss(D_B_fake_decision, Variable(torch.ones(D_B_fake_decision.size()).cuda()))
                # ---  local   ----
                fake_B_eyes = self.crop_eyes(fake_B, i, "A")
                D_B_fake_eyes_decision = self.D_B_eyes(fake_B_eyes)
                G_A_eyes_loss = MSE_Loss(D_B_fake_eyes_decision, Variable(torch.ones(D_B_fake_eyes_decision.size()).cuda()))
                # mouth
                fake_B_mouth = self.crop_mouth(fake_B, i, "A")
                D_B_fake_mouth_decision = self.D_B_mouth(fake_B_mouth)
                G_A_mouth_loss = MSE_Loss(D_B_fake_mouth_decision, Variable(torch.ones(D_B_fake_mouth_decision.size()).cuda()))

                # forward cycle loss
                recon_A = self.G_B(fake_B)
                cycle_A_loss = L1_Loss(recon_A, real_A) * self.params['lambdaA']
                # ---  local   ----
                recon_A_eyes = self.crop_eyes(recon_A, i, "A")
                cycle_A_eyes_loss = L1_Loss(recon_A_eyes, real_A_eyes) * self.params['lambdaC']
                # mouth
                recon_A_mouth = self.crop_mouth(recon_A, i, "A")
                cycle_A_mouth_loss = L1_Loss(recon_A_mouth, real_A_mouth) * self.params['lambdaD']

                # B --> A
                fake_A = self.G_B(real_B)
                D_A_fake_decision = self.D_A(fake_A)
                G_B_loss = MSE_Loss(D_A_fake_decision, Variable(torch.ones(D_A_fake_decision.size()).cuda()))
                # ---  local   ----
                fake_A_eyes = self.crop_eyes(fake_A, i, "B")
                D_A_fake_eyes_decision = self.D_A_eyes(fake_A_eyes)
                G_B_eyes_loss = MSE_Loss(D_A_fake_eyes_decision, Variable(torch.ones(D_A_fake_eyes_decision.size()).cuda()))
                # mouth
                fake_A_mouth = self.crop_mouth(fake_A, i, "B")
                D_A_fake_mouth_decision = self.D_A_mouth(fake_A_mouth)
                G_B_mouth_loss = MSE_Loss(D_A_fake_mouth_decision, Variable(torch.ones(D_A_fake_mouth_decision.size()).cuda()))

                # backward cycle loss
                recon_B = self.G_A(fake_A)
                cycle_B_loss = L1_Loss(recon_B, real_B) * self.params['lambdaB']
                # ---  local   ----
                recon_B_eyes = self.crop_eyes(recon_B, i, "B")
                cycle_B_eyes_loss = L1_Loss(recon_B_eyes, real_B_eyes) * self.params['lambdaC']
                # mouth
                recon_B_mouth = self.crop_mouth(recon_B, i, "B")
                cycle_B_mouth_loss = L1_Loss(recon_B_mouth, real_B_mouth) * self.params['lambdaD']

                G_eyes_loss =  G_A_eyes_loss + G_B_eyes_loss + cycle_A_eyes_loss  + cycle_B_eyes_loss
                G_mouth_loss =  G_A_mouth_loss + G_B_mouth_loss + cycle_A_mouth_loss  + cycle_B_mouth_loss
                # Back propagation
                G_loss = G_A_loss + G_B_loss + cycle_A_loss + cycle_B_loss + self.params['eyes_weight'] * G_eyes_loss + self.params['mouth_weight'] * G_mouth_loss

                G_optimizer.zero_grad()
                G_loss.backward()
                G_optimizer.step()


                # -------------------------- train discriminator self.D_A --------------------------
                D_A_real_decision = self.D_A(real_A)
                D_A_real_loss = MSE_Loss(D_A_real_decision, Variable(torch.ones(D_A_real_decision.size()).cuda()))

                fake_A = fake_A_pool.query(fake_A)

                D_A_fake_decision = self.D_A(fake_A)
                D_A_fake_loss = MSE_Loss(D_A_fake_decision, Variable(torch.zeros(D_A_fake_decision.size()).cuda()))

                # Back propagation
                D_A_loss = (D_A_real_loss + D_A_fake_loss) * 0.5
                D_A_optimizer.zero_grad()
                D_A_loss.backward()
                D_A_optimizer.step()

                # -------------------------- train discriminator self.D_A_eyes --------------------------

                D_A_real_eyes_decision = self.D_A(real_A_eyes)
                D_A_real_eyes_loss = MSE_Loss(D_A_real_eyes_decision, Variable(torch.ones(D_A_real_eyes_decision.size()).cuda()))

                fake_A_eyes = fake_A_eyes_pool.query(fake_A_eyes)

                D_A_fake_eyes_decision = self.D_A(fake_A_eyes)
                D_A_fake_eyes_loss = MSE_Loss(D_A_fake_eyes_decision, Variable(torch.zeros(D_A_fake_eyes_decision.size()).cuda()))

                # Back propagation
                D_A_eyes_loss = (D_A_real_eyes_loss + D_A_fake_eyes_loss) * 0.5
                D_A_eyes_optimizer.zero_grad()
                D_A_eyes_loss.backward()
                D_A_eyes_optimizer.step()

                # -------------------------- train discriminator self.D_A_mouth --------------------------

                D_A_real_mouth_decision = self.D_A(real_A_mouth)
                D_A_real_mouth_loss = MSE_Loss(D_A_real_mouth_decision, Variable(torch.ones(D_A_real_mouth_decision.size()).cuda()))

                fake_A_mouth = fake_A_mouth_pool.query(fake_A_mouth)

                D_A_fake_mouth_decision = self.D_A(fake_A_mouth)
                D_A_fake_mouth_loss = MSE_Loss(D_A_fake_mouth_decision, Variable(torch.zeros(D_A_fake_mouth_decision.size()).cuda()))

                # Back propagation
                D_A_mouth_loss = (D_A_real_mouth_loss + D_A_fake_mouth_loss) * 0.5
                D_A_mouth_optimizer.zero_grad()
                D_A_mouth_loss.backward()
                D_A_mouth_optimizer.step()

                # -------------------------- train discriminator self.D_B --------------------------
                D_B_real_decision = self.D_B(real_B)
                D_B_real_loss = MSE_Loss(D_B_real_decision, Variable(torch.ones(D_B_fake_decision.size()).cuda()))

                fake_B = fake_B_pool.query(fake_B)

                D_B_fake_decision = self.D_B(fake_B)
                D_B_fake_loss = MSE_Loss(D_B_fake_decision, Variable(torch.zeros(D_B_fake_decision.size()).cuda()))

                # Back propagation
                D_B_loss = (D_B_real_loss + D_B_fake_loss) * 0.5
                D_B_optimizer.zero_grad()
                D_B_loss.backward()
                D_B_optimizer.step()

                # -------------------------- train discriminator self.D_B_eyes --------------------------

                D_B_real_eyes_decision = self.D_B(real_B_eyes)
                D_B_real_eyes_loss = MSE_Loss(D_B_real_eyes_decision, Variable(torch.ones(D_B_real_eyes_decision.size()).cuda()))

                fake_B_eyes = fake_B_eyes_pool.query(fake_B_eyes)

                D_B_fake_eyes_decision = self.D_B(fake_B_eyes)
                D_B_fake_eyes_loss = MSE_Loss(D_B_fake_eyes_decision, Variable(torch.zeros(D_B_fake_eyes_decision.size()).cuda()))

                # Back propagation
                D_B_eyes_loss = (D_B_real_eyes_loss + D_B_fake_eyes_loss) * 0.5
                D_B_eyes_optimizer.zero_grad()
                D_B_eyes_loss.backward()
                D_B_eyes_optimizer.step()

                # -------------------------- train discriminator self.D_B_mouth --------------------------

                D_B_real_mouth_decision = self.D_B(real_B_mouth)
                D_B_real_mouth_loss = MSE_Loss(D_B_real_mouth_decision, Variable(torch.ones(D_B_real_mouth_decision.size()).cuda()))

                fake_B_mouth = fake_B_mouth_pool.query(fake_B_mouth)

                D_B_fake_mouth_decision = self.D_B(fake_B_mouth)
                D_B_fake_mouth_loss = MSE_Loss(D_B_fake_mouth_decision, Variable(torch.zeros(D_B_fake_mouth_decision.size()).cuda()))

                # Back propagation
                D_B_mouth_loss = (D_B_real_mouth_loss + D_B_fake_mouth_loss) * 0.5
                D_B_mouth_optimizer.zero_grad()
                D_B_mouth_loss.backward()
                D_B_mouth_optimizer.step()


                # ------------------------ Print -----------------------------
                # loss values
                D_A_losses.append(D_A_loss.item())
                D_B_losses.append(D_B_loss.item())
                G_A_losses.append(G_A_loss.item())
                G_B_losses.append(G_B_loss.item())
                cycle_A_losses.append(cycle_A_loss.item())
                cycle_B_losses.append(cycle_B_loss.item())
                # -- local --
                D_A_eyes_losses.append(D_A_eyes_loss.item())
                D_B_eyes_losses.append(D_B_eyes_loss.item())
                D_A_mouth_losses.append(D_A_mouth_loss.item())
                D_B_mouth_losses.append(D_B_mouth_loss.item())
                if i%100 == 0:
                    print('Epoch [%d/%d], Step [%d/%d]' %  (epoch+1, self.params['num_epochs'], i+1, len(trainA)))
                    print('D_A_loss: %.4f, D_B_loss: %.4f, G_A_loss: %.4f, G_B_loss: %.4f'
                          % (D_A_loss.item(), D_B_loss.item(), G_A_loss.item(), G_B_loss.item()))
                    print('D_A_eyes_loss: %.4f, D_B_eyes_loss: %.4f, G_A_mouth_loss: %.4f, G_B_mouth_loss: %.4f'
                          % (D_A_eyes_loss.item(), D_B_eyes_loss.item(), G_A_mouth_loss.item(), G_B_mouth_loss.item()))
                step += 1

            end_time = time.time()
            print(" --- Each Epoch Time : %.2f --- " % (end_time - start_time))

            D_A_avg_loss = torch.mean(torch.FloatTensor(D_A_losses))
            D_B_avg_loss = torch.mean(torch.FloatTensor(D_B_losses))
            G_A_avg_loss = torch.mean(torch.FloatTensor(G_A_losses))
            G_B_avg_loss = torch.mean(torch.FloatTensor(G_B_losses))
            cycle_A_avg_loss = torch.mean(torch.FloatTensor(cycle_A_losses))
            cycle_B_avg_loss = torch.mean(torch.FloatTensor(cycle_B_losses))
            D_A_eyes_avg_loss = torch.mean(torch.FloatTensor(D_A_eyes_losses))
            D_B_eyes_avg_loss = torch.mean(torch.FloatTensor(D_B_eyes_losses))
            D_A_mouth_avg_loss = torch.mean(torch.FloatTensor(D_A_mouth_losses))
            D_B_mouth_avg_loss = torch.mean(torch.FloatTensor(D_B_mouth_losses))

            # avg loss values for plot
            D_A_avg_losses.append(D_A_avg_loss.item())
            D_B_avg_losses.append(D_B_avg_loss.item())
            G_A_avg_losses.append(G_A_avg_loss.item())
            G_B_avg_losses.append(G_B_avg_loss.item())
            cycle_A_avg_losses.append(cycle_A_avg_loss.item())
            cycle_B_avg_losses.append(cycle_B_avg_loss.item())
            D_A_eyes_avg_losses.append(D_A_eyes_avg_loss.item())
            D_B_eyes_avg_losses.append(D_B_eyes_avg_loss.item())
            D_A_mouth_avg_losses.append(D_A_mouth_avg_loss.item())
            D_B_mouth_avg_losses.append(D_B_mouth_avg_loss.item())

            #record image in each iteration
            train_fake_B = self.G_A(train_real_A)
            train_recon_A = self.G_B(train_fake_B)

            train_fake_A = self.G_B(train_real_B)
            train_recon_B = self.G_A(train_fake_A)

            if (epoch == 0):
                save_image(train_real_A, 'tmp/', 'real_A')
                save_image(train_real_B, 'tmp/', 'real_B')
            if (epoch % 10 == 0):
                save_image(train_fake_B, 'tmp/', 'fake_B', epoch)
                save_image(train_fake_A, 'tmp/', 'fake_A', epoch)
                save_image(train_recon_A , 'tmp/', 'recons_A', epoch)
                save_image(train_recon_B , 'tmp/', 'recons_B', epoch)
                self.save_checkpoints(epoch)
            #saving model
            self.save_model()

        all_losses = pd.DataFrame()
        all_losses['D_A_avg_losses'] = D_A_avg_losses
        all_losses['D_B_avg_losses'] = D_B_avg_losses
        all_losses['G_A_avg_losses'] = G_A_avg_losses
        all_losses['G_B_avg_losses'] = G_B_avg_losses
        all_losses['cycle_A_avg_losses'] = cycle_A_avg_losses
        all_losses['cycle_B_avg_losses'] = cycle_B_avg_losses
        all_losses['D_A_eyes_avg_losses'] = D_A_eyes_avg_losses
        all_losses['D_B_eyes_avg_losses'] = D_B_eyes_avg_losses
        all_losses['D_A_mouth_avg_losses'] = D_A_mouth_avg_losses
        all_losses['D_B_mouth_avg_losses'] = D_B_mouth_avg_losses
        all_losses.to_csv('avg_losses',index=False)

    def test(self, testA, testB):
        self.load_model()
        #test
        for i, data in enumerate(zip(testA, testB)):
            # translate
            dataA, _ = data[0]
            dataB, _ = data[1]
            test_real_A = dataA.cuda()
            test_real_B = dataB.cuda()
            test_fake_B = self.G_A(test_real_A)
            test_recon_A = self.G_B(test_fake_B)
            test_fake_A = self.G_B(test_real_B)
            test_recon_B = self.G_A(test_fake_A)
            # save to result
            save_image(test_real_A, 'result/', 'real_A', i)
            save_image(test_real_B, 'result/', 'real_B', i)
            save_image(test_fake_B, 'result/', 'fake_B', i)
            save_image(test_fake_A, 'result/', 'fake_A', i)
            save_image(test_recon_A , 'result/', 'recons_A', i)
            save_image(test_recon_B , 'result/', 'recons_B', i)

    def custom(self, image, model):
        self.load_model()
        img = load_custom(image)
        img = img.cuda()
        if(model == 'ga'):
            result = self.G_A(img)
            save_image(result, '', 'result', 0)
        elif(model == 'gb'):
            result = self.G_B(img)
            save_image(result, '', 'result', 0)
        elif (model == 'da'):
            decision = self.D_A(img)
            print(decision)
        elif (model == 'db'):
            decision = self.D_B(img)
            print(decision)
