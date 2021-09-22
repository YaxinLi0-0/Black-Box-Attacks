from __future__ import print_function
try:
	raw_input
except:
	raw_input = input

import numpy as np
import time
import os
from PIL import Image
from torchvision import models, transforms, datasets
from resnet import *

def orthogonal_perturbation(delta, prev_sample, target_sample):

    #Generate orthogonal perturbation.
	perturb = np.random.randn(1, 3, 32, 32)
	perturb /= np.linalg.norm(perturb, axis=(2, 3))
	perturb *= delta * np.mean(get_diff(target_sample, prev_sample))

    # Project perturbation onto sphere around target
	diff = (target_sample - prev_sample).astype(np.float32) # Orthorgonal vector to sphere surface
	diff /= get_diff(target_sample, prev_sample) # Orthogonal unit vector
	# We project onto the orthogonal then subtract from perturb
	# to get projection onto sphere surface
	perturb -= (np.vdot(perturb, diff) / np.linalg.norm(diff)**2) * diff

    # Check overflow and underflow
	overflow = (prev_sample + perturb) - 1
	perturb -= overflow * (overflow > 0)
	underflow = -(prev_sample + preturb)
	perturb += underflow * (underflow > 0)
	return perturb


def forward_perturbation(epsilon, prev_sample, target_sample):
	"""Generate forward perturbation."""
	perturb = target_sample - prev_sample
	perturb *= epsilon
	return perturb


def get_diff(sample_1, sample_2):
	"""Channel-wise norm of difference between samples."""
	return np.linalg.norm(sample_1 - sample_2, axis=(2, 3))


def boundary_attack(classifier, initial_sample, target_sample, attack_class, target_class, folder):
# Load model, images and other parameters
    adversarial_sample = initial_sample
    n_steps = 0
    n_calls = 0
    epsilon = 1.
    delta = 0.1

    # Move first step to the boundary
    while True:
        trial_sample = adversarial_sample + forward_perturbation(epsilon, adversarial_sample, target_sample)
        prediction = classifier(trial_sample)
        n_calls += 1
        if torch.argmax(prediction) == attack_class:
            adversarial_sample = trial_sample
            break
        else:
            epsilon *= 0.9

    # Iteratively run attack
    while True:
        print("Step #{}...".format(n_steps))
        # Orthogonal step
        print("\tDelta step...")
        d_step = 0
        while True:
            d_step += 1
            print("\t#{}".format(d_step))
            trial_samples = []
            for i in np.arange(10):
                trial_sample = adversarial_sample + orthogonal_perturbation(delta, adversarial_sample, target_sample)
                trial_samples.append(trial_sample)

            predictions = classifier(trial_samples)
            n_calls += 10
            predictions = np.argmax(predictions, axis=1)
            d_score = np.mean(predictions == attack_class)
            if d_score > 0.0:
                if d_score < 0.3:
                    delta *= 0.9
                elif d_score > 0.7:
                    delta /= 0.9
                adversarial_sample = np.array(trial_samples)[np.where(predictions == attack_class)[0][0]]
                break
            else:
                delta *= 0.9
        # Forward step
        print("\tEpsilon step...")
        e_step = 0
        while True:
            e_step += 1
            print("\t#{}".format(e_step))
            trial_sample = adversarial_sample + forward_perturbation(epsilon, adversarial_sample, target_sample)

            prediction = classifier(trial_sample)
            n_calls += 1
            if np.argmax(prediction) == attack_class:
                adversarial_sample = trial_sample
                epsilon /= 0.5
                break
            elif e_step > 500:
                    break
            else:
                epsilon *= 0.5

        n_steps += 1
        chkpts = [1, 5, 10, 50, 100, 500]
        if (n_steps in chkpts) or (n_steps % 500 == 0):
            print("{} steps".format(n_steps))
            save_image(np.copy(adversarial_sample), classifier, folder)
        diff = np.mean(get_diff(adversarial_sample, target_sample))
        if diff <= 1e-3 or e_step > 500:
            print("{} steps".format(n_steps))
            print("Mean Squared Error: {}".format(diff))
            save_image(np.copy(adversarial_sample), classifier, folder)
            break

        print("Mean Squared Error: {}".format(diff))
        print("Calls: {}".format(n_calls))
        print("Attack Class: {}".format(attack_class))
        print("Target Class: {}".format(target_class))
        print("Adversarial Class: {}".format(np.argmax(prediction)))


if __name__ == "__main__":

    classifier = ResNet18()

    #initial_sample = preprocess('image/awkward_moment_seal.png')
    #target_sample = preprocess('image/bad_joke_eel.png')
    test_loader  = torch.utils.data.DataLoader(
                            datasets.CIFAR10('~/Documents/data', train = False, download=True,
                            transform = transforms.ToTensor()),
                            batch_size = 1, shuffle=True) #, **kwargs)

    import ipdb
    ipdb.set_trace()

    initial_sample, attack_class = next(iter(test_loader))

    target_sample, target_class = next(iter(test_loader))
    while (attack_class == target_class):
        target_sample, target_class = next(iter(test_loader))

    folder = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    os.mkdir(os.path.join("image", folder))

    boundary_attack(classifier, initial_sample, target_sample, attack_class, target_class, folder)
