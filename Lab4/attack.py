
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def fgsm_attack(model, loss_fn, images,labels, eps, target):
    """
        This function implements FGSM Attack
        
        Parameters
        ----------
        model : Classifier network
        loss_fn : Loss function : MSE loss
        images : Images to attack
        eps : Percentual of perturbation
        target : target tha attack, if true the labels will be used as target labels
        Returns
        -------
        advImages : Tensor of the images perturbated
        attackAccuracy: return the accuracy of the attack
    """

    correct, total  = 0, 0
            
    new_tensor = images.detach().clone()
    new_tensor = new_tensor.to(device)
    
    if new_tensor.grad is not None:
            new_tensor.grad.data.fill_(0)
            
    model.zero_grad()

    # Predict label before attack
    outputs = model(images)
    _, predicted_labels = torch.max(outputs.data, 1)

    # Compute the gradient wrt images
    new_tensor.requires_grad = True
    
    outputs = model(new_tensor)
    
    # If not target i want to change the direction
    if target == False:
        loss = -loss_fn(outputs, labels)

    # If target i dont want to change the direction
    else:
        loss = loss_fn(outputs, labels)

    _, predicted_adv_labels = torch.max(outputs.data, 1)
    
    loss.backward()

    new_tensor.requires_grad = False

    #Caluclate the preturbate image by adding the noise and clamp it between 0 and 1
    perturbed_images = torch.clamp(new_tensor - eps * new_tensor.grad.data.sign(), 0, 1)
    
    correct += (predicted_labels == predicted_adv_labels).sum().item()
    total += labels.size(0)

    
    return perturbed_images, 1-correct/total
