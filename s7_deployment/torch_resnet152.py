import torch
import torchvision.models as models
import torch.jit

# Load ResNet-152 (you can use a pretrained or random initialized model)
model = models.resnet152(pretrained=True)

# Set the model to evaluation mode
model.eval()

# Create a random input tensor to use for tracing
input_tensor = torch.randn(1, 3, 224, 224)  # Assuming 3-channel images of size 224x224

# Script the model using torch.jit.script
scripted_model = torch.jit.script(model)

# Save the scripted model to a file if needed
scripted_model.save("resnet152_scripted.pt")

# Run inference with the scripted model
with torch.no_grad():
    output = scripted_model(input_tensor)

# Print the output
print(output)