import grpc
import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import Dataset, DataLoader
import generated.server_pb2 as server_pb2
import generated.server_pb2_grpc as server_pb2_grpc
import asyncio
import traceback
from util import status
from tqdm import tqdm
import os
from PIL import Image
import json
from sklearn.metrics import classification_report 

# Define a custom dataset class to load Tiny ImageNet-200 and ImageNet-OOD
class TinyImageNetOOD(Dataset):
    def __init__(self, tiny_imagenet_dir, ood_dir, transform=None):
        self.tiny_imagenet_dir = tiny_imagenet_dir
        self.ood_dir = ood_dir
        self.transform = transform

        # Load Tiny ImageNet validation set (manually assign labels)
        self.val_images, self.val_labels = self.load_val_labels()

        # Load OOD dataset (assuming OOD images are in the ood_dir folder)
        self.ood_images = [os.path.join(ood_dir, fname) for fname in os.listdir(ood_dir) if fname.endswith('.JPEG')][:2500] if os.path.exists(ood_dir) else []

    def load_val_labels(self):
        """Loads the validation images and their labels from val_annotations.txt."""
        # annotations_file = os.path.join(self.tiny_imagenet_dir, "val", "val_annotations.txt")
        annotations_file = os.path.join(self.tiny_imagenet_dir, "labels.json")
        val_images = []
        val_labels = []
        labels = [] 

        with open(annotations_file, "r") as f:
            s = f.read()
            labels = json.loads(s)
            
        val_images = [os.path.join(tiny_imagenet_dir, label[1]) for label in labels]
        val_labels = [label[0] for label in labels]

        return val_images[:800], val_labels[:800]

    def __len__(self):
        return len(self.val_images) + len(self.ood_images)

    def __getitem__(self, idx):
        if idx < len(self.val_images):
            # Return Tiny ImageNet image
            image_path = self.val_images[idx]
            label = self.val_labels[idx]
            image = Image.open(image_path).convert("RGB")  # Ensure PIL format
        else:
            # Return OOD image with a dummy label
            image_path = self.ood_images[idx - len(self.val_images)]
            image = Image.open(image_path).convert("RGB")  
            label = -1  # No label for OOD data

        if self.transform:
            image = self.transform(image)

        return image, label

# Load Tiny ImageNet-200 dataset and ImageNet-OOD dataset using PyTorch
def load_tiny_imagenet_and_ood(tiny_imagenet_dir, ood_dir):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize from 64x64 → 224x224
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
    ])
    dataset = TinyImageNetOOD(tiny_imagenet_dir=tiny_imagenet_dir, ood_dir=ood_dir, transform=transform)
    return dataset

# Perform inference asynchronously
async def perform_inference(server_address: str, dataset):
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    async with grpc.aio.insecure_channel(server_address) as channel:
        stub = server_pb2_grpc.ServerStub(channel)

        correct_predictions = 0
        total_predictions = 0
        ood_predictions = 0

        pred = []
        ground_truth = []

        print("Starting inference...")
        try:
            inference_iterator = tqdm(dataloader, desc="Inference Progress")
            for image, label in inference_iterator:
                image = image.unsqueeze(0)  # Ensure input shape is (1, C, H, W)
                # print(f"image.shape(): {image.shape}")
                input_tensor = image.numpy().tobytes()
                request = server_pb2.StartInferenceRequest(input_tensor=input_tensor)

                try:
                    # Send inference request
                    response = await stub.StartInference(request)
                    if response.status_code != status.INFERENCE_SUCCESS:
                        print(response.message)
                        continue

                    result_shape = (1, 1000)  # Tiny ImageNet has 200 classes
                    result_array = np.frombuffer(response.result, dtype=np.float64).reshape(result_shape)
                    predicted = np.argmax(result_array)
                    pred.append(predicted)
                    ground_truth.append(label.numpy()[0])

                    # print(f"predicted == label.numpy()[0], {predicted == label.numpy()[0]}, {predicted} {label.numpy()}")
                    # Check for OOD (label == -1) and skip if OOD
                    if label == -1:
                        ood_predictions += 1
                    elif predicted == label.numpy()[0]:
                        correct_predictions += 1
                    total_predictions += 1
                except grpc.aio.AioRpcError as e:
                    print(f"RPC error occurred: {e}")
                except:
                    traceback.print_exc()

            accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
            ood_accuracy = ood_predictions / total_predictions if total_predictions > 0 else 0
            print(f"Accuracy on In-Distribution: {accuracy:.2%}")
            print(f"Accuracy on OOD: {ood_accuracy:.2%}")

            # Call the GetInferenceMetrics RPC
            metrics_request = server_pb2.InferenceMetricsRequest()
            response: server_pb2.InferenceMetricsResponse = await stub.GetInferenceMetrics(metrics_request)
            print(classification_report(ground_truth, pred, digits=4))
            print(f"Total egress bytes: {response.egress_bytes}B")
            print(f"As Mega Bytes: {response.egress_bytes / (1000 * 1000)}MB")
            print(f"Time elapsed: {inference_iterator.format_dict['elapsed']}s")
            print(f"{accuracy:.2%},{response.egress_bytes}B,{response.egress_bytes / (1000 * 1000)}MB,{inference_iterator.format_dict['elapsed']}s")
            # Compute precision, recall, and F1-score for each class
        except Exception as e:
            print(e)

# Entry point
if __name__ == "__main__":
    tiny_imagenet_dir = "./data/imagenet-100"  # Directory containing Tiny ImageNet
    ood_dir = "./data/imagenet_ood"   # Directory containing ImageNet OOD (optional)
    dataset = load_tiny_imagenet_and_ood(tiny_imagenet_dir, ood_dir)
    server_address = "192.168.1.2:50051"
    asyncio.run(perform_inference(server_address, dataset))