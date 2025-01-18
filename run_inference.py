import grpc
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
import generated.server_pb2 as server_pb2
import generated.server_pb2_grpc as server_pb2_grpc
import asyncio
import traceback
from util import status
from tqdm import tqdm

# Load and preprocess CIFAR-10 data
def load_cifar10():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    return x_test, y_test

def preprocess_image(image, label):
    image = tf.cast(image, tf.float32) / 255.0  # Normalize to [0, 1]
    image = (image - 0.5) / 0.5                 # Normalize to [-1, 1]
    return image, label

# Perform inference asynchronously
async def perform_inference(server_address: str, x_test, y_test):
    test_images = x_test
    test_labels = y_test
    test_images = np.transpose(test_images, (0, 3, 1, 2))

    # Load the dataset
    test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
    test_dataset = test_dataset.map(preprocess_image).batch(1)

    async with grpc.aio.insecure_channel(server_address) as channel:
        stub = server_pb2_grpc.ServerStub(channel)

        correct_predictions = 0
        total_predictions = len(test_dataset)

        print("Starting inference...")
        for image, label in tqdm(test_dataset, total=total_predictions, desc="Inference Progress"):
            input_tensor = image.numpy().tobytes()
            request = server_pb2.StartInferenceRequest(input_tensor=input_tensor)
            
            try:
                # Send inference request
                response = await stub.StartInference(request)
                if(response.status_code != status.INFERENCE_SUCCESS):
                    print(response.message)
                    continue

                result_shape = (1,10)
                input_array = np.frombuffer(response.result, dtype=np.float32)
                predicted = np.argmax(input_array)  # Assume result is a probability distribution

                if tf.equal(np.array(predicted), label):
                    correct_predictions += 1
            except grpc.aio.AioRpcError as e:
                print(f"RPC error occurred: {e}")
            except:
                traceback.print_exc()

            # Optional: Break early for testing
            # if i >= 100:  # Test on the first 100 images
            #     break

        accuracy = correct_predictions / total_predictions
        print(f"Accuracy: {accuracy:.2%}")

# Entry point
if __name__ == "__main__":
    server_address = "127.0.0.1:50051"
    x_test, y_test = load_cifar10()
    asyncio.run(perform_inference(server_address, x_test, y_test))