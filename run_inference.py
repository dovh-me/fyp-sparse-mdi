import grpc
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
import generated.server_pb2 as server_pb2
import generated.server_pb2_grpc as server_pb2_grpc
import asyncio
import traceback
from util import status

# Load and preprocess CIFAR-10 data
def load_cifar10():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_test = x_test.astype('float32') / 255.0
    y_test = y_test.flatten()  # Flatten labels
    return x_test, y_test

# Perform inference asynchronously
async def perform_inference(server_address: str, x_test, y_test):
    async with grpc.aio.insecure_channel(server_address) as channel:
        stub = server_pb2_grpc.ServerStub(channel)

        correct_predictions = 0

        for i, (image, true_label) in enumerate(zip(x_test, y_test)):
            input_tensor = image  # Flatten the image to a list
            input_tensor = input_tensor.astype('float32') / 255.0
            input_tensor = input_tensor.tobytes()
            request = server_pb2.StartInferenceRequest(input_tensor=input_tensor)
            
            try:
                # Send inference request
                response = await stub.StartInference(request)
                if(response.status_code != status.INFERENCE_SUCCESS):
                    print(response.message)
                    continue

                predicted_label = np.argmax(response.result)  # Assume result is a probability distribution

                if predicted_label == true_label:
                    correct_predictions += 1
            except grpc.aio.AioRpcError as e:
                print(f"RPC error occurred: {e}")
            except:
                traceback.print_exc()

            # Optional: Break early for testing
            if i >= 100:  # Test on the first 100 images
                break

        accuracy = correct_predictions / (i + 1)
        print(f"Accuracy: {accuracy:.2%}")

# Entry point
if __name__ == "__main__":
    server_address = "127.0.0.1:50051"
    x_test, y_test = load_cifar10()
    asyncio.run(perform_inference(server_address, x_test, y_test))