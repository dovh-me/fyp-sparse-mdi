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
from sklearn.metrics import classification_report

# Configuration
MAX_CONCURRENT_REQUESTS = 10  # Number of concurrent inferences

# Load and preprocess CIFAR-10 data
def load_cifar10():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    return x_test, y_test

def preprocess_image(image, label):
    image = tf.cast(image, tf.float32) / 255.0  # Normalize to [0, 1]
    image = (image - 0.5) / 0.5                 # Normalize to [-1, 1]
    return image, label

async def send_inference_request(stub, image, label):
    input_tensor = image.numpy().tobytes()
    request = server_pb2.StartInferenceRequest(input_tensor=input_tensor)
    try:
        response = await stub.StartInference(request)
        if response.status_code != status.INFERENCE_SUCCESS:
            print(response.message)
            return None, None
        result_array = np.frombuffer(response.result, dtype=np.float64).reshape([1, 10])
        predicted = np.argmax(result_array)
        return predicted, label[0]
    except grpc.aio.AioRpcError as e:
        print(f"RPC error occurred: {e}")
    except:
        traceback.print_exc()
    return None, None

async def perform_inference(server_address: str, x_test, y_test):
    test_images = np.transpose(x_test, (0, 3, 1, 2))
    test_dataset = tf.data.Dataset.from_tensor_slices((test_images, y_test))
    test_dataset = test_dataset.map(preprocess_image).batch(1)
    
    async with grpc.aio.insecure_channel(server_address) as channel:
        stub = server_pb2_grpc.ServerStub(channel)

        correct_predictions = 0
        pred, ground_truth = [], []
        semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

        async def inference_task(image, label):
            async with semaphore:
                return await send_inference_request(stub, image, label)

        print("Starting inference...")
        inference_iterator = tqdm(test_dataset, total=len(y_test), desc="Inference Progress")
        tasks = [inference_task(image, label) for image, label in inference_iterator]
        results = await asyncio.gather(*tasks)

        for predicted, actual in results:
            if predicted is not None:
                pred.append(predicted)
                ground_truth.append(actual)
                if predicted == actual:
                    correct_predictions += 1

        accuracy = correct_predictions / len(ground_truth) if ground_truth else 0
        print(f"Accuracy: {accuracy:.2%}")
        print(classification_report(ground_truth, pred, digits=4))

        metrics_request = server_pb2.InferenceMetricsRequest()
        response = await stub.GetInferenceMetrics(metrics_request)
        print(f"total egress_bytes: {response.egress_bytes}B")
        print(f"As Mega Bytes: {response.egress_bytes / (1000 * 1000)}MB")
        print(f"Time elapsed: {inference_iterator.format_dict['elapsed']}s")

if __name__ == "__main__":
    server_address = "127.0.0.1:50051"
    x_test, y_test = load_cifar10()
    asyncio.run(perform_inference(server_address, x_test, y_test))
