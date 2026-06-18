import xgboost as xgb
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import onnxruntime as rt
from onnxruntime.quantization import quantize_dynamic, QuantType
import numpy as np
import time
import joblib

N_FEATURES = 16

def export_to_onnx(model: xgb.XGBClassifier, output_path: str):
    initial_type = [('float_input', FloatTensorType([None, N_FEATURES]))]
    onnx_model = convert_sklearn(model, initial_types=initial_type, target_opset=18)
    with open(output_path, 'wb') as f:
        f.write(onnx_model.SerializeToString())
    print(f'Exported ONNX model to {output_path}')
    return output_path

def quantize_model(onnx_path: str, quantized_path: str):
    quantize_dynamic(
        model_input=onnx_path,
        model_output=quantized_path,
        weight_type=QuantType.QInt8,
    )
    print(f'Quantized model saved to {quantized_path}')

def benchmark_inference(onnx_path: str, n_runs: int = 1000):
    sess = rt.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    input_name = sess.get_inputs()[0].name
    dummy_input = np.random.randn(1, N_FEATURES).astype(np.float32)
    
    for _ in range(20):
        sess.run(None, {input_name: dummy_input})
        
    latencies = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        sess.run(None, {input_name: dummy_input})
        latencies.append((time.perf_counter() - t0) * 1000)
        
    latencies.sort()
    print(f'Inference latency (n={n_runs}):',
          f'p50={latencies[int(n_runs*0.50)]:.2f}ms',
          f'p95={latencies[int(n_runs*0.95)]:.2f}ms',
          f'p99={latencies[int(n_runs*0.99)]:.2f}ms')

if __name__ == '__main__':
    model = joblib.load('model/artifacts/xgb_model.pkl')
    
    export_to_onnx(model, 'model/artifacts/model.onnx')
    quantize_model('model/artifacts/model.onnx', 'model/artifacts/model_quantized.onnx')
    
    print('\n=== Original ONNX Benchmark ===')
    benchmark_inference('model/artifacts/model.onnx')
    
    print('\n=== Quantized ONNX Benchmark ===')
    benchmark_inference('model/artifacts/model_quantized.onnx')
