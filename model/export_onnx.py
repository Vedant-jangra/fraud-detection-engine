import xgboost as xgb
from onnxmltools.convert import convert_xgboost
from skl2onnx.common.data_types import FloatTensorType
import onnxruntime as rt
from onnxruntime.quantization import quantize_dynamic, QuantType
import numpy as np
import time
import joblib

import onnx.helper

original_make_attribute = onnx.helper.make_attribute


def patched_make_attribute(key, value, doc_string=None):
    if isinstance(value, list) and len(value) > 0 and isinstance(value[0], bool):
        value = [int(v) for v in value]
    elif isinstance(value, bool):
        value = int(value)
    return original_make_attribute(key, value, doc_string)


onnx.helper.make_attribute = patched_make_attribute


def export_to_onnx(model: xgb.XGBClassifier, output_path: str):
    n_features = len(model.feature_names_in_)
    model.get_booster().feature_names = [f"f{i}" for i in range(n_features)]

    initial_type = [("float_input", FloatTensorType([None, n_features]))]
    onnx_model = convert_xgboost(model, initial_types=initial_type, target_opset=15)

    # Add ai.onnx domain missing from onnxmltools output
    opset = onnx_model.opset_import.add()
    opset.domain = ""
    opset.version = 15

    with open(output_path, "wb") as f:
        f.write(onnx_model.SerializeToString())
    print(f"Exported ONNX model to {output_path} (features: {n_features})")
    return output_path


def quantize_model(onnx_path: str, quantized_path: str):
    quantize_dynamic(
        model_input=onnx_path,
        model_output=quantized_path,
        weight_type=QuantType.QInt8,
    )
    print(f"Quantized model saved to {quantized_path}")


def benchmark_inference(onnx_path: str, n_features: int, n_runs: int = 1000):
    sess = rt.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    dummy_input = np.random.randn(1, n_features).astype(np.float32)

    for _ in range(20):
        sess.run(None, {input_name: dummy_input})

    latencies = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        sess.run(None, {input_name: dummy_input})
        latencies.append((time.perf_counter() - t0) * 1000)

    latencies.sort()
    print(
        f"Inference latency (n={n_runs}):",
        f"p50={latencies[int(n_runs*0.50)]:.2f}ms",
        f"p95={latencies[int(n_runs*0.95)]:.2f}ms",
        f"p99={latencies[int(n_runs*0.99)]:.2f}ms",
    )


if __name__ == "__main__":
    import os

    # Export simulated model (for real-time API)
    if os.path.exists("model/artifacts/xgb_model.pkl"):
        print("=== Exporting Simulated Model (for API) ===")
        model_sim = joblib.load("model/artifacts/xgb_model.pkl")
        export_to_onnx(model_sim, "model/artifacts/model.onnx")
        quantize_model(
            "model/artifacts/model.onnx", "model/artifacts/model_quantized.onnx"
        )

    # Export IEEE-CIS model (offline validation)
    if os.path.exists("model/artifacts/xgb_ieee_cis.pkl"):
        print("\n=== Exporting IEEE-CIS Model ===")
        model_ieee = joblib.load("model/artifacts/xgb_ieee_cis.pkl")
        export_to_onnx(model_ieee, "model/artifacts/ieee_cis_model.onnx")
        quantize_model(
            "model/artifacts/ieee_cis_model.onnx",
            "model/artifacts/ieee_cis_model_quantized.onnx",
        )
