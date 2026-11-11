"""Export the int8 CNN as plain integer tables for the hand-written C++ network.

The quantised ONNX model still needs ONNX Runtime to execute. This script
extracts everything a runtime-free, integer-only implementation needs (the
kind of arithmetic an FPGA would perform) and writes it to
artifacts/cnn.fixed.bin, together with ONNX Runtime's own outputs on the
benchmark samples so that the C++ side can check itself against them.

Two things make the exported network hardware-friendly:

  * Every layer is a 1-D convolution. A dense layer on a flattened
    (channels, length) tensor is the same as a convolution whose kernel spans
    the whole length, so the two dense layers of the head are exported that way.
  * Rescaling between layers uses an integer multiply and a right shift
    instead of the float multiply ONNX Runtime uses.

Usage:
    python -m lob.export_fixed
"""

from __future__ import annotations

import argparse
import math
import struct
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper

from lob.export import INPUT_NAME, run_onnx

# First four bytes of the output file: "LOBF" when written little-endian.
FIXED_MAGIC = 0x46424F4C


# Turn a real-valued scale factor into an integer multiplier and a right shift.
def quantize_multiplier(real: float) -> tuple[int, int]:
    """Return (multiplier, shift) such that x * real ~= (x * multiplier) >> shift.

    The multiplier is a 31-bit integer holding the mantissa of `real`, so the
    approximation is accurate to about one part in 2^30.
    """
    if real <= 0:
        raise ValueError(f"scale factor must be positive, got {real}")
    mantissa, exponent = math.frexp(real)  # real = mantissa * 2**exponent, mantissa in [0.5, 1)
    multiplier = round(mantissa * (1 << 31))
    if multiplier == 1 << 31:  # mantissa rounded up to 1.0: renormalise
        multiplier //= 2
        exponent += 1
    return multiplier, 31 - exponent


# Lookup helpers over an ONNX graph.
class _Graph:
    """Constant tensors plus 'who produces / who consumes' maps for a graph."""

    def __init__(self, model: onnx.ModelProto) -> None:
        self.nodes = list(model.graph.node)
        self.constants = {t.name: numpy_helper.to_array(t) for t in model.graph.initializer}
        self.producer = {out: node for node in self.nodes for out in node.output}
        self.consumers: dict[str, list[onnx.NodeProto]] = {}
        for node in self.nodes:
            for name in node.input:
                self.consumers.setdefault(name, []).append(node)

    # The single node that reads tensor `name`.
    def consumer(self, name: str) -> onnx.NodeProto:
        (node,) = self.consumers[name]
        return node

    # Scale and zero point of a QuantizeLinear / DequantizeLinear node.
    def scale_and_zero_point(self, node: onnx.NodeProto) -> tuple[np.ndarray, np.ndarray]:
        return self.constants[node.input[1]], self.constants[node.input[2]]

    # Integer values and scale of a weight or bias stored behind a DequantizeLinear.
    def quantized_constant(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        dequantize = self.producer[name]
        scale, zero_point = self.scale_and_zero_point(dequantize)
        if np.any(zero_point != 0):
            raise ValueError(f"{name}: expected symmetric quantisation (zero point 0)")
        return self.constants[dequantize.input[0]], scale


# Convert one Conv or Gemm node, with the rescaling that follows it, to integer tables.
def _convert_layer(graph: _Graph, node: onnx.NodeProto, in_length: int) -> dict:
    """Return the integer description of one layer.

    The quantised graph around every layer looks like

        input (int8) -> Conv/Gemm -> requantise -> [ReLU -> requantise] -> next layer

    and each "requantise" becomes an integer multiply-and-shift.
    """
    # Scale and zero point of the layer input.
    in_scale, in_zero_point = graph.scale_and_zero_point(graph.producer[node.input[0]])
    weights, weight_scale = graph.quantized_constant(node.input[1])
    bias, _ = graph.quantized_constant(node.input[2])

    if node.op_type == "Conv":
        stride = next(a.ints[0] for a in node.attribute if a.name == "strides")
    else:
        # Dense layer on a flattened (channels, length) tensor: view its weight
        # matrix (out, channels * length) as a kernel that spans the whole input.
        weights = weights.reshape(weights.shape[0], -1, in_length)
        stride = 1
    out_channels, in_channels, kernel = weights.shape

    # Fold the input zero point into the bias:
    #   sum((x - zp) * w) + b  ==  sum(x * w) + (b - zp * sum(w))
    # so the hardware never has to subtract the zero point from its inputs.
    weights = weights.astype(np.int64)
    folded_bias = bias.astype(np.int64) - int(in_zero_point) * weights.sum(axis=(1, 2))

    # First requantisation: accumulator (scale in_scale * weight_scale) -> int8.
    quantize = graph.consumer(node.output[0])
    out_scale, out_zero_point = graph.scale_and_zero_point(quantize)
    multipliers = [
        quantize_multiplier(float(in_scale) * float(s) / float(out_scale)) for s in weight_scale
    ]
    layer = {
        "in_channels": in_channels,
        "out_channels": out_channels,
        "kernel": kernel,
        "stride": stride,
        # Stored as [out][kernel][in] so the inner loop walks memory in time order.
        "weights": weights.transpose(0, 2, 1).astype(np.int8),
        "bias": folded_bias.astype(np.int32),
        "multiplier": np.array([m for m, _ in multipliers], dtype=np.int32),
        "shift": np.array([s for _, s in multipliers], dtype=np.int32),
        "zero_point": int(out_zero_point),
        "relu": None,
        "out_length": (in_length - kernel) // stride + 1,
    }

    # Optional ReLU followed by a second requantisation to a new scale.
    after = graph.consumers.get(graph.consumer(quantize.output[0]).output[0], [])
    if after and after[0].op_type == "Relu":
        requantize = graph.consumer(after[0].output[0])
        if requantize.op_type == "Reshape":  # flatten before the dense head
            requantize = graph.consumer(requantize.output[0])
        relu_scale, relu_zero_point = graph.scale_and_zero_point(requantize)
        multiplier, shift = quantize_multiplier(float(out_scale) / float(relu_scale))
        layer["relu"] = {
            "multiplier": multiplier,
            "shift": shift,
            "zero_point": int(relu_zero_point),
        }
    return layer


# Recover the int8 logits ONNX Runtime produced from its float output.
def reference_logits(model_path: Path, graph: _Graph, windows: np.ndarray) -> np.ndarray:
    """ONNX Runtime's int8 logits for `windows`, used as test vectors in C++.

    The graph ends with a DequantizeLinear, so its float output is exactly
    (q - zero_point) * scale and the integer q can be recovered without error.
    """
    scale, zero_point = graph.scale_and_zero_point(graph.producer["logits"])
    logits = run_onnx(model_path, windows)
    return (np.rint(logits / scale) + zero_point).astype(np.int8)


# Read the benchmark sample windows back from artifacts/samples.bin.
def read_sample_windows(path: Path) -> np.ndarray:
    blob = path.read_bytes()
    _, n, window, n_features = struct.unpack("<4I", blob[:16])
    count = n * window * n_features
    return np.frombuffer(blob, "<f4", count, offset=16).reshape(n, window, n_features)


# Serialise everything in the layout read by cpp/include/lob/fixed_net.hpp.
def write_fixed_model(
    path: Path,
    window: int,
    mean: np.ndarray,
    std: np.ndarray,
    input_scale: float,
    input_zero_point: int,
    layers: list[dict],
    reference: np.ndarray,
) -> None:
    """Write the integer network and its test vectors (all values little-endian).

    Layout:
        uint32  magic, n_layers, window, n_features
        float32 mean[n_features], std[n_features]       input normalisation
        float32 input_scale;  int32 input_zero_point    float -> int8 boundary
        per layer:
            uint32 in_channels, out_channels, kernel, stride, has_relu
            int8   weights[out_channels * kernel * in_channels]
            int32  bias[out_channels]
            int32  multiplier[out_channels], shift[out_channels], zero_point
            int32  relu_multiplier, relu_shift, relu_zero_point   (if has_relu)
        uint32  n_reference
        int8    reference_logits[n_reference * 3]
    """
    with path.open("wb") as f:
        f.write(struct.pack("<4I", FIXED_MAGIC, len(layers), window, len(mean)))
        f.write(mean.astype("<f4").tobytes())
        f.write(std.astype("<f4").tobytes())
        f.write(struct.pack("<fi", input_scale, input_zero_point))
        for layer in layers:
            relu = layer["relu"]
            f.write(
                struct.pack(
                    "<5I",
                    layer["in_channels"],
                    layer["out_channels"],
                    layer["kernel"],
                    layer["stride"],
                    int(relu is not None),
                )
            )
            f.write(layer["weights"].tobytes())
            f.write(layer["bias"].astype("<i4").tobytes())
            f.write(layer["multiplier"].astype("<i4").tobytes())
            f.write(layer["shift"].astype("<i4").tobytes())
            f.write(struct.pack("<i", layer["zero_point"]))
            if relu is not None:
                f.write(struct.pack("<3i", relu["multiplier"], relu["shift"], relu["zero_point"]))
        f.write(struct.pack("<I", len(reference)))
        f.write(reference.tobytes())


# Command-line entry point.
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    args = parser.parse_args()

    model_path = args.artifacts_dir / "cnn.int8.onnx"
    graph = _Graph(onnx.load(model_path))

    # Input side: normalisation constants and the float -> int8 boundary.
    mean, std = graph.constants["norm.mean"], graph.constants["norm.std"]
    first_layer = next(n for n in graph.nodes if n.op_type == "Conv")
    input_quantize = graph.producer[graph.producer[first_layer.input[0]].input[0]]
    input_scale, input_zero_point = graph.scale_and_zero_point(input_quantize)

    # Convert the layers in execution order, tracking the length of the time axis.
    windows = read_sample_windows(args.artifacts_dir / "samples.bin")
    length = windows.shape[1]
    layers = []
    for node in graph.nodes:
        if node.op_type in ("Conv", "Gemm"):
            layers.append(_convert_layer(graph, node, length))
            length = layers[-1]["out_length"]

    out_path = args.artifacts_dir / "cnn.fixed.bin"
    write_fixed_model(
        out_path,
        windows.shape[1],
        mean,
        std,
        float(input_scale),
        int(input_zero_point),
        layers,
        reference_logits(model_path, graph, windows),
    )

    # Report the size of the arithmetic: one MAC is one multiply-accumulate.
    total_macs = 0
    for i, layer in enumerate(layers):
        macs = layer["out_length"] * layer["out_channels"] * layer["kernel"] * layer["in_channels"]
        total_macs += macs
        print(
            f"layer {i}: {layer['in_channels']:3d} -> {layer['out_channels']:3d} channels, "
            f"kernel {layer['kernel']:2d}, stride {layer['stride']}, "
            f"output length {layer['out_length']:2d}, {macs:9,d} MACs"
        )
    print(f"total: {total_macs:,d} MACs per prediction (input: {INPUT_NAME})")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
