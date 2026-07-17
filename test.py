import onnx

m = onnx.load("onnx/Any2Full_vits.onnx")

for n in m.graph.node:
    if n.op_type == "Range":
        print(n.name)
        print(n.input)
        print(n.output)
        print()