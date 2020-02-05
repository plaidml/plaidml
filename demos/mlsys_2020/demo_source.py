# Begin with some imports
import numpy as np
import drawSvg as draw
from drawSvg.widgets import DrawingWidget
import ipywidgets as widgets

import plaidml
import plaidml.exec
from plaidml.edsl import *


class Demo:

    edsl_snippets = {
        "Matrix Multiplication": "R[i, j] += X[i, k] * Y[k, j]",
        "Sum Over Axis": "R[i] += X[i, j]",
        "Elementwise Sum": "R = X + Y",
        "Elementwise Multiply": "R = X * Y",
        "Elementwise Square Root": "R = sqrt(X)",
    }

    # Drawing helper: this can be used for inputs, outputs, etc.
    # Eventually, this will include animation logic too.
    def draw_grid(dims, vals, grid_size=10, color='#ffffff'):
        anim = draw.Drawing(330, 330, origin=(0, 0))
        for x in range(dims[0]):
            for y in range(dims[1]):
                group = draw.Group()
                group.draw(
                    draw.Rectangle(
                        100 * x + grid_size,  # origin x coords
                        100 * y + grid_size,  # origin y coords
                        100,  # grid width
                        100,  # grid height
                        stroke_width=grid_size,  # outline size
                        stroke='black',  # outline color
                        fill=color))  # fill color
                string_output = str(vals[x][y])
                font_size = 50 / len(string_output) + 25
                group.draw(
                    draw.Text(
                        string_output,
                        font_size,
                        100 * x + grid_size + font_size / 3 +
                        2 * len(string_output),  # origin x coords
                        100 * y + grid_size + 2250 / (font_size + 20),  # origin y coords
                        center=0))
                anim.append(group)
        return anim

    # The program that will be executed during the demo.
    # Eventually, this will have callbacks enabled for custom operations.
    def edsl_program(X, Y):
        I, J, K = TensorDims(3)
        i, j, k = TensorIndexes(3)
        X.bind_dims(I, K)
        Y.bind_dims(K, J)
        R = TensorOutput(I, J)
        R[i, j] += X[i, k] * Y[k, j]
        return R

    # Hardcoded placeholders for now
    # Eventually, this will be a callback to edsl_program that is triggered by the Run button.
    A = Placeholder(plaidml.DType.INT32, [3, 3])
    B = Placeholder(plaidml.DType.INT32, [3, 3])
    O = edsl_program(A, B)
    program = Program('edsl_program', [O])

    # The dimensions of each tensor are bound at this point now that the Program is created.
    # The values of each tensor are still placeholders at this point.
    input_a_dims = program.inputs[0].shape.int_dims
    input_b_dims = program.inputs[1].shape.int_dims
    output_dims = program.outputs[0].shape.int_dims

    # Create the binder and the executable so that the program can run.
    binder = plaidml.exec.Binder(program)
    executable = binder.compile()
    binder.input(A).copy_from_ndarray(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))
    binder.input(B).copy_from_ndarray(np.array([[9, 8, 7], [6, 5, 4], [3, 2, 1]]))
    executable.run()
    # The code has been run, so the values of each tensor have been bound at this point.
    input_a_vals = binder.input(A).as_ndarray()
    input_b_vals = binder.input(B).as_ndarray()
    output_vals = binder.output(O).as_ndarray()

    # Animations: create them in one place for now.
    # Output animations are not available until the executable is complete.
    input_a_anim = draw_grid(input_a_dims, input_a_vals, color='#ffddaa')
    input_b_anim = draw_grid(input_b_dims, input_b_vals, color='#ffddff')
    output_anim = draw_grid(output_dims, output_vals, color='#aaddff')
