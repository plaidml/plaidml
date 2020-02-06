# Begin with some imports
import numpy as np
import drawSvg as draw
from drawSvg.widgets import DrawingWidget, AsyncAnimation
import ipywidgets as widgets

import plaidml
import plaidml.exec
from plaidml.edsl import *


class Demo:

    def __init__(self):
        self.edsl_snippets = {
            "Contraction": {
                "Matrix Multiplication": "R[i, j] += X[i, k] * Y[k, j]",
                "Sum Over Axis": "R[i] += X[i, j]"
            },
            "Elementwise": {
                "Sum": "R = X + Y",
                "Multiply": "R = X * Y",
                "Square Root": "R = sqrt(X)"
            }
        }
        # Placeholders
        self.input_x_vals = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        self.input_y_vals = np.array([[9, 8, 7], [6, 5, 4], [3, 2, 1]])
        self.output_vals = np.full([3, 3], np.nan)
        self.input_x_anim = self.draw_grid(self.input_x_vals.shape,
                                           self.input_x_vals,
                                           color='#ffddaa')
        self.input_y_anim = self.draw_grid(self.input_y_vals.shape,
                                           self.input_y_vals,
                                           color='#ffddff')
        self.output_anim = self.draw_grid(self.output_vals.shape,
                                          self.output_vals,
                                          color='#ffffff')

    # Drawing helper: this can be used for inputs, outputs, etc.
    # Eventually, this will include animation logic too.
    def draw_grid(self, dims, vals, grid_size=10, color='#ffffff'):
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

    def runtime_handler(self, op_type, textbox_value):
        X = Placeholder(plaidml.DType.INT32, [3, 3])
        Y = Placeholder(plaidml.DType.INT32, [3, 3])
        edsl_context = globals()
        exec(
            """
def edsl_program(X, Y):
        I, J, K = TensorDims(3)
        i, j, k = TensorIndexes(3)
        X.bind_dims(I, K)
        Y.bind_dims(K, J)
        R = TensorOutput(I, J)
        {}
        return R
""".format(textbox_value), edsl_context)
        edsl_program = edsl_context['edsl_program']
        print('edsl_program:', edsl_program)
        R = edsl_program(X, Y)
        print('R:', R)
        #R = edsl_code(X, Y)
        #R = X + Y
        #print(R)
        program = Program('edsl_program', [R])

        # Create the binder and the executable so that the program can run.
        binder = plaidml.exec.Binder(program)
        executable = binder.compile()
        binder.input(X).copy_from_ndarray(self.input_x_vals)
        binder.input(Y).copy_from_ndarray(self.input_y_vals)
        executable.run()

        # The code has been run, so the values of each tensor have been bound at this point.
        self.output_vals = binder.output(R).as_ndarray()

        self.output_anim = self.draw_grid(self.output_vals.shape,
                                          self.output_vals,
                                          color='#aaddff')
