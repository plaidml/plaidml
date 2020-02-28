import re

import drawSvg as draw
import ipywidgets as widgets
import numpy as np
import plaidml
import plaidml.exec

from IPython.display import display
from plaidml.edsl import *
import demos.mlsys_2020.style.custom_html as html


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
        self.boilerplate_inputs = {"Matrix Multiplication": ['X', 'Y'], "Sum Over Axis": ['X']}
        self.boilerplate_edsl = {
            "Matrix Multiplication":
                """
def edsl_program(X, Y):
        I, J, K = TensorDims(3)
        i, j, k = TensorIndexes(3)
        X.bind_dims(I, K)
        Y.bind_dims(K, J)
        R = TensorOutput(I, J)
        {}
        return R
""",
            "Sum Over Axis":
                """
def edsl_program(X):
        I, J = TensorDims(2)
        i, j = TensorIndexes(2)
        X.bind_dims(I, J)
        R = TensorOutput(I)
        {}
        return R
"""
        }
        self.boilerplate_html = {
            "Generic_One_Input_Eltwise":
                f"""
{html.code_open}
{html.code_directive}def{html.font_close} {html.code_function_name}edsl_program{html.font_close}
({html.code_input_param}X{html.font_close}):{html.br}
{html.code_close}
""",
            "Generic_Two_Input_Eltwise":
                f"""
{html.code_open}
{html.code_directive}def{html.font_close} {html.code_function_name}edsl_program{html.font_close}
({html.code_input_param}X{html.font_close}, {html.code_input_param}Y{html.font_close}):{html.br}
{html.code_close}
"""
        }
        self.boilerplate_html_footer = f"""
{html.code_directive}return{html.font_close} {html.code_output_param}R{html.font_close}
"""
        self.boilerplate_html["Square Root"] = self.boilerplate_html["Generic_One_Input_Eltwise"]
        self.boilerplate_html['Sum'] = self.boilerplate_html["Generic_Two_Input_Eltwise"]
        self.boilerplate_html['Multiply'] = self.boilerplate_html["Generic_Two_Input_Eltwise"]
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
                string_output = str(vals[dims[1] - 1 - y][x])
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

    def draw_line(self, dims, vals, grid_size=10, color='#ffffff'):
        anim = draw.Drawing(330, 120, origin=(0, 0))
        for x in range(dims[0]):
            group = draw.Group()
            group.draw(
                draw.Rectangle(
                    100 * x + grid_size,  # origin x coords
                    grid_size,  # origin y coords
                    100,  # grid width
                    100,  # grid height
                    stroke_width=grid_size,  # outline size
                    stroke='black',  # outline color
                    fill=color))  # fill color
            string_output = str(vals[x])
            font_size = 50 / len(string_output) + 25
            group.draw(
                draw.Text(
                    string_output,
                    font_size,
                    100 * x + grid_size + font_size / 3 +
                    2 * len(string_output),  # origin x coords
                    grid_size + 2250 / (font_size + 20),  # origin y coords
                    center=0))
            anim.append(group)
        return anim

    def runtime_handler(self, op_type, op_name, textbox_value):
        X = Placeholder(plaidml.DType.INT32, [3, 3])
        Y = Placeholder(plaidml.DType.INT32, [3, 3])
        edsl_context = globals()
        if op_type == "Contraction":
            exec(self.boilerplate_edsl[op_name].format(textbox_value), edsl_context)
        else:
            exec("""
def edsl_program(X, Y):
        {}
        return R
""".format(textbox_value), edsl_context)
        edsl_program = edsl_context['edsl_program']
        if len(self.boilerplate_inputs[op_name]) == 1:
            if 'X' in self.boilerplate_inputs[op_name]:
                R = edsl_program(X)
            if 'Y' in self.boilerplate_inputs[op_name]:
                R = edsl_program(Y)
        else:
            R = edsl_program(X, Y)
        program = Program('edsl_program', [R], debug=True)
        for p in program.passes:
            name = p[0]
            print('\n')
            print(name)
            text = widgets.Textarea(value=''.join(p),
                                    placeholder='Passes',
                                    disabled=False,
                                    layout=widgets.Layout(
                                        height='100%',
                                        width='auto',
                                    ))
            box = widgets.VBox([text], layout={'height': '350px'})
            #display(box)

        # Create the binder and the executable so that the program can run.
        binder = plaidml.exec.Binder(program)
        executable = binder.compile()
        if 'X' in self.boilerplate_inputs[op_name]:
            binder.input(X).copy_from_ndarray(self.input_x_vals)
        if 'Y' in self.boilerplate_inputs[op_name]:
            binder.input(Y).copy_from_ndarray(self.input_y_vals)
        executable.run()

        # The code has been run, so the values of each tensor have been bound at this point.
        self.output_vals = binder.output(R).as_ndarray()

        if self.output_vals.ndim == 1:
            self.output_anim = self.draw_line(self.output_vals.shape,
                                              self.output_vals,
                                              color='#aaddff')
        else:
            self.output_anim = self.draw_grid(self.output_vals.shape,
                                              self.output_vals,
                                              color='#aaddff')
