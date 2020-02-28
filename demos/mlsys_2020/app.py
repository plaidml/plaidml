import drawSvg.widgets as draw_widgets
import ipywidgets as widgets
from IPython.display import display

import demos.mlsys_2020.core as demo
import plaidml
import plaidml.exec


def run(demo_name):

    my_demo = demo.Demo()

    # Functional widget: Text Box
    op_textbox = lambda x: widgets.Text(value=list(my_demo.edsl_snippets[x].values())[0],
                                        placeholder=list(my_demo.edsl_snippets[x].values())[0],
                                        disabled=False)

    # Functional widget: Dropdown
    op_dropdown = lambda x: widgets.Dropdown(options=list(my_demo.edsl_snippets[x].keys()),
                                             value=list(my_demo.edsl_snippets[x].keys())[0],
                                             disabled=False)

    # Functional widget: Contraction op

    contraction_textbox = op_textbox("Contraction")

    contraction_dropdown = op_dropdown("Contraction")

    contraction_boilerplate = widgets.HTML(
        value=my_demo.boilerplate_html[contraction_dropdown.value])

    contraction_out = widgets.Output()
    with contraction_out:
        display(contraction_boilerplate)

    def contraction_dropdown_handler(change):
        contraction_textbox.value = my_demo.edsl_snippets["Contraction"][change["new"]]
        contraction_textbox.placeholder = contraction_textbox.value
        contraction_boilerplate = widgets.HTML(value=my_demo.boilerplate_html[change["new"]])
        contraction_out.clear_output()
        with contraction_out:
            display(contraction_boilerplate)

    contraction_dropdown.observe(contraction_dropdown_handler, names='value')

    contraction_op_widget = widgets.VBox([
        widgets.HBox([widgets.Label(value="Pick an operation:"), contraction_dropdown]),
        contraction_out,
        widgets.VBox([contraction_textbox,
                      widgets.HTML(value=my_demo.boilerplate_html_footer)],
                     layout=widgets.Layout(margin='0 0 0 50px'))
    ])
    # Functional widget: Elementwise op

    elementwise_textbox = op_textbox("Elementwise")

    elementwise_dropdown = op_dropdown("Elementwise")

    elementwise_boilerplate = widgets.HTML(
        value=my_demo.boilerplate_html[elementwise_dropdown.value])

    elementwise_out = widgets.Output()
    with elementwise_out:
        display(elementwise_boilerplate)

    def elementwise_dropdown_handler(change):
        elementwise_textbox.value = my_demo.edsl_snippets["Elementwise"][change["new"]]
        elementwise_textbox.placeholder = elementwise_textbox.value
        elementwise_boilerplate = widgets.HTML(value=my_demo.boilerplate_html[change["new"]])
        elementwise_out.clear_output()
        with elementwise_out:
            display(elementwise_boilerplate)

    elementwise_dropdown.observe(elementwise_dropdown_handler, names='value')

    elementwise_op_widget = widgets.VBox([
        widgets.HBox([widgets.Label(value="Pick an operation:"), elementwise_dropdown]),
        elementwise_out,
        widgets.VBox([elementwise_textbox,
                      widgets.HTML(value=my_demo.boilerplate_html_footer)],
                     layout=widgets.Layout(margin='0 0 0 50px'))
    ])

    textboxes = [contraction_textbox, elementwise_textbox]
    dropdowns = {'Contraction': contraction_dropdown, 'Elementwise': elementwise_dropdown}

    op_tab_titles = list(my_demo.edsl_snippets.keys())
    op_widgets = [contraction_op_widget, elementwise_op_widget]
    op_tabs = widgets.Tab()
    op_tabs.children = op_widgets
    for i in range(len(op_tabs.children)):
        op_tabs.set_title(i, op_tab_titles[i])

    op_run = widgets.Button(description='Run',
                            disabled=False,
                            button_style='success',
                            tooltip='Compiles and executes your EDSL program',
                            icon='check')

    if demo_name == "mlir":
        mlir_out = widgets.Output()

    def on_run_click(cb):
        op_type = op_tab_titles[op_tabs.selected_index]
        textbox_value = textboxes[op_tabs.selected_index].value
        dropdown_value = dropdowns[op_type].value
        program = my_demo.runtime_handler(op_type, dropdown_value, textbox_value)
        if demo_name == "mlir":
            passes = program.passes
            tile_pass = ""
            affine_pass = ""
            loop_pass = ""
            for elem in passes:
                if elem[0] == "tile":
                    tile_pass = elem[1]
                elif elem[0] == "convert-pxa-to-affine":
                    affine_pass = elem[1]
                elif elem[0] == "lower-affine":
                    loop_pass = elem[1]
            mlir_out.clear_output()
            with mlir_out:
                display(
                    widgets.Textarea(value=str(tile_pass).strip()),
                    widgets.Textarea(value=str(affine_pass).strip()),
                    widgets.Textarea(value=str(loop_pass).strip()),
                )

    op_run.on_click(on_run_click)

    # Interactive user interface (visible on the left side of the display)
    left = widgets.VBox([
        widgets.HTML(value="<h2>Code</h2>"),
        widgets.HTML(
            value=
            "First, select a pre-written EDSL snippet from the list below, or write your own custom operation."
        ),
        widgets.HTML(value="Then, press Run to see the EDSL code in action!"), op_tabs, op_run
    ])

    # Live action animation (visible on the right side of the display)
    def output_anim(arg):
        return my_demo.output_anim

    if demo_name == "mlir":
        right = widgets.VBox([widgets.HTML(value="<h2>Passes</h2>"), mlir_out])
        title = widgets.HTML(
            value='<div style="text-align:center"><h1>MLIR Lowering Demo</h1></div>')

    if demo_name == "edsl":
        right = widgets.VBox([
            widgets.HBox([
                widgets.VBox([
                    widgets.HTML(value="<h2>Input Array: X</h2>"),
                    draw_widgets.DrawingWidget(my_demo.input_x_anim)
                ]),
                widgets.VBox([
                    widgets.HTML(value="<h2>Input Array: Y</h2>"),
                    draw_widgets.DrawingWidget(my_demo.input_y_anim)
                ])
            ]),
            widgets.HTML(value="<h2>Output Array: R</h2>"),
            draw_widgets.AsyncAnimation(1, output_anim, click_pause=False),
            widgets.Label(value="Note: all matrices are shown in row-major order")
        ])
        title = widgets.HTML(value='<div style="text-align:center"><h1>EDSL Demo</h1></div>')

    # Full-screen layout

    hbox_layout = widgets.Layout()
    hbox_layout.width = '100%'
    hbox_layout.justify_content = 'space-around'

    subdemo = widgets.HBox([left, right])
    subdemo.layout = hbox_layout

    full_demo = widgets.VBox([title, subdemo])
    display(full_demo)
