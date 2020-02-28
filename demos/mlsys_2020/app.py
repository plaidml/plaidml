import drawSvg.widgets as draw_widgets
import ipywidgets as widgets
from IPython.display import display

import demos.mlsys_2020.core as demo
import plaidml
import plaidml.exec


def run():

    myDemo = demo.Demo()

    # Functional widget: Text Box
    op_textbox = lambda x: widgets.Text(value=list(myDemo.edsl_snippets[x].values())[0],
                                        placeholder=list(myDemo.edsl_snippets[x].values())[0],
                                        disabled=False)

    # Functional widget: Dropdown
    op_dropdown = lambda x: widgets.Dropdown(options=list(myDemo.edsl_snippets[x].keys()),
                                             value=list(myDemo.edsl_snippets[x].keys())[0],
                                             disabled=False)

    # Functional widget: Contraction op

    contraction_textbox = op_textbox("Contraction")

    contraction_dropdown = op_dropdown("Contraction")

    def contraction_dropdown_handler(change):
        contraction_textbox.value = myDemo.edsl_snippets["Contraction"][change["new"]]
        contraction_textbox.placeholder = contraction_textbox.value

    contraction_dropdown.observe(contraction_dropdown_handler, names='value')

    contraction_op_widget = widgets.VBox([
        widgets.HBox([widgets.Label(value="Pick an operation:"), contraction_dropdown]),
        widgets.HTML(value="<code> \
            <font class=\"code-directive\">def</font> <font class=\"code-function-name\">edsl_program</font>(<font class=\"code-input-param\">X</font>, <font class=\"code-input-param\">Y</font>): <br>\
            </code>"),
        widgets.VBox([
            widgets.HTML(value="<code> \
                    <font class=\"code-tensor-dim\">I</font>, <font class=\"code-tensor-dim\">J</font>, <font class=\"code-tensor-dim\">K</font> = <font class=\"code-directive\">TensorDims</font>(<font class=\"code-numeric\">3</font>) <br>\
                    <font class=\"code-tensor-index\">i</font>, <font class=\"code-tensor-index\">j</font>, <font class=\"code-tensor-index\">k</font> = <font class=\"code-directive\">TensorIndexes</font>(<font class=\"code-numeric\">3</font>) <br>\
                    <font class=\"code-input-param\">X</font>.<font class=\"code-directive\">bind_dims</font>(<font class=\"code-tensor-dim\">I</font>, <font class=\"code-tensor-dim\">K</font>) <br>\
                    <font class=\"code-input-param\">Y</font>.<font class=\"code-directive\">bind_dims</font>(<font class=\"code-tensor-dim\">K</font>, <font class=\"code-tensor-dim\">J</font>) <br>\
                    <font class=\"code-output-param\">R</font> = <font class=\"code-directive\">TensorOutput</font>(<font class=\"code-tensor-dim\">I</font>, <font class=\"code-tensor-dim\">J</font>) <br>\
                    </code>"),
            contraction_textbox,
            widgets.HTML(value="<code> \
                    <font class=\"code-directive\">return</font> <font class=\"code-output-param\">R</font> <br>\
                    </code>"),
        ],
                     layout=widgets.Layout(margin='0 0 0 5%'))
    ])
    # Functional widget: Elementwise op

    elementwise_textbox = op_textbox("Elementwise")

    elementwise_dropdown = op_dropdown("Elementwise")

    def elementwise_dropdown_handler(change):
        elementwise_textbox.value = myDemo.edsl_snippets["Elementwise"][change["new"]]
        elementwise_textbox.placeholder = elementwise_textbox.value

    elementwise_dropdown.observe(elementwise_dropdown_handler, names='value')

    elementwise_op_widget = widgets.VBox([
        widgets.HBox([widgets.Label(value="Pick an operation:"), elementwise_dropdown]),
        widgets.HTML(value="<code> \
            def edsl_program(X, Y): <br>\
            </code>"),
        widgets.VBox([
            elementwise_textbox,
            widgets.HTML(value="<code> \
                    return R <br>\
                    </code>")
        ],
                     layout=widgets.Layout(margin='0 0 0 5%'))
    ])

    textboxes = [contraction_textbox, elementwise_textbox]
    dropdowns = {'Contraction': contraction_dropdown, 'Elementwise': elementwise_dropdown}

    op_tab_titles = list(myDemo.edsl_snippets.keys())
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

    def on_run_click(cb):
        op_type = op_tab_titles[op_tabs.selected_index]
        textbox_value = textboxes[op_tabs.selected_index].value
        dropdown_value = dropdowns[op_type].value
        myDemo.runtime_handler(op_type, dropdown_value, textbox_value)

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
        return myDemo.output_anim

    right = widgets.VBox([
        widgets.HTML(value="<h2>Inputs</h2>"),
        widgets.HBox([
            draw_widgets.DrawingWidget(myDemo.input_x_anim),
            draw_widgets.DrawingWidget(myDemo.input_y_anim)
        ]),
        widgets.HTML(value="<h2>Result</h2>"),
        draw_widgets.AsyncAnimation(1, output_anim, click_pause=False),
    ])

    # Full-screen layout

    hbox_layout = widgets.Layout()
    hbox_layout.width = '100%'
    hbox_layout.justify_content = 'space-around'

    # do this in a flexbox-friendlier way
    edsl_title = widgets.HTML(value='<div style="text-align:center"><h1>EDSL Demo</h1></div>')

    edsl_subdemo = widgets.HBox([left, right])
    edsl_subdemo.layout = hbox_layout

    edsl_demo = widgets.VBox([edsl_title, edsl_subdemo])
    display(edsl_demo)
