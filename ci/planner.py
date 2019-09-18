import kivy
kivy.require('1.11.1')  # replace with your current kivy version !

from kivy.app import App
from kivy.uix.label import Label
from kivy.uix.gridlayout import GridLayout
import yaml
import util


class Planner(GridLayout):

    def __init__(self, **kwargs):
        super(Planner, self).__init__(**kwargs)
        with open('ci/plan.yml') as file_:
            plan = yaml.safe_load(file_)
        self.cols = 3
        variants = []
        for variant in plan['VARIANTS'].keys():
            self.add_widget(Label(text=variant, col=1))

        tests = []


class MyApp(App):

    def build(self):
        return Planner()


if __name__ == '__main__':
    MyApp().run()
