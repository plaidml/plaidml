import kivy
kivy.require('1.11.1')  # replace with your current kivy version !

from kivy.app import App
from kivy.uix.label import Label
from kivy.uix.button import Button
from functools import partial

from kivy.uix.boxlayout import BoxLayout
import yaml
import util


class Planner(BoxLayout):

    def __init__(self, **kwargs):
        super(Planner, self).__init__(**kwargs)

        with open('ci/plan.yml') as file_:
            plan = yaml.safe_load(file_)

        def callback(instance):
            print("button pressed: " + instance.text)
            for pkey, platform in plan['PLATFORMS'].items():
                print(pkey, platform)
                pinfo = plan['PLATFORMS'][pkey]
                variant = pinfo['variant']
                displayed = []
                if instance.text in variant:
                    self.platform = Button(text=pkey)
                    self.add_widget(self.platform)
                    displayed.append(self.platform)
            print(*displayed)

            # for platform in plan['PLATFORMS'].keys():
            #     self.add_widget(Button(text=platform))

        self.add_widget(self.label)

        variants = []
        for variant in plan['VARIANTS'].keys():
            self.variant = Button(text=variant)
            self.variant.bind(on_press=callback)
            self.add_widget(self.variant)
        # platforms = []
        # for platform in plan['PLATFORMS'].keys():
        #     self.add_widget(Button(text=platform))
        #suites = []
        #for suite in plan['SUITES'].keys():
        #self.add_widget(Button(text=suite))


class PlannerApp(App):

    def build(self):
        return Planner()


if __name__ == '__main__':
    layout = BoxLayout(padding=10)
    label = Label(text="PlaidML")
    PlannerApp().run()
