"""Entry point for the MOT image analysis application.

Supports two modes:
- Standalone: Runs as a desktop PyQt5 application with manual HDF5 loading.
- ARTIQ applet: Runs inside the ARTIQ dashboard, receiving live camera
  frames via dataset subscriptions.
"""

import sys
from PyQt5 import QtWidgets
from artiq.applets.simple import SimpleApplet
from model import FitModel
from view import FitView
from controller import FitController


def main():
    """Launches the application in standalone or ARTIQ applet mode.

    If command-line arguments are present, assumes ARTIQ applet mode
    and registers the view as an applet subscribing to camera datasets.
    Otherwise, opens a standalone window with manual file loading.
    """
    if len(sys.argv) > 1:

        class AppletFitView(FitView):
            """FitView subclass that auto-creates model and controller."""

            def __init__(self, args, req):
                super().__init__(args, req)
                self.model = FitModel()
                self.controller = FitController(self.model, self)

        applet = SimpleApplet(AppletFitView)
        applet.add_dataset("img", "Camera image")
        applet.run()
    else:
        app = QtWidgets.QApplication(sys.argv)
        model = FitModel()
        view = FitView(None, None)
        controller = FitController(model, view)
        view.setWindowTitle("MOT Image Analysis")
        view.show()
        sys.exit(app.exec_())


if __name__ == "__main__":
    main()