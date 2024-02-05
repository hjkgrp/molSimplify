# @file miniGUI.py
#  Defines mini GUI class for drawing molecules from the command line
#
#  Written by Terry Gani for HJK Group
#
#  Dpt of Chemical Engineering, MIT

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QWidget, QGridLayout
from PyQt5.QtGui import QPalette
from molSimplify.Classes.mWidgets import mQMainWindow, mSvgWidget
import sys

# Mini GUI class for drawing molecules from the command line


class miniGUI(QApplication):
    def __init__(self, args):
        QApplication.__init__(self, args)
        p = QPalette()
        p.setColor(QPalette.Background, Qt.white)
        self.window = mQMainWindow()
        self.lwindow = QWidget()
        self.lwindow.setPalette(p)
        self.lgrid = QGridLayout()
        self.lwindow.setLayout(self.lgrid)
        self.lwindow.setWindowTitle('SVG Drawing')

    # def qcloseligs(self):
        self.lwindow.hide()

    def addsvg(self, filename):
        self.svgwidget = mSvgWidget(filename)
        self.lgrid.addWidget(self.svgwidget, 0, 1)
        # self.lwclose = QPushButton('Close')
        # self.lwclose.clicked.connect(self.qcloseligs)
        # self.lgrid.addWidget(self.lwclose,1,0)

    def show(self):
        self.lwindow.show()
        sys.exit(self.exec_())
