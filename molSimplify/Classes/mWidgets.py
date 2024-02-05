# @file mWidgets.py
#  Defines auxiliary classes for building GUI.
#
#  Written by Tim Ioannidis for HJK Group
#
#  Dpt of Chemical Engineering, MIT
try:
    from PyQt5.QtWidgets import (QApplication, QDesktopWidget, QMainWindow,
                                 QPushButton, QSizePolicy, QCheckBox,
                                 QComboBox, QSlider, QDialog, QMessageBox,
                                 QTextEdit, QLineEdit, QLabel, QSpinBox,
                                 QCompleter)
    from PyQt5.QtGui import QFont, QFontMetrics, QPixmap
    from PyQt5.QtCore import Qt, QSortFilterProxyModel
    from PyQt5.QtSvg import QSvgWidget
except ImportError:
    pass
import imghdr
import struct


# py 2/3 interop
import sys
if sys.version_info[0] >= 3:
    str = str

# Get screen size


def getscreensize():
    screenShape = QDesktopWidget().screenGeometry()
    return [screenShape.width(), screenShape.height()]

# Center main widget on screen


def center(self):
    frameGm = self.frameGeometry()
    screen = QApplication.desktop().screenNumber(
        QApplication.desktop().cursor().pos())
    centerPoint = QApplication.desktop().screenGeometry(screen).center()
    frameGm.moveCenter(centerPoint)
    self.move(frameGm.topLeft())

# Relative resize


def relresize(self, parent, scale):
    parentgeom = parent.frameGeometry()
    width = parentgeom.width()*scale
    height = parentgeom.height()*scale
    xmarg = 0.5*(1.0-scale)*parentgeom.width()
    ymarg = 0.5*(1.0-scale)*parentgeom.height()
    self.setGeometry(xmarg, ymarg, width, height)

# GUI main window class


class mQMainWindow(QMainWindow):
    def __init__(self):
        super(mQMainWindow, self).__init__()

    def closeEvent(self, event):
        sys.exit()

# GUI push button class


class mQPushButton(QPushButton):
    def __init__(self, txt, ctip, fontsize):
        super(mQPushButton, self).__init__()
        self.setText(txt)
        self.setToolTip(ctip)
        f = QFont("Helvetica", fontsize)
        self.setFont(f)
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        # self.setMouseTracking(True)
        self.show()
    # def enterEvent(self,event):
    #    self.setStyleSheet("background-color:#F3BEAC;")
    # def leaveEvent(self,event):
    #    self.setStyleSheet("background-color:;")

# GUI checkbox class


class mQCheckBox(QCheckBox):
    # margins and scale are fractional
    def __init__(self, txt, ctip, fontsize):
        super(mQCheckBox, self).__init__()
        self.state = False
        self.setText(txt)
        f = QFont("Helvetica", fontsize)
        self.stateChanged.connect(self.changestate)
        self.setFont(f)
        self.setToolTip(ctip)
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)

    def changestate(self):
        if self.isChecked():
            self.state = False
        else:
            self.state = True

    def getState(self):
        if self.isChecked():
            return 1
        else:
            return 0

# GUI dropdown box class


class mQComboBox(QComboBox):
    # margins and scale are fractional
    def __init__(self, txt, ctip, fontsize):
        super(mQComboBox, self).__init__()
        self.state = 0
        for t in txt:
            self.addItem(t)
        f = QFont("Helvetica", fontsize)
        self.setToolTip(ctip)
        self.setFont(f)
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)

    def getState(self):
        return self.currentIndex()

    def getText(self):
        return self.currentText()

# GUI slider class


class mQSlider(QSlider):
    # margins and scale are fractional
    def __init__(self, ctip):
        super(mQSlider, self).__init__()
        self.setOrientation(Qt.Horizontal)
        self.setRange(0, 100)
        self.setValue(0)
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)

# GUI box folder class


class qBoxFolder(QDialog):
    # constructor needs window, marginW, marginH and Font size
    # margins and scale are fractional
    def __init__(self, window, toptxt, txt):
        super(qBoxFolder, self).__init__()
        self.setParent(window)
        self.msgBox = QMessageBox()
        self.msgBox.setIcon(QMessageBox.Question)
        self.msgBox.setWindowTitle(toptxt)
        self.msgBox.setText(txt)
        self.msgBox.addButton(QPushButton('Keep both'), QMessageBox.YesRole)
        self.msgBox.addButton(QPushButton('Replace'), QMessageBox.NoRole)
        self.msgBox.addButton(QPushButton('Skip'), QMessageBox.RejectRole)

    def getaction(self):
        ret = self.msgBox.exec_()
        if ret == 1:
            return 'replace'
        elif ret == 2:
            return 'skip'
        elif ret == 0:
            return 'keep'
        else:
            return False

# GUI popup boxes class


class mQDialogInf(QDialog):
    def __init__(self, toptxt, txt):
        super(mQDialogInf, self).__init__()
        self.msgBox = QMessageBox()
        self.msgBox.setIcon(QMessageBox.Information)
        self.msgBox.setWindowTitle(toptxt)
        self.msgBox.setText(txt)
        self.msgBox.addButton(QPushButton('OK'), QMessageBox.YesRole)
        self.msgBox.exec_()

# GUI error dialog box class


class mQDialogErr(QDialog):
    def __init__(self, toptxt, txt):
        super(mQDialogErr, self).__init__()
        self.msgBox = QMessageBox()
        self.msgBox.setIcon(QMessageBox.Critical)
        self.msgBox.setWindowTitle(toptxt)
        self.msgBox.setText(txt)
        self.msgBox.addButton(QPushButton('OK'), QMessageBox.YesRole)
        self.msgBox.exec_()

# GUI warning dialog box class


class mQDialogWarn(QDialog):
    def __init__(self, toptxt, txt):
        super(mQDialogWarn, self).__init__()
        self.msgBox = QMessageBox()
        self.msgBox.setIcon(QMessageBox.Warning)
        self.msgBox.setWindowTitle(toptxt)
        self.msgBox.setText(txt)
        self.msgBox.addButton(QPushButton('OK'), QMessageBox.YesRole)
        self.msgBox.exec_()

# GUI message box class


class mQMessageBox(QMessageBox):
    def __init__(self, title, text, typ, autoclose):
        super(mQMessageBox, self).__init__()
        self.setText(text)
        if 'w' in typ.lower():
            self.setIcon(QMessageBox.Warning)
        elif 'e' in typ.lower():
            self.setIcon(QMessageBox.Critical)
        else:
            self.setIcon(QMessageBox.Information)
        self.addButton(QPushButton('OK'), QMessageBox.YesRole)
        self.autoclose = autoclose
        self.show()

    def showEvent(self, event):
        if self.autoclose:
            self.hide()

# GUI editor class


class mQTextEdit(QTextEdit):
    def __init__(self, txt, align, fontsize):
        super(mQTextEdit, self).__init__()
        self.setText(txt)
        if align in 'Ll':
            self.setAlignment(Qt.AlignLeft)
        elif align in 'Rr':
            self.setAlignment(Qt.AlignRight)
        else:
            self.setAlignment(Qt.AlignCenter)
        f = QFont("Helvetica", fontsize)
        self.setFont(f)
        self.show()

# GUI static texts class


class mQLabel(QLabel):
    # constructor needs text, tip, alignment, fontsize
    def __init__(self, txt, ctip, align, fontsize):
        super(mQLabel, self).__init__()
        self.setText(txt)  # set text
        # alignment
        if align in 'Ll':
            self.setAlignment(Qt.AlignLeft)
        if align in 'Rr':
            self.setAlignment(Qt.AlignRight)
        if align in 'C':
            self.setAlignment(Qt.AlignVCenter)
        if align in 'c':
            self.setAlignment(Qt.AlignHCenter)
        f = QFont("Helvetica", fontsize)
        self.setFont(f)
        self.setToolTip(ctip)
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        self.setWordWrap(True)

    def resize2Event(self, event):
        super(mQLabel, self).resizeEvent(event)
        if not self.text():
            return
        # --- fetch current parameters ----
        f = self.font()
        cr = self.contentsRect()
        # --- iterate to find the font size that fits the contentsRect ---
        dw = event.size().width() - event.oldSize().width()   # width change
        dh = event.size().height() - event.oldSize().height()  # height change
        fs = max(f.pixelSize(), 1)
        while True:
            f.setPixelSize(fs)
            br = QFontMetrics(f).boundingRect(self.text())
            if dw >= 0 and dh >= 0:  # label is expanding
                if br.height() <= cr.height() and br.width() <= cr.width():
                    fs += 1
                else:
                    f.setPixelSize(max(fs - 1, 1))  # backtrack
                    break
            else:  # label is shrinking
                if br.height() > cr.height() or br.width() > cr.width():
                    fs -= 1
                else:
                    break
            if fs < 1:
                break
        # --- update font size ---
        self.setFont(f)

# GUI edit texts class


class mQLineEdit(QLineEdit):
    # constructor needs text, tip, alignment, fontsize
    def __init__(self, txt, ctip, align, fontsize):
        super(mQLineEdit, self).__init__()
        self.setText(txt)  # set text
        # alignment
        if align in 'Ll':
            self.setAlignment(Qt.AlignLeft)
        elif align in 'Rr':
            self.setAlignment(Qt.AlignRight)
        else:
            self.setAlignment(Qt.AlignCenter)
        f = QFont("Helvetica", fontsize)
        self.setFont(f)
        self.setToolTip(ctip)
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        self.setCursorPosition(0)

# Another GUI edit texts class


class mQLineEditL(QComboBox):
    # constructor needs text, tip, alignment, fontsize
    def __init__(self, txt, ctip, align, fontsize, licores):
        super(mQLineEditL, self).__init__()
        f = QFont("Helvetica", fontsize)
        self.setFont(f)
        self.setToolTip(ctip)
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setEditable(True)
        # add a filter model to filter matching items
        self.pFilterModel = QSortFilterProxyModel(self)
        self.pFilterModel.setFilterCaseSensitivity(Qt.CaseInsensitive)
        self.pFilterModel.setSourceModel(self.model())
        # add a completer, which uses the filter model
        self.completer = QCompleter(self.pFilterModel, self)
        # always show all (filtered) completions
        self.completer.setCompletionMode(QCompleter.UnfilteredPopupCompletion)
        self.setCompleter(self.completer)
        # connect signals

        def filter(text):
            self.pFilterModel.setFilterFixedString(str(text))
        self.lineEdit().textEdited[str].connect(filter)
        self.completer.activated.connect(self.on_completer_activated)
        self.addItems(licores)
        self.setCurrentText(txt)  # set text to empty
    # on selection of an item from the completer, select the corresponding item from combobox

    def on_completer_activated(self, text):
        if text:
            index = self.findText(str(text))
            self.setCurrentIndex(index)

# GUI spin box class


class mQSpinBox(QSpinBox):
    # constructor needs text, tip, alignment, fontsize
    def __init__(self, ctip):
        super(mQSpinBox, self).__init__()
        self.setMinimum(0)  # set minimum
        self.setValue(1)
        self.setToolTip(ctip)
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)

# Determine image type and return its size
#
#  From draco


def get_image_size(fname):
    with open(fname, 'rb') as fhandle:
        head = fhandle.read(24)
        if len(head) != 24:
            return
        if imghdr.what(fname) == 'png':
            check = struct.unpack('>i', head[4:8])[0]
            if check != 0x0d0a1a0a:
                return
            width, height = struct.unpack('>ii', head[16:24])
        elif imghdr.what(fname) == 'gif':
            width, height = struct.unpack('<HH', head[6:10])
        elif imghdr.what(fname) == 'jpeg':
            try:
                fhandle.seek(0)  # Read 0xff next
                size = 2
                ftype = 0
                while not 0xc0 <= ftype <= 0xcf:
                    fhandle.seek(size, 1)
                    byte = fhandle.read(1)
                    while ord(byte) == 0xff:
                        byte = fhandle.read(1)
                    ftype = ord(byte)
                    size = struct.unpack('>H', fhandle.read(2))[0] - 2
                # We are at a SOFn block
                fhandle.seek(1, 1)  # Skip `precision' byte.
                height, width = struct.unpack('>HH', fhandle.read(4))
            except Exception:  # IGNORE:W0703
                return
        else:
            return
        return width, height

# GUI pixmap class


class mQPixmap(QLabel):
    def __init__(self, picpath):
        super(mQPixmap, self).__init__()
        self.pixmap = QPixmap(picpath)
        self.setPixmap(self.pixmap)
        self.setScaledContents(True)
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)

    def resizeEvent(self, event):
        w = self.width()
        h = self.height()
        self.setPixmap(self.pixmap.scaled(
            w, h, Qt.KeepAspectRatio, Qt.SmoothTransformation))

# GUI svg widget class


class mSvgWidget(QSvgWidget):
    def __init__(self, svgpath):
        super(mSvgWidget, self).__init__()
        self.load(svgpath)
        # minimum useful size of molecule svgs
        self.setMinimumSize(200, 200)
        policy = QSizePolicy(QSizePolicy.MinimumExpanding,
                             QSizePolicy.MinimumExpanding)
        self.setSizePolicy(policy)
