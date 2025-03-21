import sys
import os
import numpy as np
import pandas as pd
from PyQt5.QtGui import QCursor
from PyQt5.QtWidgets import QGraphicsOpacityEffect
from PyQt5.QtCore import QPropertyAnimation
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QWidget,
                             QPushButton, QLabel, QLineEdit, QFileDialog, QComboBox,
                             QTabWidget, QGroupBox, QTableView, QStatusBar, QInputDialog, QDialog)
from PyQt5.QtGui import QFont, QIcon
from PyQt5.QtCore import Qt, pyqtSignal, QAbstractTableModel
import pyqtgraph as pg
from scipy.signal import hilbert, find_peaks
from scipy.fft import fft, fftfreq
class DragOverlay(QWidget):
    """Полупрозрачный оверлей на весь экран с пунктирной границей и надписью по центру."""
    def __init__(self, parent=None):
        super().__init__(parent)
        # Разрешаем «сквозную» мышь, чтобы не мешать другим виджетам
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        # Убираем рамку окна
        self.setWindowFlags(Qt.Widget | Qt.FramelessWindowHint)

        # Фон полупрозрачный + пунктирная граница
        self.setStyleSheet("""
            QWidget {
                background-color: rgba(0, 0, 0, 128);
                border: 2px dashed #FFFFFF;
            }
        """)

        # Надпись по центру
        self.label = QLabel("Перетащите файл сюда для загрузки", self)
        self.label.setStyleSheet("color: #FFFFFF; font-size: 24px; font-weight: bold;")
        self.label.setAlignment(Qt.AlignCenter)

        # Эффект прозрачности для анимации
        self.opacity_effect = QGraphicsOpacityEffect(self)
        self.setGraphicsEffect(self.opacity_effect)

        # Анимация прозрачности (fade-in/fade-out)
        self.animation = QPropertyAnimation(self.opacity_effect, b"opacity")
        self.animation.setDuration(300)  # 300 мс
        self.animation.setStartValue(0.0)
        self.animation.setEndValue(1.0)

        # Изначально скрыто
        self.hide()
        self.opacity_effect.setOpacity(0.0)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # Растягиваем надпись на всю область оверлея, чтобы она была по центру
        self.label.setGeometry(self.rect())

    def showOverlay(self):
        self.show()
        self.animation.stop()
        self.animation.setDirection(QPropertyAnimation.Forward)
        self.animation.start()

    def hideOverlay(self):
        self.animation.stop()
        self.animation.setDirection(QPropertyAnimation.Backward)
        self.animation.start()
        self.animation.finished.connect(self._check_hide)

    def _check_hide(self):
        if self.animation.direction() == QPropertyAnimation.Backward:
            self.hide()


# ----- PandasModel для отображения больших таблиц -----
class PandasModel(QAbstractTableModel):
    def __init__(self, data, parent=None):
        super().__init__(parent)
        self._data = data

    def rowCount(self, parent=None):
        return self._data.shape[0]

    def columnCount(self, parent=None):
        return self._data.shape[1]

    def data(self, index, role=Qt.DisplayRole):
        if index.isValid() and role == Qt.DisplayRole:
            return str(self._data.iloc[index.row(), index.column()])
        return None

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role == Qt.DisplayRole:
            if orientation == Qt.Horizontal:
                return str(self._data.columns[section])
            else:
                return str(section)
        return None

    def update(self, data):
        self.beginResetModel()
        self._data = data
        self.endResetModel()

# ----- Кликабельный QLabel -----
class ClickableLabel(QLabel):
    clicked = pyqtSignal()
    def __init__(self, text, parent=None):
        super().__init__(text, parent)
        self.setStyleSheet("color: #00B4FF; font-weight: bold;")
        self.setCursor(Qt.PointingHandCursor)
    def mousePressEvent(self, event):
        self.clicked.emit()
        super().mousePressEvent(event)

# ----- Диалог выбора типа экспорта -----
class ExportTypeDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Export Type")
        self.setStyleSheet("""
            QDialog { background-color: #2E2E2E; color: #FFFFFF; }
            QLabel { color: #FFFFFF; }
            QPushButton { background-color: #505050; color: #FFFFFF; padding: 5px; }
            QComboBox { background-color: #353535; color: #FFFFFF; padding: 3px; }
        """)
        layout = QVBoxLayout(self)
        label = QLabel("Choose export format:")
        layout.addWidget(label)
        self.combo = QComboBox()
        self.combo.addItems(["Raster (PNG)", "Vector (SVG)"])
        layout.addWidget(self.combo)
        btn_layout = QHBoxLayout()
        self.ok_btn = QPushButton("OK")
        self.cancel_btn = QPushButton("Cancel")
        btn_layout.addWidget(self.ok_btn)
        btn_layout.addWidget(self.cancel_btn)
        layout.addLayout(btn_layout)
        self.ok_btn.clicked.connect(self.accept)
        self.cancel_btn.clicked.connect(self.reject)
        self.setFixedWidth(self.sizeHint().width() + 80)

# ----- Кнопка для загрузки файла с поддержкой drag-and-drop -----

    class FileDropButton(QPushButton):
        fileDropped = pyqtSignal(str)

        def __init__(self, text, parent=None):
            super().__init__(text, parent)
            self.setAcceptDrops(True)
            self.setAcceptDrops(True)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            if urls:
                file_path = urls[0].toLocalFile()
                if file_path.lower().endswith(('.txt', '.csv', '.xlsx')):
                    event.acceptProposedAction()
                    self.setText("Перетащите файл сюда")
                    return
        event.ignore()
    def dragLeaveEvent(self, event):
        self.setText(self.default_text)
        event.accept()
    def dropEvent(self, event):
        if event.mimeData().hasUrls():
            file_path = event.mimeData().urls()[0].toLocalFile()
            if file_path.lower().endswith(('.txt', '.csv', '.xlsx')):
                self.setText(self.default_text)
                self.fileDropped.emit(file_path)
                event.acceptProposedAction()
                return
        event.ignore()

# ----- Функция для стилизованного ввода числа (Sampling Frequency) -----
def getStyledDouble(parent, title, label,
                    min_val=0.1, max_val=100000.0):
    dialog = QInputDialog(parent)
    dialog.setWindowTitle(title)
    dialog.setLabelText(label)

    # Жёстко задаём 1000.0 как дефолт
    value = 1000.0
    dialog.setDoubleValue(value)
    dialog.setDoubleRange(min_val, max_val)
    dialog.setDoubleDecimals(2)

    # Убираем кнопку справки (иконку «?»)
    dialog.setWindowFlags(dialog.windowFlags() & ~Qt.WindowContextHelpButtonHint)

    # Стилизуем
    dialog.setStyleSheet(
        "QInputDialog { background-color: #2E2E2E; color: white; }"
        "QLabel { color: white; }"
        "QLineEdit { background-color: #353535; color: white; }"
        "QPushButton { background-color: #505050; color: white; border: 1px solid #606060; }"
    )

    dialog.adjustSize()
    dialog.setFixedWidth(dialog.width() + 100)

    result = dialog.exec_()
    if result == QDialog.Accepted:
        return dialog.doubleValue(), True
    else:
        return value, False

# ----- Виджет Raw Data с QTableView и белым шрифтом -----
class RawDataWidget(QWidget):
    def __init__(self, data, parent=None):
        super().__init__(parent)
        self.data = data
        self.init_ui()
    def init_ui(self):
        layout = QVBoxLayout(self)
        self.table = QTableView()
        self.table.setStyleSheet("""
            QTableView { background-color: #2E2E2E; color: #FFFFFF; }
            QHeaderView::section { background-color: #2E2E2E; color: #FFFFFF; }
        """)
        self.model = PandasModel(self.data)
        self.table.setModel(self.model)
        self.table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.table)
    def update_data(self, data):
        self.data = data
        self.model.update(data)

# ----- Виджет OriginalSignalWidget -----
class OriginalSignalWidget(QWidget):
    def __init__(self, signal, sampling_rate, parent=None):
        super().__init__(parent)
        self.signal = signal
        self.sampling_rate = sampling_rate
        self.peakShown = False
        self.peaksCurve = None
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        self.plot = pg.PlotWidget()
        self.plot.setBackground("#2E2E2E")
        t = np.arange(len(self.signal)) / self.sampling_rate
        self.curve = self.plot.plot(t, self.signal, pen=pg.mkPen('#00FF00', width=2))
        self.plot.setLabel('left', 'Amplitude', color='white')
        self.plot.setLabel('bottom', 'Points', color='white')
        self.plot.showGrid(x=True, y=True)
        layout.addWidget(self.plot)

        toolbar = QWidget()
        toolbar_layout = QHBoxLayout(toolbar)
        btn_zoom = QPushButton("Zoom")
        btn_find_peaks = QPushButton("Find Peaks")
        btn_export = QPushButton("Export Plot")
        toolbar_layout.addWidget(btn_zoom)
        toolbar_layout.addWidget(btn_find_peaks)
        toolbar_layout.addWidget(btn_export)
        layout.addWidget(toolbar)

        btn_zoom.clicked.connect(lambda: self.plot.getViewBox().autoRange())
        btn_find_peaks.clicked.connect(self.toggle_peaks)
        btn_export.clicked.connect(self.export_plot)

        self.setLayout(layout)

    def toggle_peaks(self):
        t = np.arange(len(self.signal)) / self.sampling_rate
        if not self.peakShown:
            peaks, _ = find_peaks(self.signal, prominence=1)
            self.peaksCurve = self.plot.plot(t[peaks], self.signal[peaks],
                                             pen=None, symbol='x', symbolBrush='r', symbolSize=10)
            self.peakShown = True
        else:
            if self.peaksCurve is not None:
                self.plot.removeItem(self.peaksCurve)
            self.peakShown = False

    def export_plot(self):
        dialog = ExportTypeDialog(self)
        if dialog.exec_() == QDialog.Accepted:
            export_type = dialog.combo.currentText()
            if export_type == "Raster (PNG)":
                file_name, _ = QFileDialog.getSaveFileName(self, "Save Plot", "",
                                                           "PNG Images (*.png);;All Files (*)")
                if not file_name:
                    return
                import pyqtgraph.exporters as exporters
                exporter = exporters.ImageExporter(self.plot.plotItem)
                exporter.parameters()['width'] = 1920
                exporter.export(file_name)
            elif export_type == "Vector (SVG)":
                file_name, _ = QFileDialog.getSaveFileName(self, "Save Plot as SVG", "",
                                                           "SVG Files (*.svg);;All Files (*)")
                if not file_name:
                    return
                import pyqtgraph.exporters as exporters
                exporter = exporters.SVGExporter(self.plot.plotItem)
                exporter.export(file_name)

# ----- Виджет SpectrumWidget -----
class SpectrumWidget(QWidget):
    def __init__(self, mained_signal, sampling_rate, zero_padding=0, parent=None):
        super().__init__(parent)
        self.mained_signal = mained_signal
        self.sampling_rate = sampling_rate
        self.zero_padding = zero_padding
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        self.plot = pg.PlotWidget()
        self.plot.setBackground("#2E2E2E")
        self.plot.setLabel('left', 'Amplitude', color='white')
        self.plot.setLabel('bottom', 'Frequency (Hz)', color='white')
        self.plot.showGrid(x=True, y=True)

        padded_signal = np.pad(self.mained_signal, (0, self.zero_padding), 'constant')
        N = len(padded_signal)
        T = 1.0 / self.sampling_rate
        yf = fft(padded_signal)
        xf = fftfreq(N, T)[:N // 2]
        amplitude = 2.0 / N * np.abs(yf[:N // 2])
        if len(xf) > 1:
            xf = xf[1:]
            amplitude = amplitude[1:]

        self.plot.plot(xf, amplitude, pen=pg.mkPen('#00FFFF', width=2))
        layout.addWidget(self.plot)

        toolbar = QWidget()
        toolbar_layout = QHBoxLayout(toolbar)
        btn_zoom = QPushButton("Zoom")
        btn_pan = QPushButton("Pan")
        btn_export = QPushButton("Export Plot")
        toolbar_layout.addWidget(btn_zoom)
        toolbar_layout.addWidget(btn_pan)
        toolbar_layout.addWidget(btn_export)
        layout.addWidget(toolbar)

        btn_zoom.clicked.connect(lambda: self.plot.getViewBox().autoRange())
        btn_pan.clicked.connect(lambda: self.plot.getViewBox().setMouseMode(pg.ViewBox.PanMode))
        btn_export.clicked.connect(self.export_plot)

        self.setLayout(layout)

    def export_plot(self):
        dialog = ExportTypeDialog(self)
        if dialog.exec_() == QDialog.Accepted:
            export_type = dialog.combo.currentText()
            if export_type == "Raster (PNG)":
                file_name, _ = QFileDialog.getSaveFileName(self, "Save Plot", "",
                                                           "PNG Images (*.png);;All Files (*)")
                if not file_name:
                    return
                import pyqtgraph.exporters as exporters
                exporter = exporters.ImageExporter(self.plot.plotItem)
                exporter.parameters()['width'] = 1920
                exporter.export(file_name)
            elif export_type == "Vector (SVG)":
                file_name, _ = QFileDialog.getSaveFileName(self, "Save Plot as SVG", "",
                                                           "SVG Files (*.svg);;All Files (*)")
                if not file_name:
                    return
                import pyqtgraph.exporters as exporters
                exporter = exporters.SVGExporter(self.plot.plotItem)
                exporter.export(file_name)

# ----- Виджет PhaseWidget -----
class PhaseWidget(QWidget):
    def __init__(self, time, phase, parent=None):
        super().__init__(parent)
        self.time = time
        self.phase = phase
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        self.plot = pg.PlotWidget()
        self.plot.setBackground("#2E2E2E")
        self.plot.plot(self.time, self.phase, pen=pg.mkPen('#00FFFF', width=2))
        self.plot.setLabel('left', 'Phase (radians)', color='white')
        self.plot.setLabel('bottom', 'Time (s)', color='white')
        self.plot.showGrid(x=True, y=True)

        layout.addWidget(self.plot)

        toolbar = QWidget()
        toolbar_layout = QHBoxLayout(toolbar)
        btn_zoom = QPushButton("Zoom")
        btn_pan = QPushButton("Pan")
        btn_export = QPushButton("Export Plot")
        toolbar_layout.addWidget(btn_zoom)
        toolbar_layout.addWidget(btn_pan)
        toolbar_layout.addWidget(btn_export)
        layout.addWidget(toolbar)

        btn_zoom.clicked.connect(lambda: self.plot.getViewBox().autoRange())
        btn_pan.clicked.connect(lambda: self.plot.getViewBox().setMouseMode(pg.ViewBox.PanMode))
        btn_export.clicked.connect(self.export_plot)

        self.setLayout(layout)

    def export_plot(self):
        dialog = ExportTypeDialog(self)
        if dialog.exec_() == QDialog.Accepted:
            export_type = dialog.combo.currentText()
            if export_type == "Raster (PNG)":
                file_name, _ = QFileDialog.getSaveFileName(self, "Save Plot", "",
                                                           "PNG Images (*.png);;All Files (*)")
                if not file_name:
                    return
                import pyqtgraph.exporters as exporters
                exporter = exporters.ImageExporter(self.plot.plotItem)
                exporter.parameters()['width'] = 1920
                exporter.export(file_name)
            elif export_type == "Vector (SVG)":
                file_name, _ = QFileDialog.getSaveFileName(self, "Save Plot as SVG", "",
                                                           "SVG Files (*.svg);;All Files (*)")
                if not file_name:
                    return
                import pyqtgraph.exporters as exporters
                exporter = exporters.SVGExporter(self.plot.plotItem)
                exporter.export(file_name)

# ----- Виджет MainWidget -----
class MainWidget(QWidget):
    def __init__(self, signal, sampling_rate=1000.0, parent=None):
        super().__init__(parent)
        self.signal = signal
        self.sampling_rate = sampling_rate
        self.original_signal = signal.copy()
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)

        zero_pad_layout = QHBoxLayout()
        zero_label = QLabel("Number of Zeros:")
        zero_label.setStyleSheet("color: #FFFFFF;")
        self.zero_input = QLineEdit("0")
        self.zero_input.setStyleSheet("background-color: #353535; color: #FFFFFF; padding: 3px;")
        self.btn_update = QPushButton("Update Spectrum")
        self.btn_update.setStyleSheet("background-color: #00B4FF; font-weight: bold; padding: 5px;")
        self.btn_update.clicked.connect(self.update_spectrum)
        zero_pad_layout.addWidget(zero_label)
        zero_pad_layout.addWidget(self.zero_input, 1)
        zero_pad_layout.addWidget(self.btn_update, 1)
        layout.addLayout(zero_pad_layout)

        if self.signal.size == 0:
            self.mained_signal = np.array([])
            self.instantaneous_phase = np.array([])
            self.time = np.array([])
        else:
            analytic_signal = hilbert(self.signal)
            self.mained_signal = np.abs(analytic_signal)
            self.instantaneous_phase = np.unwrap(np.angle(analytic_signal))
            self.time = np.arange(len(self.signal)) / self.sampling_rate

        # Original Signal
        self.label_original = ClickableLabel("Original Signal")
        self.label_original.setStyleSheet("color: #00B4FF; font-weight: bold;")
        self.label_original.clicked.connect(self.open_original_tab)
        layout.addWidget(self.label_original)

        self.plot_original = pg.PlotWidget()
        self.plot_original.setBackground("#2E2E2E")
        if self.signal.size:
            self.plot_original.plot(self.signal, pen=pg.mkPen('#00FF00', width=2))
        self.plot_original.setLabel('left', 'Amplitude', color='white')
        self.plot_original.setLabel('bottom', 'Points', color='white')
        self.plot_original.showGrid(x=True, y=True)
        layout.addWidget(self.plot_original)

        # Spectrum
        self.label_spectrum = ClickableLabel("Spectrum of mained Signal")
        self.label_spectrum.clicked.connect(self.open_spectrum_tab)
        layout.addWidget(self.label_spectrum)

        self.plot_spectrum = pg.PlotWidget()
        self.plot_spectrum.setBackground("#2E2E2E")
        self.plot_spectrum.setLabel('left', 'Amplitude', color='white')
        self.plot_spectrum.setLabel('bottom', 'Frequency (Hz)', color='white')
        self.plot_spectrum.showGrid(x=True, y=True)
        layout.addWidget(self.plot_spectrum)
        # Phase
        self.label_phase = ClickableLabel("Phase vs Time")
        self.label_phase.clicked.connect(self.open_phase_tab)
        layout.addWidget(self.label_phase)

        self.plot_phase = pg.PlotWidget()
        self.plot_phase.setBackground("#2E2E2E")
        self.plot_phase.plot(self.time, self.instantaneous_phase, pen=pg.mkPen('#00FFFF', width=2))
        self.plot_phase.setLabel('left', 'Phase (radians)', color='white')
        self.plot_phase.setLabel('bottom', 'Time (s)', color='white')
        self.plot_phase.showGrid(x=True, y=True)
        layout.addWidget(self.plot_phase)

        # Toolbar
        self.main_toolbar = QWidget()
        main_tb_layout = QHBoxLayout(self.main_toolbar)
        self.btn_zoom = QPushButton("Zoom")
        self.btn_pan = QPushButton("Pan")
        self.btn_export_plots = QPushButton("Export All Plots")
        main_tb_layout.addWidget(self.btn_zoom)
        main_tb_layout.addWidget(self.btn_pan)
        main_tb_layout.addWidget(self.btn_export_plots)
        layout.addWidget(self.main_toolbar)

        self.btn_zoom.clicked.connect(lambda: self.plot_spectrum.getViewBox().setMouseMode(3))
        self.btn_pan.clicked.connect(lambda: self.plot_spectrum.getViewBox().setMouseMode(1))
        self.btn_export_plots.clicked.connect(self.export_all_plots)

        self.setLayout(layout)
        self.update_spectrum()

    def update_spectrum(self):
        try:
            n_zeros = int(self.zero_input.text())
        except ValueError:
            n_zeros = 0

        if self.original_signal.size == 0:
            self.plot_spectrum.clear()
            return

        # Добавляем нули
        padded_signal = np.pad(self.original_signal, (0, n_zeros), 'constant')
        self.signal = padded_signal.copy()
        self.time = np.arange(len(self.signal)) / self.sampling_rate

        # Redraw Original
        self.plot_original.clear()
        self.plot_original.plot(self.signal, pen=pg.mkPen('#00FF00', width=2))

        analytic_signal = hilbert(padded_signal)
        main_signal = np.abs(analytic_signal)
        N = len(main_signal)
        T = 1.0 / self.sampling_rate
        yf = fft(main_signal)
        xf = fftfreq(N, T)[:N // 2]
        amplitude = 2.0 / N * np.abs(yf[:N // 2])

        # remove first harmonic
        if len(xf) > 1:
            xf = xf[1:]
            amplitude = amplitude[1:]

        self.plot_spectrum.clear()
        self.plot_spectrum.plot(xf, amplitude, pen=pg.mkPen('#00FFFF', width=2))

    def export_all_plots(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Folder to Save Plots")
        if not folder:
            return
        base_name, ok = QInputDialog.getText(self, "File Base Name", "Enter base name for plots:")
        if not ok or not base_name:
            return

        import pyqtgraph.exporters as exporters
        plots = {
            "original": self.plot_original,
            "spectrum": self.plot_spectrum,
            "phase": self.plot_phase
        }
        for key, plot in plots.items():
            exporter = exporters.ImageExporter(plot.plotItem)
            exporter.parameters()['width'] = 1920
            file_name = os.path.join(folder, f"{base_name}_{key}.png")
            exporter.export(file_name)

        parent = self.window()
        if parent is not None and hasattr(parent, 'statusBar'):
            parent.statusBar().showMessage("Plots exported successfully")

    def open_original_tab(self):
        if self.signal.size == 0:
            self.window().openOriginalSignalTab(np.array([]), self.sampling_rate)
            return
        freq, ok = getStyledDouble(
            self,
            "Sampling Frequency",
            "Enter sampling frequency (Hz):",
            min_val=0.1, max_val=100000.0
        )
        if ok:
            self.window().openOriginalSignalTab(self.signal, freq)

    def open_spectrum_tab(self):
        """Открываем диалог ввода частоты дискретизации и вызываем метод главного окна openSpectrumTab(...)"""
        main_win = self.window()
        if not hasattr(main_win, 'data') or main_win.data is None:
            if main_win is not None and hasattr(main_win, 'statusBar'):
                main_win.statusBar().showMessage("No data loaded")
            return

        freq, ok = getStyledDouble(
            self,
            "Sampling Frequency",
            "Enter sampling frequency (Hz):",
            min_val=0.1, max_val=100000.0
        )
        if ok:
            # Вызываем метод главного окна, передавая mained_signal
            main_win.openSpectrumTab(self.mained_signal, freq)

    def open_phase_tab(self):
        if self.window().switchToExistingTab("Phase vs Time"):
            return
        self.window().openPhaseTab(self.time, self.instantaneous_phase)

    def reset_zoom(self):
        self.plot_original.getViewBox().autoRange()
        self.plot_spectrum.getViewBox().autoRange()
        self.plot_phase.getViewBox().autoRange()

    def update_signal(self, signal, sampling_rate):
        self.original_signal = signal.copy()
        self.signal = signal.copy()
        self.sampling_rate = sampling_rate

        if self.signal.size:
            analytic_signal = hilbert(self.signal)
            self.mained_signal = np.abs(analytic_signal)
            self.instantaneous_phase = np.unwrap(np.angle(analytic_signal))
            self.time = np.arange(len(self.signal)) / self.sampling_rate
        else:
            self.mained_signal = np.array([])
            self.instantaneous_phase = np.array([])
            self.time = np.array([])

        self.plot_original.clear()
        if self.signal.size:
            self.plot_original.plot(self.signal, pen=pg.mkPen('#00FF00', width=2))

        self.plot_phase.clear()
        if self.time.size and self.instantaneous_phase.size:
            self.plot_phase.plot(self.time, self.instantaneous_phase, pen=pg.mkPen('#00FFFF', width=2))

        self.update_spectrum()

# ----- Главное окно приложения -----
class AdvancedInterferometerApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Multi-Beam Interferometer Analyzer")
        self.setGeometry(100, 50, 1280, 800)
        self.setWindowIcon(QIcon('interferometer_icon.png'))

        # Разрешаем Drag & Drop только на главное окно
        self.setAcceptDrops(True)

        # Стили для окна

        self.setStyleSheet("""
            QMainWindow { background-color: #2E2E2E; }
            QGroupBox { background-color: #404040; border: 2px solid #606060; border-radius: 5px; margin-top: 1ex; color: #FFFFFF; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 3px; color: #00B4FF; }
            QPushButton { background-color: #505050; border: 1px solid #606060; border-radius: 3px; color: #FFFFFF; padding: 5px; min-width: 80px; }
            QPushButton:hover { background-color: #606060; }
            QLineEdit, QComboBox { background-color: #353535; border: 1px solid #505050; color: #FFFFFF; padding: 3px; }
            QTabWidget::pane { background-color: #2E2E2E; border: 1px solid #606060; margin: 2px; }
            QTabBar::tab { background: #404040; color: #FFFFFF; padding: 8px 15px; border-top-left-radius: 5px; border-top-right-radius: 5px; }
            QTabBar::tab:selected { background: #505050; }
        """)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QHBoxLayout(self.central_widget)

        self.left_panel = QWidget()
        self.left_panel.setFixedWidth(300)

        self.left_panel.setAcceptDrops(False)

        self.left_layout = QVBoxLayout(self.left_panel)
        self.main_layout.addWidget(self.left_panel)

        self.tab_widget = QTabWidget()
        self.tab_widget.setTabPosition(QTabWidget.West)

        self.tab_widget.setAcceptDrops(False)
        self.tab_widget.tabCloseRequested.connect(self.closeTab)
        self.main_layout.addWidget(self.tab_widget)

        self.peaks_visible = False
        self.data = None
        self.current_channel = 0
        self.processed_data = None

        # По умолчанию вкладка Main
        self.main_widget = MainWidget(np.array([]), 1000.0)
        self.tab_widget.addTab(self.main_widget, "Main")

        # Создаём оверлей для Drag & Drop (предполагается, что класс DragOverlay уже есть)
        self.drag_overlay = DragOverlay(self.central_widget)
        self.drag_overlay.raise_()

        self.init_data_controls()
        self.init_processing_controls()
        ###################################################################
        # Метод resizeEvent: оверлей растягивается на всё окно
        ###################################################################
    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.drag_overlay.setGeometry(self.central_widget.rect())

        ###################################################################
        # Drag & Drop методы
        ###################################################################
    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            if urls:
                file_path = urls[0].toLocalFile()
                if file_path.lower().endswith(('.txt', '.csv', '.xlsx')):
                    event.acceptProposedAction()
                    # Показываем оверлей, если ещё не виден
                    if not self.drag_overlay.isVisible():
                        self.drag_overlay.showOverlay()
                    return
        event.ignore()

    def dragMoveEvent(self, event):
        # При перемещении курсора обновляем положение и остаёмся в режиме принятия, если курсор внутри окна
        if self.rect().contains(self.mapFromGlobal(QCursor.pos())):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragLeaveEvent(self, event):
        # Скрывать оверлей только если курсор действительно покинул окно
        if not self.rect().contains(self.mapFromGlobal(QCursor.pos())):
            self.drag_overlay.hideOverlay()
        event.accept()

    def dropEvent(self, event):
        # Не скрываем оверлей сразу, а вызываем его скрытие после успешной загрузки файла
        if event.mimeData().hasUrls():
            file_path = event.mimeData().urls()[0].toLocalFile()
            if file_path.lower().endswith(('.txt', '.csv', '.xlsx')):
                self.open_file(file_path)
                # После загрузки данных (или даже при ошибке) скрываем оверлей
                self.drag_overlay.hideOverlay()
                event.acceptProposedAction()
                return
        event.ignore()
        ###################################################################
        # Инициализация UI-блоков (Data controls, Processing, Analysis)
        ##################################################################

    def init_data_controls(self):
        group = QGroupBox("Data Management")
        layout = QVBoxLayout(group)

        self.btn_load_data = QPushButton("Load Data File")
        self.btn_load_data.setIcon(QIcon('icons/load_icon.png'))
        self.btn_load_data.clicked.connect(self.load_data)

        self.channel_selector = QComboBox()
        self.channel_selector.currentIndexChanged.connect(self.update_channel)

        layout.addWidget(self.btn_load_data)
        layout.addWidget(QLabel("Select Channel:"))
        layout.addWidget(self.channel_selector)

        self.data_info = QLabel("No data loaded")
        self.data_info.setAlignment(Qt.AlignCenter)
        self.data_info.setStyleSheet("color: #AAAAAA;")
        layout.addWidget(self.data_info)

        self.btn_show_raw_data = QPushButton("Show Raw Data")
        self.btn_show_raw_data.clicked.connect(self.open_raw_data_tab)
        layout.addWidget(self.btn_show_raw_data)

        self.left_layout.addWidget(group)

    def init_processing_controls(self):
        group = QGroupBox("Signal Processing")
        layout = QVBoxLayout(group)

        self.filter_type = QComboBox()
        self.filter_type.addItems(['Moving Average', 'Gaussian', 'Savitzky-Golay'])

        self.filter_param = QLineEdit('5')
        self.filter_param.setPlaceholderText('Window size')

        self.btn_process = QPushButton("Apply Filter")
        self.btn_process.clicked.connect(self.apply_filter)

        layout.addWidget(QLabel("Filter Type:"))
        layout.addWidget(self.filter_type)
        layout.addWidget(QLabel("Filter Parameters:"))
        layout.addWidget(self.filter_param)
        layout.addWidget(self.btn_process)

        self.left_layout.addWidget(group)

    def open_file(self, file_path):
        self.statusBar().showMessage("Loading data...")
        QApplication.processEvents()

        try:
            if file_path.lower().endswith('.csv'):
                self.data = pd.read_csv(file_path)
            elif file_path.lower().endswith('.xlsx'):
                self.data = pd.read_excel(file_path)
            elif file_path.lower().endswith('.txt'):
                try:
                    self.data = pd.read_csv(file_path, sep='\s+', header=None)
                except Exception:
                    arr = np.loadtxt(file_path)
                    self.data = pd.DataFrame(arr)
            else:
                self.data = pd.read_csv(file_path)

            if self.data.empty:
                raise ValueError("File is empty")

            if not np.issubdtype(self.data.iloc[:, 0].dtype, np.number):
                raise ValueError("Non-numeric data detected")
            # Перевод из dB (мощностных) в амплитуду: amp = 10^(dB/20)
            self.data = self.data.apply(lambda col: 10 ** (col / 20))  # Преобразование dB в амплитуду
            self.update_interface()
            self.statusBar().showMessage(f"Loaded: {os.path.basename(file_path)}")

        except Exception as e:
            self.statusBar().showMessage(f"Error: {str(e)}")
            self.data = None
            self.processed_data = None
            self.update_interface()

    def load_data(self):
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Open Data File", "",
            "Text Files (*.txt);;CSV Files (*.csv);;Excel Files (*.xlsx);;All Files (*)"
        )
        if file_name:
            self.open_file(file_name)

    def load_data_dropped(self, file_path):
        self.open_file(file_path)

    def update_interface(self):
        if self.data is None:
            self.channel_selector.clear()
            self.data_info.setText("No data loaded")
            return

        self.channel_selector.clear()
        self.channel_selector.addItems(self.data.columns.astype(str))

        self.data_info.setText(
            f"Channels: {len(self.data.columns)}\n"
            f"Samples: {len(self.data)}\n"
            f"Type: {self.data.iloc[:, 0].dtype}"
        )

        # Обновляем Main виджет
        signal = self.data.iloc[:, self.current_channel].values
        self.main_widget.update_signal(signal, 1000.0)

    def update_channel(self, index):
        self.current_channel = index
        if self.data is not None:
            signal = self.data.iloc[:, self.current_channel].values
            self.main_widget.update_signal(signal, 1000.0)

    def apply_filter(self):
        try:
            window = int(self.filter_param.text())
            raw = self.data.iloc[:, self.current_channel].values
            if self.filter_type.currentText() == 'Moving Average':
                kernel = np.ones(window) / window
                self.processed_data = np.convolve(raw, kernel, mode='same')
            self.statusBar().showMessage("Filter applied successfully")
        except Exception as e:
            self.statusBar().showMessage(f"Filter Error: {str(e)}")

    def open_raw_data_tab(self):
        if self.data is None:
            self.statusBar().showMessage("No data loaded")
            return
        for i in range(self.tab_widget.count()):
            if self.tab_widget.tabText(i) == "Raw Data":
                raw_tab = self.tab_widget.widget(i)
                if hasattr(raw_tab, "update_data"):
                    raw_tab.update_data(self.data)
                self.tab_widget.setCurrentIndex(i)
                return
        raw_widget = RawDataWidget(self.data)
        self.tab_widget.addTab(raw_widget, "Raw Data")
        self.tab_widget.setCurrentWidget(raw_widget)

    def openOriginalSignalTab(self, signal, sampling_rate):
        for i in range(self.tab_widget.count()):
            if self.tab_widget.tabText(i) == "Original Signal":
                self.tab_widget.setCurrentIndex(i)
                return
        widget = OriginalSignalWidget(signal, sampling_rate)
        self.tab_widget.addTab(widget, "Original Signal")
        self.tab_widget.setCurrentWidget(widget)

    def openSpectrumTab(self, mained_signal, sampling_rate):
        for i in range(self.tab_widget.count()):
            if self.tab_widget.tabText(i) == "Spectrum of mained Signal":
                self.tab_widget.setCurrentIndex(i)
                return
        widget = SpectrumWidget(mained_signal, sampling_rate, zero_padding=0)
        self.tab_widget.addTab(widget, "Spectrum of mained Signal")
        self.tab_widget.setCurrentWidget(widget)

    def openPhaseTab(self, time, phase):
        for i in range(self.tab_widget.count()):
            if self.tab_widget.tabText(i) == "Phase vs Time":
                self.tab_widget.setCurrentIndex(i)
                return
        widget = PhaseWidget(time, phase)
        self.tab_widget.addTab(widget, "Phase vs Time")
        self.tab_widget.setCurrentWidget(widget)

    def open_Main_widget(self):
        self.tab_widget.setCurrentWidget(self.main_widget)

    def switchToExistingTab(self, title):
        for i in range(self.tab_widget.count()):
            if self.tab_widget.tabText(i) == title:
                self.tab_widget.setCurrentIndex(i)
                return True
        return False

    def closeTab(self, index):
        self.tab_widget.removeTab(index)
if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setFont(QFont("Helvetica", 10))
    window = AdvancedInterferometerApp()
    window.show()
    sys.exit(app.exec_())

