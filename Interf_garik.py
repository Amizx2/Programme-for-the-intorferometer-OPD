import sys
import os
import numpy as np
import pandas as pd
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QWidget,
                             QPushButton, QLabel, QLineEdit, QFileDialog, QComboBox,
                             QTabWidget, QGroupBox, QStatusBar, QInputDialog, QDialog, QTableView)
from PyQt5.QtGui import QFont, QIcon
from PyQt5.QtCore import Qt, pyqtSignal, QAbstractTableModel
import pyqtgraph as pg
from scipy.signal import hilbert, find_peaks
from scipy.fft import fft, fftfreq

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
        self.default_text = text
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

# ----- Виджет Raw Data, теперь с QTableView -----
class RawDataWidget(QWidget):
    def __init__(self, data, parent=None):
        super().__init__(parent)
        self.data = data
        self.init_ui()
    def init_ui(self):
        layout = QVBoxLayout(self)
        self.table = QTableView()
        self.model = PandasModel(self.data)
        self.table.setModel(self.model)
        self.table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.table)
    def update_data(self, data):
        self.data = data
        self.model.update(data)

# ----- Виджет для отображения оригинального сигнала в новом окне -----
class OriginalSignalWidget(QWidget):
    def __init__(self, signal, sampling_rate, parent=None):
        super().__init__(parent)
        self.signal = signal
        self.sampling_rate = sampling_rate
        self.init_ui()
    def init_ui(self):
        layout = QVBoxLayout(self)
        self.plot = pg.PlotWidget(title="Original Signal")
        self.plot.setBackground('k')
        t = np.arange(len(self.signal)) / self.sampling_rate
        self.plot.plot(t, self.signal, pen=pg.mkPen('#00FF00', width=2))
        self.plot.setLabel('left', 'Amplitude')
        self.plot.setLabel('bottom', 'Time (s)')
        self.plot.showGrid(x=True, y=True)
        layout.addWidget(self.plot)

# ----- Виджет для отображения спектра (без первой гармоники) -----
class SpectrumWidget(QWidget):
    def __init__(self, demodulated_signal, sampling_rate, zero_padding=0, parent=None):
        super().__init__(parent)
        self.demodulated_signal = demodulated_signal
        self.sampling_rate = sampling_rate
        self.zero_padding = zero_padding
        self.init_ui()
    def init_ui(self):
        layout = QVBoxLayout(self)
        self.plot = pg.PlotWidget(title="Spectrum of Demodulated Signal")
        self.plot.setBackground('k')
        self.plot.setLabel('left', 'Amplitude')
        self.plot.setLabel('bottom', 'Frequency (Hz)')
        self.plot.showGrid(x=True, y=True)
        padded_signal = np.pad(self.demodulated_signal, (0, self.zero_padding), 'constant')
        N = len(padded_signal)
        T = 1.0 / self.sampling_rate
        yf = fft(padded_signal)
        xf = fftfreq(N, T)[:N // 2]
        amplitude = 2.0 / N * np.abs(yf[:N // 2])
        # Исключаем первую гармонику (первый элемент)
        xf = xf[1:]
        amplitude = amplitude[1:]
        self.plot.plot(xf, amplitude, pen=pg.mkPen('#00FFFF', width=2))
        layout.addWidget(self.plot)

# ----- Виджет для отображения фазы -----
class PhaseWidget(QWidget):
    def __init__(self, time, phase, parent=None):
        super().__init__(parent)
        self.time = time
        self.phase = phase
        self.init_ui()
    def init_ui(self):
        layout = QVBoxLayout(self)
        self.plot = pg.PlotWidget(title="Phase vs Time")
        self.plot.setBackground('k')
        self.plot.plot(self.time, self.phase, pen=pg.mkPen('#00FFFF', width=2))
        self.plot.setLabel('left', 'Phase (radians)')
        self.plot.setLabel('bottom', 'Time (s)')
        self.plot.showGrid(x=True, y=True)
        layout.addWidget(self.plot)

# ----- Виджет демодуляции с кликабельными заголовками и возможностью обновления данных -----
class DemodulationWidget(QWidget):
    def __init__(self, signal, sampling_rate=1000.0, parent=None):
        super().__init__(parent)
        self.signal = signal
        self.sampling_rate = sampling_rate
        self.init_ui()
    def init_ui(self):
        layout = QVBoxLayout(self)
        # Блок управления нулями
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
        # Если сигнал пустой, задаём пустые массивы
        if self.signal.size == 0:
            self.demodulated_signal = np.array([])
            self.instantaneous_phase = np.array([])
            self.time = np.array([])
        else:
            analytic_signal = hilbert(self.signal)
            self.demodulated_signal = np.abs(analytic_signal)
            self.instantaneous_phase = np.unwrap(np.angle(analytic_signal))
            self.time = np.arange(len(self.signal)) / self.sampling_rate

        # --- Блок "Original Signal" с кликабельным заголовком ---
        self.label_original = ClickableLabel("Original Signal")
        self.label_original.clicked.connect(self.open_original_tab)
        layout.addWidget(self.label_original)
        self.plot_original = pg.PlotWidget()
        self.plot_original.setBackground('k')
        if self.signal.size:
            self.plot_original.plot(self.signal, pen=pg.mkPen('#00FF00', width=2))
        self.plot_original.setLabel('left', 'Amplitude')
        self.plot_original.setLabel('bottom', 'Sample Index')
        self.plot_original.showGrid(x=True, y=True)
        layout.addWidget(self.plot_original)

        # --- Блок "Spectrum of Demodulated Signal" с кликабельным заголовком ---
        self.label_spectrum = ClickableLabel("Spectrum of Demodulated Signal")
        self.label_spectrum.clicked.connect(self.open_spectrum_tab)
        layout.addWidget(self.label_spectrum)
        self.plot_spectrum = pg.PlotWidget()
        self.plot_spectrum.setBackground('k')
        self.plot_spectrum.setLabel('left', 'Amplitude')
        self.plot_spectrum.setLabel('bottom', 'Frequency (Hz)')
        self.plot_spectrum.showGrid(x=True, y=True)
        layout.addWidget(self.plot_spectrum)

        # --- Блок "Phase vs Time" с кликабельным заголовком ---
        self.label_phase = ClickableLabel("Phase vs Time")
        self.label_phase.clicked.connect(self.open_phase_tab)
        layout.addWidget(self.label_phase)
        self.plot_phase = pg.PlotWidget()
        self.plot_phase.setBackground('k')
        self.plot_phase.plot(self.time, self.instantaneous_phase, pen=pg.mkPen('#00FFFF', width=2))
        self.plot_phase.setLabel('left', 'Phase (radians)')
        self.plot_phase.setLabel('bottom', 'Time (s)')
        self.plot_phase.showGrid(x=True, y=True)
        layout.addWidget(self.plot_phase)

        # Панель инструментов для Demodulation
        self.demod_toolbar = QWidget()
        demod_tb_layout = QHBoxLayout(self.demod_toolbar)
        self.btn_zoom = QPushButton("Zoom")
        self.btn_pan = QPushButton("Pan")
        self.btn_export_plots = QPushButton("Export All Plots")
        demod_tb_layout.addWidget(self.btn_zoom)
        demod_tb_layout.addWidget(self.btn_pan)
        demod_tb_layout.addWidget(self.btn_export_plots)
        layout.addWidget(self.demod_toolbar)
        self.btn_zoom.clicked.connect(lambda: self.plot_spectrum.getViewBox().setMouseMode(3))
        self.btn_pan.clicked.connect(lambda: self.plot_spectrum.getViewBox().setMouseMode(1))
        self.btn_export_plots.clicked.connect(self.export_all_plots)
        self.update_spectrum()
    def update_spectrum(self):
        try:
            n_zeros = int(self.zero_input.text())
        except ValueError:
            n_zeros = 0
        if self.demodulated_signal.size == 0:
            self.plot_spectrum.clear()
            return
        padded_signal = np.pad(self.demodulated_signal, (0, n_zeros), 'constant')
        N = len(padded_signal)
        T = 1.0 / self.sampling_rate
        yf = fft(padded_signal)
        xf = fftfreq(N, T)[:N // 2]
        amplitude = 2.0 / N * np.abs(yf[:N // 2])
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
        # При клике открывается новая вкладка с оригинальным сигналом (с выбором sampling frequency)
        freq, ok = QInputDialog.getDouble(self, "Sampling Frequency",
                                            "Enter sampling frequency (Hz):",
                                            value=self.sampling_rate, min=0.1, max=100000.0)
        if ok:
            self.window().openOriginalSignalTab(self.signal, freq)
    def open_spectrum_tab(self):
        freq, ok = QInputDialog.getDouble(self, "Sampling Frequency",
                                            "Enter sampling frequency (Hz):",
                                            value=self.sampling_rate, min=0.1, max=100000.0)
        if ok:
            self.window().openSpectrumTab(self.demodulated_signal, freq)
    def open_phase_tab(self):
        self.window().openPhaseTab(self.time, self.instantaneous_phase)
    def update_signal(self, signal, sampling_rate):
        # Метод для обновления данных в виджете Demodulation
        self.signal = signal
        self.sampling_rate = sampling_rate
        analytic_signal = hilbert(self.signal)
        self.demodulated_signal = np.abs(analytic_signal)
        self.instantaneous_phase = np.unwrap(np.angle(analytic_signal))
        self.time = np.arange(len(self.signal)) / self.sampling_rate
        self.plot_original.clear()
        self.plot_original.plot(self.signal, pen=pg.mkPen('#00FF00', width=2))
        self.plot_phase.clear()
        self.plot_phase.plot(self.time, self.instantaneous_phase, pen=pg.mkPen('#00FFFF', width=2))
        self.update_spectrum()

# ----- Виджет для FFT анализа (без изменений) -----
class FFTAnalysisWidget(QWidget):
    def __init__(self, signal, sample_rate, parent=None):
        super().__init__(parent)
        self.signal = signal
        self.sample_rate = sample_rate
        self.init_ui()
    def init_ui(self):
        layout = QVBoxLayout(self)
        self.fft_plot = pg.PlotWidget(title="FFT Analysis")
        self.fft_plot.setBackground('k')
        self.fft_plot.setLabel('left', 'Amplitude')
        self.fft_plot.setLabel('bottom', 'Frequency (Hz)')
        self.fft_plot.showGrid(x=True, y=True)
        layout.addWidget(self.fft_plot)
        self.toolbar = QWidget()
        tb_layout = QHBoxLayout(self.toolbar)
        self.btn_zoom = QPushButton("Zoom")
        self.btn_pan = QPushButton("Pan")
        self.btn_export_plot = QPushButton("Export Plot")
        tb_layout.addWidget(self.btn_zoom)
        tb_layout.addWidget(self.btn_pan)
        tb_layout.addWidget(self.btn_export_plot)
        layout.addWidget(self.toolbar)
        self.btn_zoom.clicked.connect(lambda: self.fft_plot.getViewBox().setMouseMode(3))
        self.btn_pan.clicked.connect(lambda: self.fft_plot.getViewBox().setMouseMode(1))
        self.btn_export_plot.clicked.connect(self.export_plot)
        self.perform_fft()
    def perform_fft(self):
        N = len(self.signal)
        T = 1.0 / self.sample_rate
        yf = fft(self.signal)
        xf = fftfreq(N, T)[:N // 2]
        self.fft_plot.plot(xf, 2.0 / N * np.abs(yf[:N // 2]), pen=pg.mkPen('#00FFFF', width=2), name='FFT')
    def export_plot(self):
        dialog = ExportTypeDialog(self)
        if dialog.exec_() == QDialog.Accepted:
            export_type = dialog.combo.currentText()
            if export_type == "Raster (PNG)":
                self.export_raster_plot()
            elif export_type == "Vector (SVG)":
                self.export_vector_plot()
    def export_raster_plot(self):
        file_name, _ = QFileDialog.getSaveFileName(
            self, "Сохранить график", "",
            "PNG Images (*.png);;All Files (*)"
        )
        if not file_name:
            return
        import pyqtgraph.exporters as exporters
        exporter = exporters.ImageExporter(self.fft_plot.plotItem)
        exporter.parameters()['width'] = 1920
        exporter.export(file_name)
    def export_vector_plot(self):
        file_name, _ = QFileDialog.getSaveFileName(
            self, "Сохранить график как SVG", "",
            "SVG Files (*.svg);;All Files (*)"
        )
        if not file_name:
            return
        import pyqtgraph.exporters as exporters
        exporter = exporters.SVGExporter(self.fft_plot.plotItem)
        exporter.export(file_name)

# ----- Главное окно приложения -----
class AdvancedInterferometerApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Multi-Beam Interferometer Analyzer")
        self.setGeometry(100, 50, 1280, 800)
        self.setWindowIcon(QIcon('interferometer_icon.png'))
        # Стиль главного окна и QTabWidget (фон QTabWidget — тёмно-серый, как и фон программы)
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
        self.left_layout = QVBoxLayout(self.left_panel)
        self.main_layout.addWidget(self.left_panel)
        # QTabWidget создаётся сразу с дефолтной вкладкой Demodulation
        self.tab_widget = QTabWidget()
        self.tab_widget.setTabPosition(QTabWidget.West)
        self.main_layout.addWidget(self.tab_widget)
        self.peaks_visible = False
        self.data = None
        self.current_channel = 0
        self.processed_data = None
        self.demod_widget = DemodulationWidget(np.array([]), 1000.0)
        self.tab_widget.addTab(self.demod_widget, "Demodulation")
        self.init_data_controls()
        self.init_processing_controls()
        self.init_analysis_tools()
    def init_data_controls(self):
        group = QGroupBox("Data Management")
        layout = QVBoxLayout(group)
        self.btn_load_data = FileDropButton("Load Data File")
        self.btn_load_data.setIcon(QIcon('icons/load_icon.png'))
        self.btn_load_data.clicked.connect(self.load_data)
        self.btn_load_data.fileDropped.connect(self.load_data_dropped)
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
    def init_analysis_tools(self):
        group = QGroupBox("Analysis Tools")
        layout = QVBoxLayout(group)
        self.btn_fft = QPushButton("FFT Analysis")
        self.btn_fft.clicked.connect(self.fft_analysis)
        self.btn_peaks = QPushButton("Find Peaks")
        self.btn_peaks.clicked.connect(self.find_peaks)
        self.btn_demod = QPushButton("Demodulation")
        self.btn_demod.clicked.connect(self.open_demodulation_widget)
        layout.addWidget(self.btn_fft)
        layout.addWidget(self.btn_peaks)
        layout.addWidget(self.btn_demod)
        self.left_layout.addWidget(group)
    def open_file(self, file_path):
        try:
            self.statusBar().showMessage("Loading data...")
            QApplication.processEvents()
            if file_path.lower().endswith('.csv'):
                self.data = pd.read_csv(file_path)
            elif file_path.lower().endswith('.xlsx'):
                self.data = pd.read_excel(file_path)
            elif file_path.lower().endswith('.txt'):
                try:
                    self.data = pd.read_csv(file_path, delim_whitespace=True, header=None)
                except Exception:
                    arr = np.loadtxt(file_path)
                    self.data = pd.DataFrame(arr)
            else:
                self.data = pd.read_csv(file_path)
            if self.data.empty:
                raise ValueError("File is empty")
            if not np.issubdtype(self.data.iloc[:, 0].dtype, np.number):
                raise ValueError("Non-numeric data detected")
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
        self.channel_selector.clear()
        self.channel_selector.addItems(self.data.columns.astype(str))
        self.data_info.setText(
            f"Channels: {len(self.data.columns)}\n"
            f"Samples: {len(self.data)}\n"
            f"Type: {self.data.iloc[:, 0].dtype}"
        )
        # Обновляем данные во вкладке Demodulation с sampling frequency 1000 Hz
        signal = self.data.iloc[:, self.current_channel].values
        self.demod_widget.update_signal(signal, 1000.0)
    def update_channel(self, index):
        self.current_channel = index
        if self.data is not None:
            signal = self.data.iloc[:, self.current_channel].values
            self.demod_widget.update_signal(signal, 1000.0)
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
    def fft_analysis(self):
        if self.data is not None:
            sample_rate, ok = QInputDialog.getDouble(
                self, "Sampling Rate",
                "Enter sampling frequency (Hz):",
                value=1000.0, min=0.1, max=100000.0
            )
            if ok:
                for i in range(self.tab_widget.count()):
                    if self.tab_widget.tabText(i) == "FFT Analysis":
                        self.tab_widget.setCurrentIndex(i)
                        return
                signal = self.data.iloc[:, self.current_channel].values
                fft_widget = FFTAnalysisWidget(signal, sample_rate)
                self.tab_widget.addTab(fft_widget, "FFT Analysis")
                self.tab_widget.setCurrentWidget(fft_widget)
    def find_peaks(self):
        if self.data is None:
            return
        try:
            data = self.processed_data if self.processed_data is not None else self.data.iloc[:, self.current_channel].values
            peaks, _ = find_peaks(data, prominence=1)
            self.peaks_visible = True
            self.btn_peaks.setText("Hide Peaks")
            self.statusBar().showMessage(f"Found {len(peaks)} peaks")
        except Exception as e:
            self.statusBar().showMessage(f"Peak Finding Error: {str(e)}")
    def export_plot(self):
        dialog = ExportTypeDialog(self)
        if dialog.exec_() == QDialog.Accepted:
            export_type = dialog.combo.currentText()
            if export_type == "Raster (PNG)":
                self.export_raster_plot()
            elif export_type == "Vector (SVG)":
                self.export_vector_plot()
    def export_raster_plot(self):
        file_name, _ = QFileDialog.getSaveFileName(
            self, "Сохранить график", "",
            "PNG Images (*.png);;All Files (*)"
        )
        if not file_name:
            return
        self.statusBar().showMessage(f"График сохранён в {file_name}")
    def export_vector_plot(self):
        file_name, _ = QFileDialog.getSaveFileName(
            self, "Сохранить график как SVG", "",
            "SVG Files (*.svg);;All Files (*)"
        )
        if not file_name:
            return
        self.statusBar().showMessage(f"График сохранён в {file_name} (векторный формат)")
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
    # ----- Методы для открытия вкладок по клику в Demodulation -----
    def openOriginalSignalTab(self, signal, sampling_rate):
        widget = OriginalSignalWidget(signal, sampling_rate)
        self.tab_widget.addTab(widget, "Original Signal")
        self.tab_widget.setCurrentWidget(widget)
    def openSpectrumTab(self, demodulated_signal, sampling_rate):
        widget = SpectrumWidget(demodulated_signal, sampling_rate, zero_padding=0)
        self.tab_widget.addTab(widget, "Spectrum of Demodulated Signal")
        self.tab_widget.setCurrentWidget(widget)
    def openPhaseTab(self, time, phase):
        widget = PhaseWidget(time, phase)
        self.tab_widget.addTab(widget, "Phase vs Time")
        self.tab_widget.setCurrentWidget(widget)
    def open_demodulation_widget(self):
        # Теперь вкладка Demodulation всегда присутствует, поэтому просто переключаемся на неё
        self.tab_widget.setCurrentWidget(self.demod_widget)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setFont(QFont("Helvetica", 10))
    window = AdvancedInterferometerApp()
    window.show()
    sys.exit(app.exec_())
d