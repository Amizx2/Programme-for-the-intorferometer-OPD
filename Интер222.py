import sys
import numpy as np
import pandas as pd
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QWidget,
                             QPushButton, QLabel, QLineEdit, QFileDialog, QComboBox,
                             QTabWidget, QGroupBox, QTableWidget, QTableWidgetItem,
                             QHeaderView, QStatusBar, QInputDialog)
from PyQt5.QtGui import QFont, QIcon
from PyQt5.QtCore import Qt, pyqtSignal
import pyqtgraph as pg
from scipy.signal import savgol_filter, find_peaks, hilbert
from scipy.signal.windows import gaussian
from scipy.fft import fft, fftfreq
import os

# Пользовательский виджет для drag-and-drop загрузки файлов
class FileDropButton(QPushButton):
    # Сигнал, передающий путь к перетянутому файлу
    fileDropped = pyqtSignal(str)
    
    def __init__(self, text, parent=None):
        super().__init__(text, parent)
        self.setAcceptDrops(True)
        self.default_text = text  # Исходный текст кнопки
    
    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            if urls:
                file_path = urls[0].toLocalFile()
                # Проверяем, что расширение файла допустимо
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
                # Излучаем сигнал с путем к файлу
                self.fileDropped.emit(file_path)
                event.acceptProposedAction()
                return
        event.ignore()

# Окно демодуляции с элементами управления и графиками
class DemodulationWindow(QMainWindow):
    def __init__(self, signal, sampling_rate=1000.0):
        super().__init__()
        self.setWindowTitle("Demodulation Analysis")
        self.setGeometry(150, 150, 900, 700)
        self.signal = signal
        self.sampling_rate = sampling_rate
        self.init_ui()

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # --- Блок управления добавлением нулей ---
        zero_pad_layout = QHBoxLayout()

        zero_label = QLabel("Number of Zeros:")
        zero_label.setStyleSheet("color: #FFFFFF;")

        self.zero_input = QLineEdit("0")
        self.zero_input.setStyleSheet("background-color: #353535; color: #FFFFFF; padding: 3px;")

        self.btn_update = QPushButton("Update Spectrum")
        self.btn_update.setStyleSheet("""
            background-color: #00B4FF;
            font-size: 12px;
            font-weight: bold;
            padding: 5px;
        """)
        self.btn_update.clicked.connect(self.update_spectrum)

        zero_pad_layout.addWidget(zero_label)
        zero_pad_layout.addWidget(self.zero_input, 1)
        zero_pad_layout.addWidget(self.btn_update, 1)

        main_layout.addLayout(zero_pad_layout)

        # Расчет аналитического сигнала для демодуляции
        analytic_signal = hilbert(self.signal)
        self.demodulated_signal = np.abs(analytic_signal)
        self.instantaneous_phase = np.unwrap(np.angle(analytic_signal))
        self.time = np.arange(len(self.signal)) / self.sampling_rate

        # --- График: Исходный сигнал ---
        self.plot_original = pg.PlotWidget(title="Original Signal")
        self.plot_original.setBackground('#202020')
        self.plot_original.plot(self.signal, pen=pg.mkPen('#00FF00', width=2))
        self.plot_original.setLabel('left', 'Amplitude')
        self.plot_original.setLabel('bottom', 'Sample Index')
        self.plot_original.showGrid(x=True, y=True)
        main_layout.addWidget(self.plot_original)

        # --- График: Спектр демодулированного сигнала ---
        self.plot_spectrum = pg.PlotWidget(title="Spectrum of Demodulated Signal")
        self.plot_spectrum.setBackground('#202020')
        self.plot_spectrum.setLabel('left', 'Amplitude')
        self.plot_spectrum.setLabel('bottom', 'Frequency (Hz)')
        self.plot_spectrum.showGrid(x=True, y=True)
        main_layout.addWidget(self.plot_spectrum)

        # --- График: Фаза от времени ---
        self.plot_phase = pg.PlotWidget(title="Phase vs Time")
        self.plot_phase.setBackground('#202020')
        self.plot_phase.plot(self.time, self.instantaneous_phase, pen=pg.mkPen('#00FFFF', width=2))
        self.plot_phase.setLabel('left', 'Phase (radians)')
        self.plot_phase.setLabel('bottom', 'Time (s)')
        self.plot_phase.showGrid(x=True, y=True)
        main_layout.addWidget(self.plot_phase)

        # Построение спектра (без добавления нулей)
        self.update_spectrum()

    def update_spectrum(self):
        try:
            n_zeros = int(self.zero_input.text())
        except ValueError:
            n_zeros = 0

        padded_signal = np.pad(self.demodulated_signal, (0, n_zeros), 'constant')
        N = len(padded_signal)
        T = 1.0 / self.sampling_rate
        yf = fft(padded_signal)
        xf = fftfreq(N, T)[:N // 2]
        amplitude = 2.0 / N * np.abs(yf[:N // 2])

        self.plot_spectrum.clear()
        self.plot_spectrum.plot(xf, amplitude, pen=pg.mkPen('#00FFFF', width=2))

# Главное окно приложения
class AdvancedInterferometerApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Multi-Beam Interferometer Analyzer")
        self.setGeometry(100, 50, 1280, 800)
        self.setWindowIcon(QIcon('interferometer_icon.png'))

        self.setStyleSheet("""
            QMainWindow { background-color: #2E2E2E; }
            QGroupBox { background-color: #404040; border: 2px solid #606060; border-radius: 5px; margin-top: 1ex; color: #FFFFFF; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 3px; color: #00B4FF; }
            QPushButton { background-color: #505050; border: 1px solid #606060; border-radius: 3px; color: #FFFFFF; padding: 5px; min-width: 80px; }
            QPushButton:hover { background-color: #606060; }
            QLineEdit, QComboBox { background-color: #353535; border: 1px solid #505050; color: #FFFFFF; padding: 3px; }
            QTableWidget { background-color: #353535; color: #FFFFFF; gridline-color: #505050; }
            QTabWidget::pane { border: 1px solid #606060; margin: 2px; }
            QTabBar::tab { background: #404040; color: #FFFFFF; padding: 8px 15px; border-top-left-radius: 5px; border-top-right-radius: 5px; }
            QTabBar::tab:selected { background: #505050; }
            QHeaderView::section { background-color: #404040; color: #FFFFFF; border: 1px solid #505050; }
        """)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QHBoxLayout(self.central_widget)

        self.left_panel = QWidget()
        self.left_panel.setFixedWidth(300)
        self.left_layout = QVBoxLayout(self.left_panel)
        self.main_layout.addWidget(self.left_panel)

        self.tab_widget = QTabWidget()
        self.tab_widget.setTabPosition(QTabWidget.West)
        self.main_layout.addWidget(self.tab_widget)

        # Инициализация компонентов
        self.init_data_controls()
        self.init_processing_controls()
        self.init_visualization_tab()
        self.init_raw_data_tab()
        self.init_status_bar()
        self.init_analysis_controls()
        
        self.data = None
        self.current_channel = 0
        self.processed_data = None

    def init_data_controls(self):
        """Панель управления данными"""
        group = QGroupBox("Data Management")
        layout = QVBoxLayout(group)

        # Используем FileDropButton для поддержки drag-and-drop
        self.btn_load_data = FileDropButton("Load Data File")
        self.btn_load_data.setIcon(QIcon('icons/load_icon.png'))
        # При клике – открывается диалог выбора файла
        self.btn_load_data.clicked.connect(self.load_data)
        # При перетаскивании файла – открывается файл аналогично кнопке
        self.btn_load_data.fileDropped.connect(self.load_data_dropped)

        self.channel_selector = QComboBox()
        self.channel_selector.currentIndexChanged.connect(self.update_channel)

        self.data_info = QLabel("No data loaded")
        self.data_info.setAlignment(Qt.AlignCenter)
        self.data_info.setStyleSheet("color: #AAAAAA;")

        layout.addWidget(self.btn_load_data)
        layout.addWidget(QLabel("Select Channel:"))
        layout.addWidget(self.channel_selector)
        layout.addWidget(self.data_info)
        self.left_layout.addWidget(group)

    def init_processing_controls(self):
        """Панель обработки сигнала"""
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

    def init_visualization_tab(self):
        """Вкладка визуализации"""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground('#202020')
        self.plot_widget.addLegend()
        self.plot_widget.showGrid(x=True, y=True)
        self.plot_widget.setLabel('left', 'Intensity', units='dB')
        self.plot_widget.setLabel('bottom', 'Sample Index')

        self.raw_plot = self.plot_widget.plot(pen=pg.mkPen('#00FF00', width=2), name='Raw Data')
        self.processed_plot = self.plot_widget.plot(pen=pg.mkPen('#FF0000', width=2), name='Processed')

        self.plot_toolbar = QWidget()
        plot_tools = QHBoxLayout(self.plot_toolbar)

        self.btn_zoom = QPushButton("Zoom")
        self.btn_pan = QPushButton("Pan")
        self.btn_compare = QPushButton("Compare Channels")
        self.btn_export_plot = QPushButton("Export Plot")

        plot_tools.addWidget(self.btn_zoom)
        plot_tools.addWidget(self.btn_pan)
        plot_tools.addWidget(self.btn_compare)
        plot_tools.addWidget(self.btn_export_plot)

        layout.addWidget(self.plot_widget)
        layout.addWidget(self.plot_toolbar)
        self.tab_widget.addTab(tab, "Visualization")

        self.btn_zoom.clicked.connect(lambda: self.plot_widget.getViewBox().setMouseMode(3))
        self.btn_pan.clicked.connect(lambda: self.plot_widget.getViewBox().setMouseMode(1))
        self.btn_export_plot.clicked.connect(self.export_plot)
        self.btn_compare.clicked.connect(self.toggle_channel_comparison)

    def init_raw_data_tab(self):
        """Вкладка с сырыми данными"""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        self.data_table = QTableWidget()
        self.data_table.setColumnCount(2)
        self.data_table.setHorizontalHeaderLabels(['Raw Data', 'Processed'])
        self.data_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

        layout.addWidget(self.data_table)
        self.tab_widget.addTab(tab, "Raw Data")

    def init_status_bar(self):
        """Строка состояния"""
        self.statusBar().showMessage("Ready")
        self.statusBar().setStyleSheet("color: #00FF00; background-color: #404040;")

    def open_file(self, file_path):
        """Общий метод открытия файла"""
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
        """Загрузка данных через диалог выбора файла"""
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Open Data File", "",
            "CSV Files (*.csv);;Excel Files (*.xlsx);;Text Files (*.txt);;All Files (*)"
        )

        if file_name:
            self.open_file(file_name)

    def load_data_dropped(self, file_path):
        """Загрузка данных из перетянутого файла"""
        self.open_file(file_path)

    def update_interface(self):
        """Обновление интерфейса после загрузки данных"""
        self.channel_selector.clear()
        self.channel_selector.addItems(self.data.columns.astype(str))
        self.data_info.setText(
            f"Channels: {len(self.data.columns)}\n"
            f"Samples: {len(self.data)}\n"
            f"Type: {self.data.iloc[:, 0].dtype}"
        )
        self.update_table()
        self.update_plot()

    def update_channel(self, index):
        """Обновление выбранного канала"""
        self.current_channel = index
        self.update_plot()
        self.update_table()

    def update_plot(self):
        """Обновление графика на вкладке визуализации"""
        if self.data is not None:
            y = self.data.iloc[:, self.current_channel].values
            self.raw_plot.setData(y)
            if self.processed_data is not None:
                self.processed_plot.setData(self.processed_data)
            else:
                self.processed_plot.setData([])

    def update_table(self):
        """Обновление таблицы с данными"""
        if self.data is not None:
            self.data_table.setRowCount(len(self.data))
            col_data = self.data.iloc[:, self.current_channel]
            processed = self.processed_data if self.processed_data is not None else [None] * len(col_data)
            for i, (raw_val, proc_val) in enumerate(zip(col_data, processed)):
                self.data_table.setItem(i, 0, QTableWidgetItem(f"{raw_val:.6f}"))
                item = QTableWidgetItem(f"{proc_val:.6f}") if proc_val is not None else QTableWidgetItem("")
                self.data_table.setItem(i, 1, item)

    def apply_filter(self):
        """Применение фильтра к данным"""
        try:
            window = int(self.filter_param.text())
            raw = self.data.iloc[:, self.current_channel].values
            if self.filter_type.currentText() == 'Moving Average':
                kernel = np.ones(window) / window
                self.processed_data = np.convolve(raw, kernel, mode='same')
            self.update_plot()
            self.update_table()
            self.statusBar().showMessage("Filter applied successfully")
        except Exception as e:
            self.statusBar().showMessage(f"Filter Error: {str(e)}")

    def fft_analysis(self):
        """Анализ FFT"""
        if self.data is not None:
            sample_rate, ok = QInputDialog.getDouble(
                self, "Sampling Rate",
                "Enter sampling frequency (Hz):",
                value=1000.0, min=0.1, max=100000.0
            )
            if ok:
                signal = self.data.iloc[:, self.current_channel].values
                N = len(signal)
                T = 1.0 / sample_rate
                yf = fft(signal)
                xf = fftfreq(N, T)[:N // 2]
                self.plot_widget.clear()
                self.plot_widget.plot(xf, 2.0 / N * np.abs(yf[0:N // 2]), pen='#00FFFF', name='FFT')
                self.plot_widget.setLabel('left', 'Amplitude')
                self.plot_widget.setLabel('bottom', 'Frequency (Hz)')

    def init_analysis_controls(self):
        """Панель расширенного анализа"""
        group = QGroupBox("Advanced Analysis")
        layout = QVBoxLayout(group)

        self.btn_fft = QPushButton("FFT Analysis")
        self.btn_fft.clicked.connect(self.fft_analysis)

        self.btn_peaks = QPushButton("Find Peaks")
        self.btn_peaks.clicked.connect(self.find_peaks)

        # Кнопка для открытия окна демодуляции с управлением добавлением нулей
        self.btn_demod = QPushButton("Demodulation Window")
        self.btn_demod.setStyleSheet("""
            background-color: #00B4FF;
            font-size: 14px;
            font-weight: bold;
            padding: 10px;
        """)
        self.btn_demod.setMinimumHeight(30)
        self.btn_demod.clicked.connect(self.open_demodulation_window)

        layout.addWidget(self.btn_fft)
        layout.addWidget(self.btn_peaks)
        layout.addWidget(self.btn_demod)
        self.left_layout.addWidget(group)

    def open_demodulation_window(self):
        """Открытие окна демодуляции"""
        if self.data is not None:
            signal = self.data.iloc[:, self.current_channel].values
            sampling_rate, ok = QInputDialog.getDouble(
                self, "Sampling Rate",
                "Enter sampling frequency (Hz):",
                value=1000.0, min=0.1, max=100000.0
            )
            if ok:
                self.demod_window = DemodulationWindow(signal, sampling_rate)
                self.demod_window.show()
        else:
            self.statusBar().showMessage("No data loaded for demodulation.")

    def find_peaks(self):
        """Нахождение пиков в данных"""
        if self.data is not None:
            try:
                data = self.processed_data if self.processed_data is not None else self.data.iloc[:, self.current_channel].values
                peaks, _ = find_peaks(data, prominence=1)
                self.plot_widget.clear()
                self.raw_plot = self.plot_widget.plot(data, pen='#00FF00', name='Data')
                self.peaks_plot = self.plot_widget.plot(peaks, data[peaks], pen=None, symbol='x', symbolSize=10,
                                                        symbolBrush='#FF00FF', name='Peaks')
                self.statusBar().showMessage(f"Found {len(peaks)} peaks")
            except Exception as e:
                self.statusBar().showMessage(f"Peak Finding Error: {str(e)}")

    def export_plot(self):
        """Экспорт графика в PNG"""
        file_name, _ = QFileDialog.getSaveFileName(
            self, "Save Plot", "",
            "PNG Images (*.png);;All Files (*)"
        )
        if file_name:
            exporter = pg.exporters.ImageExporter(self.plot_widget.plotItem)
            exporter.export(file_name)
            self.statusBar().showMessage(f"Plot saved to {file_name}")

    def toggle_channel_comparison(self):
        """Режим сравнения каналов"""
        if self.data is not None:
            items, ok = QInputDialog.getMultiChoice(
                self, "Select Channels",
                "Choose channels to compare:",
                self.data.columns.astype(str).tolist()
            )
            if ok and items:
                self.plot_widget.clear()
                colors = ['#00FF00', '#FF0000', '#00FFFF', '#FF00FF']
                for i, col in enumerate(items):
                    y = self.data[col].values
                    self.plot_widget.plot(y, pen=colors[i % len(colors)], name=col)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setFont(QFont("Helvetica", 10))
    window = AdvancedInterferometerApp()
    window.show()
    sys.exit(app.exec_())
