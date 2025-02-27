import sys
import numpy as np
import pandas as pd
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QWidget,
                             QPushButton, QLabel, QLineEdit, QFileDialog, QComboBox,
                             QTabWidget, QGroupBox, QTableWidget, QTableWidgetItem,
                             QHeaderView, QStatusBar)
from PyQt5.QtGui import QFont, QIcon
from PyQt5.QtCore import Qt, QTimer
import pyqtgraph as pg
from scipy.signal import savgol_filter
from scipy.signal.windows import gaussian
from scipy.fft import fft, fftfreq
import os
from PyQt5.QtWidgets import QInputDialog

class AdvancedInterferometerApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Multi-Beam Interferometer Analyzer")
        self.setGeometry(100, 50, 1280, 800)
        self.setWindowIcon(QIcon('interferometer_icon.png'))

        # Стилизация интерфейса
        self.setStyleSheet("""
            QMainWindow {
                background-color: #2E2E2E;
            }
            QGroupBox {
                background-color: #404040;
                border: 2px solid #606060;
                border-radius: 5px;
                margin-top: 1ex;
                color: #FFFFFF;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 3px;
                color: #00B4FF;
            }
            QPushButton {
                background-color: #505050;
                border: 1px solid #606060;
                border-radius: 3px;
                color: #FFFFFF;
                padding: 5px;
                min-width: 80px;
            }
            QPushButton:hover {
                background-color: #606060;
            }
            QLineEdit, QComboBox {
                background-color: #353535;
                border: 1px solid #505050;
                color: #FFFFFF;
                padding: 3px;
            }
            QTableWidget {
                background-color: #353535;
                color: #FFFFFF;
                gridline-color: #505050;
            }
            }
            QTabWidget::pane {          # <-- ДОБАВИТЬ ЭТОТ БЛОК
                border: 1px solid #606060;
                margin: 2px;
            }
            QTabBar::tab {              # <-- ДОБАВИТЬ ЭТОТ БЛОК
                background: #404040;
                color: #FFFFFF;
                padding: 8px 15px;
                border-top-left-radius: 5px;
                border-top-right-radius: 5px;
            }
            QTabBar::tab:selected {     # <-- ДОБАВИТЬ ЭТОТ БЛОК
                background: #505050;
            }
            QHeaderView::section {
                background-color: #404040;
                color: #FFFFFF;
                border: 1px solid #505050;
            }
        """)

        # Основной виджет и layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QHBoxLayout(self.central_widget)

        # Левая панель инструментов
        self.left_panel = QWidget()
        self.left_panel.setFixedWidth(300)
        self.left_layout = QVBoxLayout(self.left_panel)
        self.main_layout.addWidget(self.left_panel)

        # Правая панель с вкладками
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
        # Данные
        self.data = None
        self.current_channel = 0
        self.processed_data = None

    def init_data_controls(self):
        """Панель управления данными"""
        group = QGroupBox("Data Management")
        layout = QVBoxLayout(group)

        # Кнопки загрузки
        self.btn_load_csv = QPushButton("Load CSV/Excel")
        self.btn_load_csv.setIcon(QIcon('icons/load_icon.png'))
        self.btn_load_csv.clicked.connect(self.load_data)

        # Выбор канала
        self.channel_selector = QComboBox()
        self.channel_selector.currentIndexChanged.connect(self.update_channel)

        # Информация о данных
        self.data_info = QLabel("No data loaded")
        self.data_info.setAlignment(Qt.AlignCenter)
        self.data_info.setStyleSheet("color: #AAAAAA;")

        layout.addWidget(self.btn_load_csv)
        layout.addWidget(QLabel("Select Channel:"))
        layout.addWidget(self.channel_selector)
        layout.addWidget(self.data_info)
        self.left_layout.addWidget(group)

    def init_processing_controls(self):
        """Панель обработки данных"""
        group = QGroupBox("Signal Processing")
        layout = QVBoxLayout(group)

        # Параметры фильтра
        self.filter_type = QComboBox()
        self.filter_type.addItems(['Moving Average', 'Gaussian', 'Savitzky-Golay'])

        self.filter_param = QLineEdit('5')
        self.filter_param.setPlaceholderText('Window size')

        # Кнопка обработки
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

        # График
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground('#202020')
        self.plot_widget.addLegend()
        self.plot_widget.showGrid(x=True, y=True)
        self.plot_widget.setLabel('left', 'Intensity', units='dB')
        self.plot_widget.setLabel('bottom', 'Sample Index')

        self.raw_plot = self.plot_widget.plot(pen=pg.mkPen('#00FF00', width=2), name='Raw Data')
        self.processed_plot = self.plot_widget.plot(pen=pg.mkPen('#FF0000', width=2), name='Processed')

        # Панель инструментов графика
        self.plot_toolbar = QWidget()
        plot_tools = QHBoxLayout(self.plot_toolbar)

        self.btn_zoom = QPushButton("Zoom")
        self.btn_pan = QPushButton("Pan")
        self.btn_export_plot = QPushButton("Export Plot")

        plot_tools.addWidget(self.btn_zoom)
        plot_tools.addWidget(self.btn_pan)
        plot_tools.addWidget(self.btn_export_plot)

        layout.addWidget(self.plot_widget)
        layout.addWidget(self.plot_toolbar)
        self.tab_widget.addTab(tab, "Visualization")

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
        """Статус бар"""
        self.statusBar().showMessage("Ready")
        self.statusBar().setStyleSheet("color: #00FF00; background-color: #404040;")

    def load_data(self):
        """Загрузка данных"""
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Open Data File", "",
            "CSV Files (*.csv);;Excel Files (*.xlsx);;All Files (*)"
        )

        if file_name:
            try:
                if file_name.endswith('.csv'):
                    self.data = pd.read_csv(file_name)
                else:
                    self.data = pd.read_excel(file_name)

                self.update_interface()
                self.statusBar().showMessage(f"Loaded: {file_name}")

            except Exception as e:
                self.statusBar().showMessage(f"Error: {str(e)}")

    def update_interface(self):
        """Обновление интерфейса после загрузки данных"""
        # Обновление селектора каналов
        self.channel_selector.clear()
        self.channel_selector.addItems(self.data.columns)

        # Обновление информации о данных
        self.data_info.setText(
            f"Channels: {len(self.data.columns)}\n"
            f"Samples: {len(self.data)}\n"
            f"Type: {self.data.iloc[:, 0].dtype}"
        )

        # Обновление таблицы
        self.update_table()

        # Обновление графика
        self.update_plot()

    def update_channel(self, index):
        """Обновление выбранного канала"""
        self.current_channel = index
        self.update_plot()
        self.update_table()

    def update_plot(self):
        """Обновление графика"""
        if self.data is not None:
            y = self.data.iloc[:, self.current_channel].values
            self.raw_plot.setData(y)

            if self.processed_data is not None:
                self.processed_plot.setData(self.processed_data)
            else:
                self.processed_plot.setData([])

    def update_table(self):
        """Обновление таблицы данных"""
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
        """Анализ Фурье"""
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

        # Кнопки анализа
        self.btn_fft = QPushButton("FFT Analysis")
        self.btn_fft.clicked.connect(self.fft_analysis)

        self.btn_peaks = QPushButton("Find Peaks")
        self.btn_peaks.clicked.connect(self.find_peaks)

        layout.addWidget(self.btn_fft)
        layout.addWidget(self.btn_peaks)
        self.left_layout.addWidget(group)

    def find_peaks(self):
        """Поиск пиков в данных"""
        if self.data is not None:
            try:
                from scipy.signal import find_peaks
                data = self.processed_data if self.processed_data is not None else self.data.iloc[:,
                                                                                   self.current_channel].values

                peaks, _ = find_peaks(data, prominence=1)
                self.plot_widget.clear()
                self.raw_plot = self.plot_widget.plot(data, pen='#00FF00', name='Data')
                self.peaks_plot = self.plot_widget.plot(peaks, data[peaks], pen=None, symbol='x', symbolSize=10,
                                                        symbolBrush='#FF00FF', name='Peaks')

                self.statusBar().showMessage(f"Found {len(peaks)} peaks")

            except Exception as e:
                self.statusBar().showMessage(f"Peak Finding Error: {str(e)}")

    def load_data(self):
        """Загрузка данных с улучшенной обработкой ошибок"""
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Open Data File", "",
            "CSV Files (*.csv);;Excel Files (*.xlsx);;All Files (*)"
        )

        if file_name:
            try:
                # Показать индикатор загрузки
                self.statusBar().showMessage("Loading data...")
                QApplication.processEvents()

                # Чтение файла
                if file_name.endswith('.csv'):
                    self.data = pd.read_csv(file_name)
                else:
                    self.data = pd.read_excel(file_name)

                # Проверка данных
                if self.data.empty:
                    raise ValueError("File is empty")
                if not np.issubdtype(self.data.iloc[:, 0].dtype, np.number):
                    raise ValueError("Non-numeric data detected")

                self.update_interface()
                self.statusBar().showMessage(f"Loaded: {os.path.basename(file_name)}")

            except Exception as e:
                self.statusBar().showMessage(f"Error: {str(e)}")
                self.data = None
                self.processed_data = None
                self.update_interface()

    def init_visualization_tab(self):
        """Вкладка визуализации"""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # График
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground('#202020')
        self.plot_widget.addLegend()
        self.plot_widget.showGrid(x=True, y=True)
        self.plot_widget.setLabel('left', 'Intensity', units='dB')
        self.plot_widget.setLabel('bottom', 'Sample Index')

        self.raw_plot = self.plot_widget.plot(pen=pg.mkPen('#00FF00', width=2), name='Raw Data')
        self.processed_plot = self.plot_widget.plot(pen=pg.mkPen('#FF0000', width=2), name='Processed')

        # Панель инструментов графика
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

        # Инициализация обработчиков
        self.btn_zoom.clicked.connect(lambda: self.plot_widget.getViewBox().setMouseMode(3))
        self.btn_pan.clicked.connect(lambda: self.plot_widget.getViewBox().setMouseMode(1))
        self.btn_export_plot.clicked.connect(self.export_plot)
        self.btn_compare.clicked.connect(self.toggle_channel_comparison)


    def toggle_channel_comparison(self):
        """Включение/выключение сравнения каналов"""
        if self.data is not None:
            from PyQt5.QtWidgets import QInputDialog
            items, ok = QInputDialog.getMultiChoice(
                self, "Select Channels",
                "Choose channels to compare:",
                self.data.columns.tolist()
            )

            if ok and items:
                self.plot_widget.clear()
                colors = ['#00FF00', '#FF0000', '#00FFFF', '#FF00FF']

                for i, col in enumerate(items):
                    y = self.data[col].values
                    self.plot_widget.plot(y, pen=colors[i % len(colors)], name=col)

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
if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setFont(QFont("Helvetica", 10))
    window = AdvancedInterferometerApp()
    window.show()
    sys.exit(app.exec_())