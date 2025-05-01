import os
import sys

import numpy as np
import pandas as pd
import pyqtgraph as pg
import pyqtgraph.exporters as exporters
from PyQt5.QtCore import QEasingCurve, QTimer
from PyQt5.QtCore import QPointF
from PyQt5.QtCore import Qt, pyqtSignal, QAbstractTableModel, QPropertyAnimation
from PyQt5.QtGui import QColor
from PyQt5.QtGui import QCursor, QFont, QIcon
from PyQt5.QtGui import QStandardItemModel, QStandardItem
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QVBoxLayout,
    QWidget,
    QGraphicsDropShadowEffect,
    QDialog,
    QProgressBar,
)
from PyQt5.QtWidgets import (
    QDialogButtonBox,
)
from PyQt5.QtWidgets import (
    QHBoxLayout,
    QPushButton,
    QLabel,
    QLineEdit,
    QFileDialog,
    QComboBox,
    QTabWidget,
    QGroupBox,
    QTableView,
    QInputDialog,
    QTabBar,
    QGraphicsOpacityEffect,
)
from scipy.fft import fft, fftfreq
from scipy.signal import hilbert


# Добавляем новый класс для загрузочного окна
class LoadingWindow(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Loading...")
        self.setFixedSize(400, 150)
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setAttribute(Qt.WA_TranslucentBackground)

        # Основной контейнер
        self.container = QWidget(self)
        self.container.setGeometry(0, 0, 400, 150)
        self.container.setStyleSheet(
            """
            background: qlineargradient(
                x1:0, y1:0, x2:1, y2:0,
                stop:0 #2c3e50, stop:1 #3498db
            );
            border-radius: 15px;
            border: 1px solid rgba(255, 255, 255, 30);
        """
        )

        # Эффект тени
        shadow = QGraphicsDropShadowEffect(self.container)
        shadow.setBlurRadius(20)
        shadow.setColor(QColor(0, 0, 0, 100))
        shadow.setOffset(3, 3)
        self.container.setGraphicsEffect(shadow)

        layout = QVBoxLayout(self.container)
        layout.setContentsMargins(40, 40, 40, 40)
        layout.setSpacing(20)

        # Текст загрузки
        self.loading_text = QLabel("Loading...")
        self.loading_text.setAlignment(Qt.AlignCenter)
        self.loading_text.setStyleSheet(
            """
            color: rgba(255, 255, 255, 200);
            font-size: 18px;
            font-weight: 500;
        """
        )

        # Прогресс-бар
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setTextVisible(False)
        self.progress.setStyleSheet(
            """
            QProgressBar {
                height: 8px;
                background: rgba(255, 255, 255, 50);
                border-radius: 4px;
            }
            QProgressBar::chunk {
                background: rgba(255, 255, 255, 200);
                border-radius: 4px;
            }
        """
        )

        layout.addWidget(self.loading_text)
        layout.addWidget(self.progress)

        # Анимация прогресса
        self.progress_anim = QPropertyAnimation(self.progress, b"value")
        self.progress_anim.setDuration(500)
        self.progress_anim.setStartValue(0)
        self.progress_anim.setEndValue(100)
        self.progress_anim.setEasingCurve(QEasingCurve.OutQuad)

    def showEvent(self, event):
        self.progress_anim.start()
        super().showEvent(event)

    def closeEvent(self, event):
        self.progress_anim.stop()
        super().closeEvent(event)


class DragOverlay(QWidget):
    """Полупрозрачный оверлей на весь экран с пунктирной границей и надписью по центру."""

    def __init__(self, parent=None):
        super().__init__(parent)
        # Разрешаем «сквозную» мышь, чтобы не мешать другим виджетам
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        # Убираем рамку окна
        self.setWindowFlags(Qt.Widget | Qt.FramelessWindowHint)

        # Фон полупрозрачный + пунктирная граница
        self.setStyleSheet(
            """
            QWidget {
                background-color: rgba(0, 0, 0, 128);
                border: 2px dashed #FFFFFF;
            }
        """
        )

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
def getStyledDouble(parent, title, label, min_val=0.001, max_val=10000):
    dialog = QInputDialog(parent)
    dialog.setWindowTitle(title)
    dialog.setLabelText(label)

    # Жёстко задаём 1000.0 как дефолт

    dialog.setDoubleValue(100)
    dialog.setDoubleRange(min_val, max_val)
    dialog.setDoubleDecimals(6)

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


# ----- Виджет Raw Data с QTableView и белым шрифтом -----
class RawDataWidget(QWidget):
    def __init__(self, data, parent=None):
        super().__init__(parent)
        self.data = data
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        self.table = QTableView()
        self.table.setStyleSheet(
            """
            QTableView { background-color: #2E2E2E; color: #FFFFFF; }
            QHeaderView::section { background-color: #2E2E2E; color: #FFFFFF; }
        """
        )
        self.model = PandasModel(self.data)
        self.table.setModel(self.model)
        self.table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.table)

    def update_data(self, data):
        self.data = data
        self.model.update(data)


def calculate_x_axis(signal):
    if len(signal) == 0:
        return []
    lambda_start = 1410e-9
    lambda_end = 1490e-9
    wavelengths = np.linspace(lambda_start, lambda_end, len(signal))
    frequencies = 3e8 / wavelengths  # Гц
    return frequencies / 1e12  # Терагерцы


def get_export_settings(parent, title="Export Settings", default_name="plot"):
    """Возвращает (folder, base_name) или None если отменено"""
    # 1. Выбор папки
    folder = QFileDialog.getExistingDirectory(parent, "Select Folder")
    if not folder:
        return None

    name_dialog = FileNameDialog(parent, default_name)
    name_dialog.setWindowTitle(title)
    if name_dialog.exec_() != QDialog.Accepted:
        return None

    return (folder, name_dialog.name_edit.text().strip())


class FileNameDialog(QDialog):
    def __init__(self, parent=None, default_name=""):
        super().__init__(parent)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint)
        self.setWindowTitle("Enter File Name")

        # Стиль идентичный ExportTypeDialog (без спецстиля для OK)
        self.setStyleSheet(
            """
            QDialog {
                background-color: #2E2E2E;
                border: 1px solid #444;
                border-radius: 5px;
            }
            QLabel {
                color: #DDD;
                font-size: 12pt;
                padding: 5px;
            }
            QLineEdit {
                background-color: #353535;
                color: #FFF;
                border: 1px solid #555;
                border-radius: 3px;
                padding: 8px;
                selection-background-color: #00B4FF;
                font-size: 11pt;
                min-width: 200px;
            }
            QPushButton {
                background-color: #505050;
                color: #FFF;
                border: 1px solid #606060;
                border-radius: 3px;
                padding: 6px 12px;
                min-width: 80px;
                font-size: 11pt;
            }
            QPushButton:hover {
                background-color: #606060;
            }
            QPushButton:pressed {
                background-color: #404040;
            }
        """
        )

        # Основной layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(15)

        # Заголовок
        title_label = QLabel("Enter file name:")
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)

        # Поле ввода
        self.name_edit = QLineEdit(default_name)
        self.name_edit.setPlaceholderText("my_plot")
        layout.addWidget(self.name_edit)

        # Кнопки - теперь OK справа как в стандартном диалоге
        btn_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, Qt.Horizontal, self)
        btn_box.rejected.connect(self.reject)
        layout.addWidget(btn_box)

        # Эффект тени
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(10)
        shadow.setColor(QColor(0, 0, 0, 150))
        shadow.setOffset(3, 3)
        self.setGraphicsEffect(shadow)

        # Фиксированный размер
        self.setFixedSize(300, 180)
        self.name_edit.setFocus()

    def get_file_name(self):
        return self.name_edit.text().strip() if self.exec_() == QDialog.Accepted else None


class ExportTypeDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint)
        self.setWindowTitle("Export Format Selection")
        self.setWindowIcon(QIcon('icons/export_icon.png'))  # Добавьте иконку

        # Темная тема с тенями и анимацией
        self.setStyleSheet(
            """
            QDialog {
                background-color: #2E2E2E;
                border: 1px solid #444;
                border-radius: 5px;
            }
            QLabel {
                color: #DDD;
                font-size: 12pt;
                padding: 5px;
            }
            QComboBox {
                background-color: #353535;
                color: #FFF;
                border: 1px solid #555;
                border-radius: 3px;
                padding: 5px;
                min-width: 120px;
                selection-background-color: #00B4FF;
            }
            QComboBox QAbstractItemView {
                background-color: #353535;
                color: #FFF;
                selection-background-color: #00B4FF;
            }
            QPushButton {
                background-color: #505050;
                color: #FFF;
                border: 1px solid #606060;
                border-radius: 3px;
                padding: 5px 10px;
                min-width: 80px;
            }
            QPushButton:hover {
                background-color: #606060;
            }
            QPushButton:pressed {
                background-color: #404040;
            }
        """
        )

        # Основной layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(15)

        # Текст заголовка
        title_label = QLabel("Select export format:")
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)

        # Выбор формата
        self.combo = QComboBox()
        self.combo.addItems(["Raster (PNG)", "Vector (SVG)"])
        self.combo.setCurrentIndex(0)
        layout.addWidget(self.combo)

        # Кнопки
        btn_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btn_box.accepted.connect(self.accept)
        btn_box.rejected.connect(self.reject)
        layout.addWidget(btn_box)

        # Эффекты
        self.setGraphicsEffect(
            QGraphicsDropShadowEffect(blurRadius=10, color=QColor(0, 0, 0, 150), offset=QPointF(3, 3))
        )

        # Фиксируем размер
        self.setFixedSize(300, 180)


def master_export(
    parent,
    plots_data=None,
    base_name=None,
    folder=None,
    ask_settings=False,
    default_base_name="plot",
    dialog_title="Export Plots",
):
    """
    Универсальная функция экспорта графиков
    :param parent: Родительский виджет
    :param plots_data: Список словарей с графиками (если None - будет запрошен)
    :param base_name: Базовое имя файлов
    :param folder: Папка для сохранения
    :param ask_settings: Запрашивать ли папку и имя через диалог
    :param default_base_name: Имя по умолчанию
    :param dialog_title: Заголовок диалога
    """
    # Запрос настроек если требуется
    if ask_settings or not base_name or not folder:
        result = get_export_settings(parent, dialog_title, default_base_name)
        if not result:
            return False
        folder, base_name = result

    # Если данные графиков не переданы - запрашиваем у родителя
    if plots_data is None:
        if hasattr(parent, 'get_plots_to_export'):
            plots_data = parent.get_plots_to_export()
        else:
            raise ValueError("Plots data not provided and parent has no get_plots_to_export() method")

    # Диалог выбора типа экспорта
    dialog = ExportTypeDialog(parent)
    if dialog.exec_() != QDialog.Accepted:
        return False

    # Остальная логика экспорта
    export_type = dialog.combo.currentText()

    if export_type == "Raster (PNG)":
        ext = "png"
        ExporterClass = exporters.ImageExporter
        export_params = {'width': 1920}
    elif export_type == "Vector (SVG)":
        ext = "svg"
        ExporterClass = exporters.SVGExporter
        export_params = {}
    else:
        return False

    # Экспорт каждого графика
    for plot_data in plots_data:
        try:
            file_name = os.path.join(folder, f"{base_name} {plot_data['name']}.{ext}")
            exporter = ExporterClass(plot_data['plot'].plotItem)

            if export_params:
                for param, value in export_params.items():
                    exporter.parameters()[param] = value

            exporter.export(file_name)
        except Exception as e:
            print(f"Error exporting {plot_data['name']}: {str(e)}")
            return False

    return True


# ----- Виджет OriginalSignalWidget -----
class OriginalSignalWidget(QWidget):
    def __init__(self, signal, parent=None):
        super().__init__(parent)
        self.signal = signal
        self.peakShown = False
        self.peaksCurve = None
        self.init_ui()

    def export_plot(self):
        success = master_export(
            parent=self,
            plots_data=[{"name": "", "plot": self.plot}],
            ask_settings=True,
            default_base_name="Original",
            dialog_title="Export Signal Plot",
        )

    def init_ui(self):
        layout = QVBoxLayout(self)
        self.plot = pg.PlotWidget()
        self.plot.setBackground("#2E2E2E")

        x_axis = calculate_x_axis(self.signal)

        # 2. Построение графика с новой осью X
        self.curve = self.plot.plot(x_axis, self.signal, pen=pg.mkPen('#00FF00', width=2))

        # 3. Обновление подписей осей
        self.plot.setLabel('left', 'Amplitude')
        self.plot.setLabel('bottom', 'Frequency (THz)')
        self.plot.showGrid(x=True, y=True)
        layout.addWidget(self.plot)

        toolbar = QWidget()
        toolbar_layout = QHBoxLayout(toolbar)
        btn_zoom = QPushButton("Zoom")
        self.btn_peaks = QPushButton("Peaks: Off")
        toolbar_layout.addWidget(self.btn_peaks)
        self.btn_peaks.clicked.connect(self.toggle_peaks)
        btn_export = QPushButton("Export Plot")
        toolbar_layout.addWidget(btn_zoom)
        toolbar_layout.addWidget(btn_export)
        layout.addWidget(toolbar)
        btn_zoom.clicked.connect(lambda: self.plot.getViewBox().autoRange())
        btn_export.clicked.connect(self.export_plot)
        self.setLayout(layout)

    def toggle_peaks(self):
        if len(self.signal) == 0:
            return

        x_axis = calculate_x_axis(self.signal)

        if not self.peakShown:
            # --- Метод на основе производных ---
            # 1. Сглаживание сигнала (опционально)
            smoothed_signal = np.convolve(self.signal, np.ones(5) / 5, mode='same')  # Скользящее среднее

            # 2. Расчет первой производной
            gradient = np.gradient(smoothed_signal)

            # 3. Поиск точек, где производная меняет знак с + на -
            peaks = np.where((gradient[:-1] > 0) & (gradient[1:] < 0))[0]

            # 4. Фильтрация слабых пиков (опционально)
            min_peak_height = 0.5 * np.max(self.signal)
            peaks = peaks[self.signal[peaks] > min_peak_height]

            # Отображение пиков
            self.peaksCurve = self.plot.plot(
                x_axis[peaks], self.signal[peaks], pen=None, symbol='x', symbolBrush='r', symbolSize=10
            )
            self.peakShown = True
            self.btn_peaks.setText("Peaks: On")
        else:
            if self.peaksCurve is not None:
                self.plot.removeItem(self.peaksCurve)
            self.peakShown = False
            self.btn_peaks.setText("Peaks: Off")


# ----- Виджет SpectrumWidget -----
class SpectrumWidget(QWidget):
    def __init__(self, mained_signal, sampling_rate, zero_padding=0, parent=None):
        super().__init__(parent)
        self.mained_signal = mained_signal
        self.sampling_rate = sampling_rate
        self.zero_padding = zero_padding
        self.original_signal = None
        self.original_plot = None

        self.crosshair_enabled = False
        self.xf = None
        self.amplitude = None

        self.plot = pg.PlotWidget()
        self.vline = None
        self.hline = None
        self.marker_text = None
        self.curve = None
        self.data_table = self._create_small_table()
        self._init_ui()

    def _init_ui(self):
        main_layout = QVBoxLayout(self)

        # Отображение оригинального сигнала
        self.original_plot = pg.PlotWidget()
        self.original_plot.setBackground("#2E2E2E")
        self.original_plot.setLabel('left', 'Amplitude', color='white')
        self.original_plot.setLabel('bottom', 'Frequency (Hz)', color='white')
        self.original_plot.showGrid(x=True, y=True)
        main_layout.addWidget(self.original_plot)

        main_layout.setContentsMargins(5, 5, 5, 5)

        # График
        self.plot.setBackground("#2E2E2E")
        self.plot.setLabel('left', 'Amplitude', color='white')
        self.plot.setLabel('bottom', 'Frequency (Hz)', color='white')
        self.plot.showGrid(x=True, y=True)
        main_layout.addWidget(self.plot)

        # Таблица данных
        main_layout.addWidget(self.data_table)

        # Панель инструментов внизу
        main_layout.addWidget(self._create_toolbar())

        self._init_crosshair()
        self._calculate_spectrum()
        self._create_data_curve()
        self._reset_crosshair_position()

    def _create_small_table(self):
        table = QTableView()
        table.setMaximumHeight(60)
        table.setFixedWidth(300)
        table.horizontalHeader().setStretchLastSection(True)
        table.verticalHeader().hide()
        table.setStyleSheet(
            """
            QTableView {
                background-color: #2E2E2E;
                color: #FFFFFF;
                border: none;
                font-size: 12px;
            }
            QHeaderView::section {
                background-color: #404040;
                color: #FFFFFF;
                padding: 2px;
            }
        """
        )
        table.hide()
        return table

    def export_plot(self):
        plots_to_export = [
            {"name": "original", "plot": self.original_plot},
            {"name": "spectrum", "plot": self.plot},
        ]
        success = master_export(
            parent=self, plots_data=plots_to_export, ask_settings=True, default_base_name="Data", dialog_title="Export"
        )

    def _create_toolbar(self):
        toolbar = QWidget()
        layout = QHBoxLayout(toolbar)
        layout.setContentsMargins(0, 5, 0, 0)

        self.btn_zoom = QPushButton("Zoom")
        self.btn_marker = QPushButton("Marker")
        self.btn_export = QPushButton("Export")

        button_style = """
            QPushButton {
                background-color: #505050;
                color: white;
                border: none;
                padding: 6px 12px;
                min-width: 80px;
            }
            QPushButton:hover { background-color: #606060; }
            QPushButton:checked { background-color: #707070; }
        """

        for btn in [self.btn_zoom, self.btn_marker, self.btn_export]:
            btn.setStyleSheet(button_style)

        self.btn_marker.setCheckable(True)
        self.btn_marker.setToolTip("Toggle measurement marker")

        layout.addStretch(1)
        layout.addWidget(self.btn_zoom)
        layout.addWidget(self.btn_marker)
        layout.addWidget(self.btn_export)
        layout.addStretch(1)

        self.btn_zoom.clicked.connect(self._reset_zoom)
        self.btn_marker.clicked.connect(self._toggle_marker_mode)
        self.btn_export.clicked.connect(self.export_plot)

        return toolbar

    def _init_crosshair(self):
        line_style = {'color': "#FFA500", 'width': 1, 'style': Qt.DashLine}

        self.vline = pg.InfiniteLine(
            angle=90,
            movable=False,
            pen=pg.mkPen(**line_style),
            hoverPen=pg.mkPen(color="#FF0000", width=2),
            bounds=(0, None),
        )
        self.vline.setVisible(False)

        self.hline = pg.InfiniteLine(angle=0, movable=False, pen=pg.mkPen(**line_style))
        self.hline.setVisible(False)

        self.marker_text = pg.TextItem(
            anchor=(0.5, 0.5), fill=(45, 45, 45, 200), color='#FFFF00', border=pg.mkPen(color='#808080', width=1)
        )
        self.marker_text.setVisible(False)
        self.marker_text.mouseClickEvent = self._toggle_table

        self.plot.addItem(self.vline)
        self.plot.addItem(self.hline)
        self.plot.addItem(self.marker_text)
        self.vline.sigPositionChanged.connect(self._update_crosshair)

    def _toggle_table(self, event):
        self.data_table.setVisible(not self.data_table.isVisible())
        if self.data_table.isVisible():
            self._update_table_data()

    def _update_table_data(self):
        model = QStandardItemModel(1, 2)
        model.setHorizontalHeaderLabels(['Frequency (Hz)', 'Amplitude'])

        freq = self.vline.value()
        idx = np.abs(self.xf - freq).argmin()
        amp = self.amplitude[idx]

        model.setItem(0, 0, QStandardItem(f"{freq:.4f}"))
        model.setItem(0, 1, QStandardItem(f"{amp:.4f}"))

        for col in range(2):
            model.item(0, col).setTextAlignment(Qt.AlignCenter)

        self.data_table.setModel(model)
        self.data_table.resizeColumnsToContents()

    def _calculate_spectrum(self):
        try:
            if len(self.mained_signal) == 0:
                return

            # Вычитание среднего значения из сигнала
            signal_without_dc = self.mained_signal - np.mean(self.mained_signal)

            # Применение zero-padding к обработанному сигналу
            padded_signal = np.pad(signal_without_dc, (0, self.zero_padding), 'constant')

            N = len(padded_signal)
            T = 1.0 / self.sampling_rate
            yf = fft(padded_signal)
            xf = fftfreq(N, T)[: N // 2]
            amplitude = 2.0 / N * np.abs(yf[: N // 2])

            # Убрано удаление первой гармоники ([1:])
            self.xf = xf.copy() if len(xf) > 0 else None
            self.amplitude = amplitude.copy() if len(amplitude) > 0 else None

        except Exception as e:
            print(f"Error calculating spectrum: {e}")
            self.xf = None
            self.amplitude = None

    def _create_data_curve(self):
        if self.curve:
            self.plot.removeItem(self.curve)
        if self.xf is not None and self.amplitude is not None:
            self.curve = self.plot.plot(self.xf, self.amplitude, pen=pg.mkPen('#00FFFF', width=2))

    def _toggle_marker_mode(self):
        self.crosshair_enabled = self.btn_marker.isChecked()
        self._update_marker_controls()

    def _update_marker_controls(self):
        self.vline.setMovable(self.crosshair_enabled)
        self.vline.setVisible(self.crosshair_enabled)
        self.hline.setVisible(self.crosshair_enabled)
        self.marker_text.setVisible(self.crosshair_enabled)
        self.btn_marker.setText("Marker: On" if self.crosshair_enabled else "Marker: Off")
        self.plot.setMouseEnabled(x=True, y=not self.crosshair_enabled)

    def _update_crosshair(self):
        if self.xf is None or self.amplitude is None:
            return

        x_val = self.vline.value()
        idx = np.abs(self.xf - x_val).argmin()
        x_val = self.xf[idx]
        y_val = self.amplitude[idx]

        self.hline.setValue(y_val)
        self._update_marker_text(x_val, y_val)

    def _update_marker_text(self, x, y):
        self.marker_text.setText(f"Freq: {x:.2f} Hz\nAmp: {y:.4f}", color='#FFFF00')
        x_pos = x + (self.xf[-1] - self.xf[0]) * 0.02
        y_pos = self.amplitude.max() * 0.95

        if x_pos > self.xf[-1] * 0.98:
            x_pos = x - (self.xf[-1] - self.xf[0]) * 0.1
            self.marker_text.setAnchor((1, 0.5))
        else:
            self.marker_text.setAnchor((0, 0.5))

        self.marker_text.setPos(x_pos, y_pos)
        if self.data_table.isVisible():
            self._update_table_data()

    def _reset_crosshair_position(self):
        if self.xf is not None and len(self.xf) > 0:
            self.vline.setValue(self.xf[len(self.xf) // 2])
            self._update_crosshair()

    def _reset_zoom(self):
        self.plot.autoRange()

    def update_spectrum(self, new_signal, new_sr, original_signal=None):
        self.mained_signal = new_signal
        self.sampling_rate = new_sr
        self._calculate_spectrum()
        self._create_data_curve()
        self._reset_crosshair_position()
        self.original_signal = original_signal
        # Отрисовка оригинального сигнала
        self.original_plot.clear()
        x_axis = calculate_x_axis(self.original_signal)
        self.original_plot.plot(x_axis, self.original_signal, pen=pg.mkPen('#00FF00', width=2))

    def clear(self):
        self.plot.clear()
        self.xf = None
        self.amplitude = None
        self.curve = None
        self.vline.hide()
        self.hline.hide()
        self.marker_text.hide()
        self.data_table.hide()

    def headerData(self, section, orientation, role):
        if role == Qt.DisplayRole:
            if orientation == Qt.Horizontal:
                return self._data.columns[section]
            return str(section + 1)
        return None


# ----- Виджет PhaseWidget -----
class PhaseWidget(QWidget):
    def __init__(self, time, phase, parent=None):
        super().__init__(parent)
        self.time = time
        self.phase = phase
        self.init_ui()

    def export_plot(self):
        success = master_export(
            parent=self,
            plots_data=[{"name": "", "plot": self.plot}],
            ask_settings=True,
            default_base_name="Data",
            dialog_title="Export Signal Plot",
        )

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
        # btn_pan = QPushButton("Pan")
        btn_export = QPushButton("Export Plot")
        toolbar_layout.addWidget(btn_zoom)
        # toolbar_layout.addWidget(btn_pan)
        toolbar_layout.addWidget(btn_export)
        layout.addWidget(toolbar)

        btn_zoom.clicked.connect(lambda: self.plot.getViewBox().autoRange())
        # btn_pan.clicked.connect(lambda: self.plot.getViewBox().setMouseMode(pg.ViewBox.PanMode))
        btn_export.clicked.connect(self.export_plot)

        self.setLayout(layout)


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
        x_axis = calculate_x_axis(self.signal)

        self.plot_original.plot(x_axis, self.signal, pen=pg.mkPen('#00FF00', width=2))
        self.plot_original.setLabel('left', 'Amplitude', color='white')
        self.plot_original.setLabel('bottom', 'Frequency (THz)', color='white')  # Исправлено!
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
        # self.btn_pan = QPushButton("Pan")
        self.btn_export_plots = QPushButton("Export All Plots")
        main_tb_layout.addWidget(self.btn_zoom)
        # main_tb_layout.addWidget(self.btn_pan)
        main_tb_layout.addWidget(self.btn_export_plots)
        layout.addWidget(self.main_toolbar)

        self.btn_zoom.clicked.connect(self.reset_zoom)
        # self.btn_pan.clicked.connect(lambda: self.plot_spectrum.getViewBox().setMouseMode(1))
        self.btn_export_plots.clicked.connect(self.export_plot)

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
        padded_signal = np.pad(self.original_signal, (n_zeros, 0), 'constant')
        self.signal = padded_signal.copy()
        self.time = np.arange(len(self.signal)) / self.sampling_rate

        # Redraw Original
        self.plot_original.clear()
        # Расчет частоты из длины волн (1410-1490 нм)
        self.plot_original.clear()
        x_axis = calculate_x_axis(self.signal)
        self.plot_original.plot(x_axis, self.signal, pen=pg.mkPen('#00FF00', width=2))

        analytic_signal = hilbert(padded_signal)
        main_signal = np.abs(analytic_signal)
        N = len(main_signal)
        T = 1.0 / self.sampling_rate
        yf = fft(main_signal)
        xf = fftfreq(N, T)[: N // 2]
        amplitude = 2.0 / N * np.abs(yf[: N // 2])

        # remove first harmonic
        if len(xf) > 1:
            xf = xf[1:]
            amplitude = amplitude[1:]

        self.plot_spectrum.clear()
        self.plot_spectrum.plot(xf, amplitude, pen=pg.mkPen('#00FFFF', width=2))

    def export_plot(self):
        plots_to_export = [
            {"name": "original", "plot": self.plot_original},
            {"name": "spectrum", "plot": self.plot_spectrum},
            {"name": "phase", "plot": self.plot_phase},
        ]
        success = master_export(
            parent=self,
            plots_data=plots_to_export,
            ask_settings=True,
            default_base_name="Data",
            dialog_title="Export All Plots",
        )

    def open_original_tab(self):
        if self.signal.size == 0:
            self.window().openOriginalSignalTab(np.array([]), self.sampling_rate)
            return

        self.window().openOriginalSignalTab(self.signal)

    def open_spectrum_tab(self):
        """Открываем диалог ввода частоты дискретизации и вызываем метод главного окна openSpectrumTab(...)"""
        main_win = self.window()
        if not hasattr(main_win, 'data') or main_win.data is None:
            if main_win is not None and hasattr(main_win, 'statusBar'):
                main_win.statusBar().showMessage("No data loaded")
            return

        freq, ok = getStyledDouble(
            self, "Sampling Frequency", "Enter sampling frequency (Hz):", min_val=0.00001, max_val=10000
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
        x_axis = calculate_x_axis(self.signal)
        self.plot_original.setLabel('bottom', 'Frequency (THz)', color='white')

        self.plot_phase.clear()
        if self.time.size and self.instantaneous_phase.size:
            self.plot_phase.plot(self.time, self.instantaneous_phase, pen=pg.mkPen('#00FFFF', width=2))

        self.update_spectrum()


# ----- Главное окно приложения -----
class CustomTabWidget(QTabWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTabsClosable(True)
        self.setMovable(False)

    def tabInserted(self, index):
        super().tabInserted(index)
        if self.tabText(index) == "Main":
            self.tabBar().setTabButton(index, QTabBar.RightSide, None)


class AdvancedInterferometerApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Multi-Beam Interferometer Analyzer")
        self.setGeometry(100, 50, 1280, 800)
        self.setWindowIcon(QIcon('interferometer_icon.png'))

        # Разрешаем Drag & Drop только на главное окно
        self.setAcceptDrops(True)

        # Стили для окна

        self.setStyleSheet(
            """
            QMainWindow { background-color: #2E2E2E; }
            QGroupBox { background-color: #404040; border: 2px solid #606060; border-radius: 5px; margin-top: 1ex; color: #FFFFFF; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 3px; color: #00B4FF; }
            QPushButton { background-color: #505050; border: 1px solid #606060; border-radius: 3px; color: #FFFFFF; padding: 5px; min-width: 80px; }
            QPushButton:hover { background-color: #606060; }
            QLineEdit, QComboBox { background-color: #353535; border: 1px solid #505050; color: #FFFFFF; padding: 3px; }
            QTabWidget::pane { background-color: #2E2E2E; border: 1px solid #606060; margin: 2px; }
            QTabBar::tab { background: #404040; color: #FFFFFF; padding: 8px 15px; border-top-left-radius: 5px; border-top-right-radius: 5px; }
            QTabBar::tab:selected { background: #505050; }
        """
        )

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QHBoxLayout(self.central_widget)

        self.left_panel = QWidget()
        self.left_panel.setFixedWidth(300)

        self.left_panel.setAcceptDrops(False)

        self.left_layout = QVBoxLayout(self.left_panel)
        self.main_layout.addWidget(self.left_panel)

        self.tab_widget = CustomTabWidget()
        self.tab_widget.setTabPosition(QTabWidget.West)
        self.tab_widget.tabCloseRequested.connect(self.closeTab)
        self.main_layout.addWidget(self.tab_widget)

        self.peaks_visible = False
        self.data = None
        self.current_channel = 0
        self.processed_data = None

        # По умолчанию вкладка Main
        self.main_widget = MainWidget(np.array([]))
        self.tab_widget.addTab(self.main_widget, "Main")

        # Создаём оверлей для Drag & Drop (предполагается, что класс DragOverlay уже есть)
        self.drag_overlay = DragOverlay(self.central_widget)
        self.drag_overlay.raise_()

        self.init_data_controls()
        self.init_processing_controls()

    def closeTab(self, index):
        self.tab_widget.removeTab(index)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.drag_overlay.setGeometry(self.central_widget.rect())

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
        if self.rect().contains(self.mapFromGlobal(QCursor.pos())):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragLeaveEvent(self, event):
        if not self.rect().contains(self.mapFromGlobal(QCursor.pos())):
            self.drag_overlay.hideOverlay()
        event.accept()

    def dropEvent(self, event):
        if event.mimeData().hasUrls():
            file_path = event.mimeData().urls()[0].toLocalFile()
            if file_path.lower().endswith(('.txt', '.csv', '.xlsx')):
                self.open_file(file_path)
                self.drag_overlay.hideOverlay()
                event.acceptProposedAction()
                return
        event.ignore()

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
            self, "Open Data File", "", "Text Files (*.txt);;CSV Files (*.csv);;Excel Files (*.xlsx);;All Files (*)"
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
            f"Channels: {len(self.data.columns)}\n" f"Samples: {len(self.data)}\n" f"Type: {self.data.iloc[:, 0].dtype}"
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
        if self.data is None:
            self.statusBar().showMessage("No data to filter")
            return
        try:
            window = int(self.filter_param.text())
            raw = self.data.iloc[:, self.current_channel].values

            if self.filter_type.currentText() == 'Moving Average':
                kernel = np.ones(window) / window
                self.processed_data = np.convolve(raw, kernel, mode='same')
            elif self.filter_type.currentText() == 'Gaussian':
                from scipy.ndimage import gaussian_filter1d

                self.processed_data = gaussian_filter1d(raw, window)
            elif self.filter_type.currentText() == 'Savitzky-Golay':
                from scipy.signal import savgol_filter

                self.processed_data = savgol_filter(raw, window, 3)

            self.statusBar().showMessage("Filter applied successfully")

            self.main_widget.update_signal(self.processed_data, 1000.0)

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

    def openOriginalSignalTab(self, signal):
        for i in range(self.tab_widget.count()):
            if self.tab_widget.tabText(i) == "Original Signal":
                self.tab_widget.setCurrentIndex(i)
                return
        widget = OriginalSignalWidget(signal)
        self.tab_widget.addTab(widget, "Original Signal")
        self.tab_widget.setCurrentWidget(widget)

    def openSpectrumTab(self, mained_signal, sampling_rate):
        for i in range(self.tab_widget.count()):
            if self.tab_widget.tabText(i) == "Spectrum of mained Signal":
                self.tab_widget.setCurrentIndex(i)
                return
        widget = SpectrumWidget(mained_signal, sampling_rate, zero_padding=0)
        widget.update_spectrum(mained_signal, sampling_rate, original_signal=self.main_widget.signal)
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

    def reset_zoom(self):
        self.plot_original.getViewBox().resetTransform()
        self.plot_spectrum.getViewBox().resetTransform()
        self.plot_phase.getViewBox().resetTransform()
        self.btn_zoom.clicked.connect(self.reset_zoom)

    def switchToExistingTab(self, title):
        for i in range(self.tab_widget.count()):
            if self.tab_widget.tabText(i) == title:
                self.tab_widget.setCurrentIndex(i)
                return True
        return False


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setFont(QFont("Helvetica", 10))

    splash = LoadingWindow()
    splash.show()

    window = AdvancedInterferometerApp()

    QTimer.singleShot(800, splash.close)
    QTimer.singleShot(800, window.show)

    sys.exit(app.exec_())
