#!/usr/bin/env python3
import sys
import time
import threading
import queue
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Vector3
from std_msgs.msg import Bool, Float32
from sensor_msgs.msg import CompressedImage

from PyQt5.QtWidgets import QApplication, QDialog
from PyQt5.QtCore import Qt, pyqtSignal, QObject, QTimer
from PyQt5.QtGui import QImage, QPixmap
from PyQt5 import uic

import cv2


# ── ROS2 Signals bridge ───────────────────────────────────────────────────────
class ROSSignals(QObject):
    attitude_received = pyqtSignal(float, float, float)
    zoom_received     = pyqtSignal(float)
    gimbal_connected  = pyqtSignal(bool)   # True=connected, False=lost
    cpu_percent       = pyqtSignal(float)
    cpu_total         = pyqtSignal(float)
    mem_percent       = pyqtSignal(float)
    cpu_temp          = pyqtSignal(float)
    wifi_quality      = pyqtSignal(float)
    wifi_signal       = pyqtSignal(float)
    wifi_bandwidth    = pyqtSignal(float)
    eth_bandwidth     = pyqtSignal(float)


# ── ROS2 Node ─────────────────────────────────────────────────────────────────
class ZR10ROSNode(Node):
    def __init__(self, signals: ROSSignals, frame_queue: queue.Queue):
        super().__init__('zr10_gui')
        self.signals     = signals
        self.frame_queue = frame_queue

        # Gimbal connection watchdog
        self.last_attitude_time = 0.0
        self.gimbal_timeout     = 3.0  # seconds without attitude = lost

        # Publishers
        self.pub_center = self.create_publisher(Bool,    '/zr10/gimbal/center', 10)
        self.pub_angle  = self.create_publisher(Vector3, '/zr10/gimbal/angle',  10)
        self.pub_zoom   = self.create_publisher(Float32, '/zr10/gimbal/zoom',   10)

        # Subscribers — gimbal feedback
        self.create_subscription(
            Vector3, '/zr10/gimbal/attitude',   self.cb_attitude,   10)
        self.create_subscription(
            Float32, '/zr10/gimbal/zoom_level', self.cb_zoom_level, 10)

        # Subscribers — video
        self.create_subscription(
            CompressedImage, '/zr10/image_raw/compressed', self.cb_image, 10)

        # Subscribers — system stats
        self.create_subscription(
            Float32, '/zr10/stats/cpu_percent',    self.cb_cpu_percent,    10)
        self.create_subscription(
            Float32, '/zr10/stats/cpu_total',      self.cb_cpu_total,      10)
        self.create_subscription(
            Float32, '/zr10/stats/mem_percent',    self.cb_mem_percent,    10)
        self.create_subscription(
            Float32, '/zr10/stats/cpu_temp',       self.cb_cpu_temp,       10)
        self.create_subscription(
            Float32, '/zr10/stats/wifi_quality',   self.cb_wifi_quality,   10)
        self.create_subscription(
            Float32, '/zr10/stats/wifi_signal',    self.cb_wifi_signal,    10)
        self.create_subscription(
            Float32, '/zr10/stats/wifi_bandwidth', self.cb_wifi_bandwidth, 10)
        self.create_subscription(
            Float32, '/zr10/stats/eth_bandwidth',  self.cb_eth_bandwidth,  10)

        # Gimbal connection watchdog timer — checks every 1 second
        self.create_timer(1.0, self.check_gimbal_connection)

        self.get_logger().info("ZR10 GUI node ready")

    # ── Gimbal watchdog ───────────────────────────────────────────────────────

    def check_gimbal_connection(self):
        if self.last_attitude_time == 0.0:
            # Never received any attitude — not connected yet
            self.signals.gimbal_connected.emit(False)
            return
        elapsed = time.time() - self.last_attitude_time
        self.signals.gimbal_connected.emit(elapsed < self.gimbal_timeout)

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def cb_attitude(self, msg: Vector3):
        self.last_attitude_time = time.time()
        self.signals.attitude_received.emit(msg.x, msg.y, msg.z)

    def cb_zoom_level(self, msg: Float32):
        self.signals.zoom_received.emit(msg.data)

    def cb_image(self, msg: CompressedImage):
        try:
            np_arr = np.frombuffer(msg.data, np.uint8)
            frame  = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if frame is not None:
                if self.frame_queue.full():
                    try:
                        self.frame_queue.get_nowait()
                    except queue.Empty:
                        pass
                self.frame_queue.put_nowait(frame)
        except Exception as e:
            self.get_logger().warn(f"Image decode error: {e}")

    def cb_cpu_percent(self, msg: Float32):
        self.signals.cpu_percent.emit(msg.data)

    def cb_cpu_total(self, msg: Float32):
        self.signals.cpu_total.emit(msg.data)

    def cb_mem_percent(self, msg: Float32):
        self.signals.mem_percent.emit(msg.data)

    def cb_cpu_temp(self, msg: Float32):
        self.signals.cpu_temp.emit(msg.data)

    def cb_wifi_quality(self, msg: Float32):
        self.signals.wifi_quality.emit(msg.data)

    def cb_wifi_signal(self, msg: Float32):
        self.signals.wifi_signal.emit(msg.data)

    def cb_wifi_bandwidth(self, msg: Float32):
        self.signals.wifi_bandwidth.emit(msg.data)

    def cb_eth_bandwidth(self, msg: Float32):
        self.signals.eth_bandwidth.emit(msg.data)

    # ── Publishers ────────────────────────────────────────────────────────────

    def send_center(self):
        msg = Bool()
        msg.data = True
        self.pub_center.publish(msg)

    def send_angle(self, yaw: float, pitch: float):
        msg = Vector3()
        msg.x = float(yaw)
        msg.y = float(pitch)
        msg.z = 0.0
        self.pub_angle.publish(msg)

    def send_zoom(self, level: float):
        msg = Float32()
        msg.data = float(level)
        self.pub_zoom.publish(msg)


# ── Main GUI ──────────────────────────────────────────────────────────────────
class ZR10GUI(QDialog):
    def __init__(self, ros_node: ZR10ROSNode, signals: ROSSignals,
                 frame_queue: queue.Queue):
        super().__init__()
        uic.loadUi('zr10_control.ui', self)

        self.ros_node    = ros_node
        self.signals     = signals
        self.frame_queue = frame_queue

        self.current_yaw   = 0.0
        self.current_pitch = 0.0
        self.current_zoom  = 1.0

        # Track last frame time for no-signal detection
        self.last_frame_time   = 0.0
        self.no_signal_timeout = 3.0

        self._setup_widgets()
        self._connect_signals()
        self._connect_ros_signals()

        # Video timer — polls frame queue at 30fps
        self.video_timer = QTimer()
        self.video_timer.timeout.connect(self.poll_frame)
        self.video_timer.start(33)

        # Video no-signal watchdog — checks every second
        self.watchdog_timer = QTimer()
        self.watchdog_timer.timeout.connect(self.check_video_signal)
        self.watchdog_timer.start(1000)

        self.setWindowTitle("ZR10 Camera Control")

    # ── Widget setup ──────────────────────────────────────────────────────────

    def _setup_widgets(self):
        # z_com_slider: yaw -135 to 135
        self.z_com_slider.setMinimum(-135)
        self.z_com_slider.setMaximum(135)
        self.z_com_slider.setValue(0)

        # y_com_slider: pitch — bottom=-90, top=25
        self.y_com_slider.setMinimum(-90)
        self.y_com_slider.setMaximum(25)
        self.y_com_slider.setValue(0)
        self.y_com_slider.setInvertedAppearance(True)

        # zoom_com_dial: 10-300 (÷10 = 1.0x to 30.0x)
        self.zoom_com_dial.setMinimum(10)
        self.zoom_com_dial.setMaximum(300)
        self.zoom_com_dial.setValue(10)

        # Initial text inputs
        self.des_x.setPlainText("0")
        self.des_y.setPlainText("0")
        self.des_zoom.setPlainText("1.0")

        # Initial states
        self._show_no_signal()
        self._set_gimbal_status(connected=False)

    # ── Signal connections ────────────────────────────────────────────────────

    def _connect_signals(self):
        self.reset_gimbal.clicked.connect(self.on_reset_gimbal)
        self.set_gimbal.clicked.connect(self.on_set_gimbal)
        self.set_zoom.clicked.connect(self.on_set_zoom)
        self.reset_zoom.clicked.connect(self.on_reset_zoom)

        self.z_com_slider.valueChanged.connect(self.on_slider_changed)
        self.y_com_slider.valueChanged.connect(self.on_slider_changed)
        self.zoom_com_dial.valueChanged.connect(self.on_zoom_dial_changed)

    def _connect_ros_signals(self):
        self.signals.attitude_received.connect(self.update_attitude)
        self.signals.zoom_received.connect(self.update_zoom_display)
        self.signals.gimbal_connected.connect(self.update_gimbal_status)
        self.signals.cpu_percent.connect(self.update_cpu_percent)
        self.signals.cpu_total.connect(self.update_cpu_total)
        self.signals.mem_percent.connect(self.update_mem_percent)
        self.signals.cpu_temp.connect(self.update_cpu_temp)
        self.signals.wifi_quality.connect(self.update_wifi_quality)
        self.signals.wifi_signal.connect(self.update_wifi_signal)

    # ── Gimbal status helpers ─────────────────────────────────────────────────

    def _set_gimbal_status(self, connected: bool):
        if connected:
            self.gimbal_control_status.setText("Gimbal Normal")
            self.gimbal_control_status.setStyleSheet("color: green;")
        else:
            self.gimbal_control_status.setText("Lost Connection")
            self.gimbal_control_status.setStyleSheet("color: red;")

    def update_gimbal_status(self, connected: bool):
        # Only update if not in a temporary state (Set/Invalid)
        current = self.gimbal_control_status.text()
        if current not in ("Gimbal Set", "Invalid Input!"):
            self._set_gimbal_status(connected)
        elif connected and current == "Gimbal Set":
            # Keep "Gimbal Set" briefly then revert to Normal
            pass

    # ── No signal helpers ─────────────────────────────────────────────────────

    def _show_no_signal(self):
        self.video_label.clear()
        self.video_label.setText("No Signal")
        self.video_label.setStyleSheet(
            "background-color: black; "
            "color: red; "
            "font-size: 32px; "
            "font-weight: bold;")
        self.video_label.setAlignment(Qt.AlignCenter)

    def _show_video(self):
        self.video_label.setStyleSheet("background-color: black;")
        self.video_label.setText("")

    def check_video_signal(self):
        if self.last_frame_time == 0.0:
            self._show_no_signal()
            return
        if time.time() - self.last_frame_time > self.no_signal_timeout:
            self._show_no_signal()

    # ── Real-time controls ────────────────────────────────────────────────────

    def on_slider_changed(self):
        yaw   = float(self.z_com_slider.value())
        pitch = float(self.y_com_slider.value())
        self.ros_node.send_angle(yaw, pitch)

    def on_zoom_dial_changed(self):
        zoom = self.zoom_com_dial.value() / 10.0
        self.ros_node.send_zoom(zoom)
        self.des_zoom.blockSignals(True)
        self.des_zoom.setPlainText(f"{zoom:.1f}")
        self.des_zoom.blockSignals(False)

    # ── Button handlers ───────────────────────────────────────────────────────

    def on_reset_gimbal(self):
        self.ros_node.send_center()
        self.z_com_slider.blockSignals(True)
        self.y_com_slider.blockSignals(True)
        self.z_com_slider.setValue(0)
        self.y_com_slider.setValue(0)
        self.z_com_slider.blockSignals(False)
        self.y_com_slider.blockSignals(False)
        self.des_x.setPlainText("0")
        self.des_y.setPlainText("0")

    def on_set_gimbal(self):
        try:
            yaw   = float(self.des_x.toPlainText().strip())
            pitch = float(self.des_y.toPlainText().strip())
            yaw   = max(-135.0, min(135.0, yaw))
            pitch = max(-90.0,  min(25.0,  pitch))
            self.ros_node.send_angle(yaw, pitch)

            self.z_com_slider.blockSignals(True)
            self.y_com_slider.blockSignals(True)
            self.z_com_slider.setValue(int(yaw))
            self.y_com_slider.setValue(int(pitch))
            self.z_com_slider.blockSignals(False)
            self.y_com_slider.blockSignals(False)

            # Show "Gimbal Set" briefly then revert to Normal after 2 seconds
            self.gimbal_control_status.setText("Gimbal Set")
            self.gimbal_control_status.setStyleSheet("color: blue;")
            QTimer.singleShot(2000, lambda: self._set_gimbal_status(True))

            print(f"Gimbal set → yaw={yaw:.1f}° pitch={pitch:.1f}°")
        except ValueError:
            self.gimbal_control_status.setText("Invalid Input!")
            self.gimbal_control_status.setStyleSheet("color: red;")
            QTimer.singleShot(2000, lambda: self._set_gimbal_status(
                self.ros_node.last_attitude_time > 0))
            print("Invalid angle input — enter a valid number")

    def on_set_zoom(self):
        try:
            zoom = float(self.des_zoom.toPlainText().strip())
            zoom = max(1.0, min(30.0, zoom))
            self.ros_node.send_zoom(zoom)
            self.zoom_com_dial.blockSignals(True)
            self.zoom_com_dial.setValue(int(zoom * 10))
            self.zoom_com_dial.blockSignals(False)
            print(f"Zoom set → {zoom:.1f}x")
        except ValueError:
            print("Invalid zoom input — enter a number between 1 and 30")

    def on_reset_zoom(self):
        self.ros_node.send_zoom(1.0)
        self.zoom_com_dial.blockSignals(True)
        self.zoom_com_dial.setValue(10)
        self.zoom_com_dial.blockSignals(False)
        self.des_zoom.setPlainText("1.0")
        print("Zoom reset to 1.0x")

    # ── ROS2 feedback updates ─────────────────────────────────────────────────

    def update_attitude(self, yaw: float, pitch: float, roll: float):
        self.current_yaw   = yaw
        self.current_pitch = pitch
        self.curr_x.setText(f"{yaw:.1f}")
        self.curr_y.setText(f"{pitch:.1f}")

    def update_zoom_display(self, zoom: float):
        self.current_zoom = zoom
        self.title_14.setText(f"{zoom:.1f}")

    def update_cpu_percent(self, value: float):
        self.cpu_av_usage.setText(f"Cpu Average Usage: {value:.1f}%")

    def update_cpu_total(self, value: float):
        self.cpu_core_usage.setText(f"Cpu Core Usage: {value:.1f}%")

    def update_mem_percent(self, value: float):
        self.mem_usage.setText(f"Memory Usage: {value:.1f}%")

    def update_cpu_temp(self, value: float):
        color = "color: red;"    if value > 75 else \
                "color: orange;" if value > 65 else \
                "color: green;"
        self.cpu_core_temp.setStyleSheet(color)
        self.cpu_core_temp.setText(f"Cpu Temp: {value:.1f}°C")

    def update_wifi_quality(self, value: float):
        color = "color: red;"    if value < 30 else \
                "color: orange;" if value < 60 else \
                "color: green;"
        self.wifi_quality.setStyleSheet(color)
        self.wifi_quality.setText(f"Wifi Quality: {value:.0f}%")

    def update_wifi_signal(self, value: float):
        color = "color: red;"    if value < -70 else \
                "color: orange;" if value < -60 else \
                "color: green;"
        self.wifi_signal.setStyleSheet(color)
        self.wifi_signal.setText(f"Wifi Signal: {value:.0f}dB")

    # ── Video display ─────────────────────────────────────────────────────────

    def poll_frame(self):
        try:
            frame = self.frame_queue.get_nowait()
            self.update_video(frame)
        except queue.Empty:
            pass

    def update_video(self, frame: np.ndarray):
        self.last_frame_time = time.time()

        if self.video_label.text() == "No Signal":
            self._show_video()

        rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qt_img   = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        pixmap   = QPixmap.fromImage(qt_img)
        self.video_label.setPixmap(
            pixmap.scaled(
                self.video_label.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
        )

    # ── Cleanup ───────────────────────────────────────────────────────────────

    def closeEvent(self, event):
        self.video_timer.stop()
        self.watchdog_timer.stop()
        self.ros_node.send_center()
        self.ros_node.send_zoom(1.0)
        event.accept()


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    rclpy.init()
    signals     = ROSSignals()
    frame_queue = queue.Queue(maxsize=2)
    ros_node    = ZR10ROSNode(signals, frame_queue)

    ros_thread = threading.Thread(
        target=rclpy.spin, args=(ros_node,), daemon=True)
    ros_thread.start()

    app = QApplication(sys.argv)
    gui = ZR10GUI(ros_node, signals, frame_queue)
    gui.show()

    exit_code = app.exec_()

    ros_node.destroy_node()
    rclpy.shutdown()
    sys.exit(exit_code)


if __name__ == '__main__':
    main()