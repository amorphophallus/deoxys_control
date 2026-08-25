sudo tee /etc/udev/rules.d/77-spacemouse.rules >/dev/null <<'EOF'
KERNEL=="hidraw*", ATTRS{idVendor}=="256f", ATTRS{idProduct}=="c62e", MODE="0666", GROUP="plugdev"
SUBSYSTEM=="usb", ATTRS{idVendor}=="256f", ATTRS{idProduct}=="c62e", MODE="0666", GROUP="plugdev"
KERNEL=="hidraw*", ATTRS{idVendor}=="256f", ATTRS{idProduct}=="c63a", MODE="0666", GROUP="plugdev"
SUBSYSTEM=="usb", ATTRS{idVendor}=="256f", ATTRS{idProduct}=="c63a", MODE="0666", GROUP="plugdev"
KERNEL=="hidraw*", ATTRS{idVendor}=="256f", ATTRS{idProduct}=="c652", MODE="0666", GROUP="plugdev"
SUBSYSTEM=="usb", ATTRS{idVendor}=="256f", ATTRS{idProduct}=="c652", MODE="0666", GROUP="plugdev"
EOF

sudo udevadm control --reload-rules
sudo udevadm trigger --subsystem-match=usb
sudo udevadm trigger --subsystem-match=hidraw
