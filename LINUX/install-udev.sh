#!/usr/bin/env bash
set -euo pipefail

source_rule="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/99-sharkman-uinput.rules"
source_module="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/sharkman-uinput.conf"
target_rule="/etc/udev/rules.d/99-sharkman-uinput.rules"
target_module="/etc/modules-load.d/sharkman-uinput.conf"

if [[ "${EUID}" -ne 0 ]]; then
  echo "Run this one-time installer with sudo: sudo bash LINUX/install-udev.sh" >&2
  exit 2
fi

install -m 0644 "${source_rule}" "${target_rule}"
install -m 0644 "${source_module}" "${target_module}"
modprobe uinput || true
udevadm control --reload-rules
udevadm trigger --name-match=uinput || true
echo "Installed ${target_rule} and ${target_module}. Log out and back in before live input."
